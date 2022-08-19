import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim import Adam

from agent.common.agent import MultiTaskAgent
from agent.common.policy import GaussianPolicy, GMMPolicy
from agent.common.util import (
    grad_false,
    hard_update,
    soft_update,
    update_params,
    ts2np,
    generate_grid_schedule,
)
from agent.common.value_function import TwinnedSFNetwork

# disable api to speed up
torch.autograd.set_detect_anomaly(False)  # detect NaN
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


class SFGPIAgent(MultiTaskAgent):
    """GPI
    Andre Barreto, Transfer in Deep Reinforcement Learning Using
        Successor Features and Generalised Policy Improvement
    """

    @classmethod
    def default_config(cls):
        return dict(
            env="InvertedDoublePendulum-v4",
            env_kwargs={},
            total_timesteps=int(1e6),
            reward_scale=1,
            tau=5e-3,
            alpha=0.2,
            lr=1e-4,
            policy_lr=5e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            policy_regularization=1e-3,
            n_gauss=2,
            multi_step=1,
            updates_per_step=1,
            mini_batch_size=128,
            min_n_experience=int(1024),
            replay_buffer_size=1e6,
            td_target_update_interval=1,
            grad_clip=None,
            render=False,
            log_interval=10,
            entropy_tuning=False,
            alpha_bnd=10,
            eval=True,
            eval_interval=100,
            seed=0,
            net_kwargs={"value_sizes": [64, 64], "policy_sizes": [32, 32]},
            task_schedule=None,
            task_manual=None,
            n_rec_sfloss=40,
            task_schedule_stepsize=1.0,
            new_task_threshold_sfloss=30,
        )

    def __init__(
        self,
        config: dict = {},
    ) -> None:
        self.config = config
        super().__init__(**config)

        self.lr = config.get("lr", 1e-3)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 5e-3)
        self.td_target_update_interval = int(config.get("td_target_update_interval", 1))

        self.pol_lr = config.get("policy_lr", 1e-3)
        self.pol_reg = config.get("policy_regularization", 1e-3)
        self.n_gauss = config.get("n_gauss", 5)

        self.net_kwargs = config["net_kwargs"]
        self.grad_clip = config.get("grad_clip", None)
        self.entropy_tuning = config.get("entropy_tuning", True)

        self.policy_idx = -1
        self.sfs, self.sf_tars, self.sf_optims = [], [], []
        self.pols, self.pol_optims = [], []

        self.task_idx = 0
        self.prev_ws = []

        self.task_schedule = config.get("task_schedule", "manual")
        if self.task_schedule == "manual":
            task_manual = config.get("task_manual")
            self.w_schedule = task_manual * self.feature_scale
            self.w = self.w_schedule[self.task_idx]
        elif self.task_schedule == "grid":
            self.task_schedule_stepsize = config.get("task_schedule_stepsize", 1.0)
            self.w_schedule = generate_grid_schedule(
                self.task_schedule_stepsize, self.feature_dim, self.feature_scale
            )
            self.w = self.w_schedule[self.task_idx]

        self.update_task(self.w)
        self.learn_all_tasks = False
        self.n_sf_losses = []
        self.n_rec_sfloss = config.get("n_rec_sfloss", 40)
        self.task_thresh_sfloss = config.get("new_task_threshold_sfloss", 40)

        if self.entropy_tuning:
            self.alpha_lr = config.get("alpha_lr", 3e-4)
            # target entropy = -|A|
            self.target_entropy = -torch.prod(
                torch.Tensor(self.env.action_space.shape).to(device)
            ).item()
            # optimize log(alpha), instead of alpha
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_bnd = config.get("alpha_bnd", True)
            self.alpha = self.log_alpha.exp()
            self.alpha = torch.clip(self.alpha, -self.alpha_bnd, self.alpha_bnd)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor(config.get("alpha", 0.2)).to(device)

    def run(self):
        while True:
            self.train_episode()
            if self.learn_all_tasks:
                break
            if self.steps > self.total_timesteps:
                break

    def train_episode(self):
        if self.task_schedule is not None:
            self.learn_all_tasks = self.change_task()

        super().train_episode()

    def evaluate(self):
        return super().evaluate()

    def explore(self, obs, w):
        acts, qs = self.gpe(obs, w, "explore")
        act = self.gpi(acts, qs)
        return act

    def exploit(self, obs, w):
        acts, qs = self.gpe(obs, w, "exploit")
        act = self.gpi(acts, qs)
        return act

    def gpe(self, obs, w, mode):
        acts, qs = [], []
        for i in range(len(self.sfs)):
            if mode == "explore":
                act, _, _ = self.pols[i](obs)
            if mode == "exploit":
                _, _, act = self.pols[i](obs)
            sf0, sf1 = self.sfs[i](obs, act)
            sf = torch.min(sf0, sf1)
            q = torch.sum(w * sf, 1)

            acts.append(act)
            qs.append(q)
        return acts, qs

    def gpi(self, acts, qs):
        qs = torch.tensor(qs)
        pol_idx = torch.argmax(qs)
        act = acts[pol_idx].squeeze(1)
        return act

    def learn(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            for i in range(len(self.sfs)):
                soft_update(self.sf_tars[i], self.sfs[i], self.tau)

        batch = self.replay_buffer.sample(self.mini_batch_size)

        i = self.policy_idx
        sf_optim, sf, sf_tar = self.sf_optims[i], self.sfs[i], self.sf_tars[i]
        pol_optim, pol = self.pol_optims[i], self.pols[i]

        sf_loss, mean_sf, target_sf = self.calc_sf_loss(batch, sf, sf_tar, pol)
        pol_loss, logp = self.calc_pol_loss(batch, sf_tar, pol)

        update_params(sf_optim[0], sf.SF0, sf_loss[0], self.grad_clip)
        update_params(sf_optim[1], sf.SF1, sf_loss[1], self.grad_clip)
        update_params(pol_optim, pol, pol_loss, self.grad_clip)

        if self.learn_steps % self.log_interval == 0:
            sf_loss_var = self.nsf_losses(sf_loss, self.n_rec_sfloss)

            metrics = {
                "loss/SF0": sf_loss[0].detach().item(),
                "loss/SF1": sf_loss[1].detach().item(),
                "loss/policy": pol_loss.detach().item(),
                "loss/sfloss_var": sf_loss_var,
                "state/mean_SF0": mean_sf[0],
                "state/mean_SF1": mean_sf[1],
                "state/target_sf": target_sf.detach().mean().item(),
                "state/logp": logp.detach().mean().item(),
                "task/task_idx": self.task_idx,
                "task/policy_idx": self.policy_idx,
            }

            w = ts2np(self.w)
            for i in range(w.shape[0]):
                metrics[f"task/{i}"] = w[i]

            wandb.log(metrics)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(logp)
            update_params(self.alpha_optimizer, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
            self.alpha = torch.clip(self.alpha, -self.alpha_bnd, self.alpha_bnd)
            wandb.log(
                {
                    "loss/alpha": entropy_loss.detach().item(),
                    "state/alpha": self.alpha.detach().item(),
                }
            )

    def calc_sf_loss(self, batch, sf, sf_target, pol):
        (observations, features, actions, rewards, next_observations, dones) = batch

        cur_sf0, cur_sf1 = sf(observations, actions)
        cur_sf0, cur_sf1 = cur_sf0.squeeze(), cur_sf1.squeeze()

        target_sf, next_sfv = self.calc_target_sf(
            next_observations, features, dones, sf_target, pol
        )

        mean_sf0 = cur_sf0.detach().mean().item()
        mean_sf1 = cur_sf1.detach().mean().item()

        assert cur_sf0.shape == target_sf.shape

        sf0_loss = F.mse_loss(cur_sf0, target_sf)
        sf1_loss = F.mse_loss(cur_sf1, target_sf)

        return (sf0_loss, sf1_loss), (mean_sf0, mean_sf1), target_sf

    def calc_pol_loss(self, batch, sf_tar, pol):
        (observations, _, _, _, _, _) = batch
        act, logp, _ = pol(observations)

        sf0, sf1 = sf_tar(observations, act)
        q_hat0, q_hat1 = torch.sum(self.w * sf0, 1), torch.sum(self.w * sf1, 1)
        q_hat = torch.min(q_hat0, q_hat1)
        q_hat *= self.reward_scale

        assert logp.shape == q_hat.shape

        reg_loss = self.reg_loss(pol)
        loss = torch.mean(self.alpha * logp - q_hat) + reg_loss
        return loss, logp

    def calc_entropy_loss(self, entropy):
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
        )
        return entropy_loss

    # TODO: slow
    def calc_target_sf(self, next_observations, features, dones, sf_target, pol):
        with torch.no_grad():
            n_sample = next_observations.shape[0]
            next_actions, _, _ = pol(next_observations)  # TODO: slow
            next_sf0, next_sf1 = sf_target(next_observations, next_actions)

            next_sf0 = next_sf0.view(n_sample, -1, self.feature_dim)
            next_sf1 = next_sf1.view(n_sample, -1, self.feature_dim)

            next_sf = torch.min(next_sf0, next_sf1)
            next_sfv = torch.logsumexp(next_sf, 1)

            assert features.shape == next_sfv.shape

            target_sf = (features + (1 - dones) * self.gamma * next_sfv).squeeze(1)
        return target_sf, next_sfv

    def reg_loss(self, pol):
        reg = getattr(pol, "reg_loss", None)
        loss = pol.reg_loss() if callable(reg) else 0
        return loss

    def nsf_losses(self, sf_loss, n_record):
        sf_loss = torch.min(sf_loss[0], sf_loss[1])
        self.n_sf_losses.append(ts2np(sf_loss))
        if len(self.n_sf_losses) > n_record:
            self.n_sf_losses.pop(0)
        return np.array(self.n_sf_losses).var()

    def create_policy(self):
        """create new set of policies and successor features"""
        sf = TwinnedSFNetwork(
            observation_dim=self.observation_dim,
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            sizes=self.net_kwargs.get("value_sizes", [64, 64]),
        ).to(device)
        sf_target = (
            TwinnedSFNetwork(
                observation_dim=self.observation_dim,
                feature_dim=self.feature_dim,
                action_dim=self.action_dim,
                sizes=self.net_kwargs.get("value_sizes", [64, 64]),
            )
            .to(device)
            .eval()
        )
        hard_update(sf_target, sf)
        grad_false(sf_target)
        sf_optimizer = (
            Adam(sf.SF0.parameters(), lr=self.lr),
            Adam(sf.SF1.parameters(), lr=self.lr),
        )

        policy = GMMPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sizes=self.net_kwargs.get("policy_sizes", [64, 64]),
            n_gauss=self.n_gauss,
            reg=self.pol_reg,
        ).to(device)
        policy_optimizer = Adam(policy.parameters(), lr=self.pol_lr)

        self.sfs.append(sf)
        self.sf_tars.append(sf_target)
        self.sf_optims.append(sf_optimizer)
        self.pols.append(policy)
        self.pol_optims.append(policy_optimizer)
        self.policy_idx += 1
        self.n_sf_losses = []

    def change_task(self):
        """if a current task is mastered then update task according to schedule"""
        if self.master_curtask():

            if self.task_idx < len(self.w_schedule):
                self.task_idx += 1
                w = self.w_schedule[self.task_idx]
                self.update_task(w)
            else:
                print("no more task to learn")
                return True

        return None

    def master_curtask(self):
        master = False

        var = np.array(self.n_sf_losses).var()
        if var < self.task_thresh_sfloss and len(self.n_sf_losses) >= self.n_rec_sfloss:
            master = True

        return master

    def update_task(self, w):
        """update task and create corresponding policy and successor feature

        Args:
            w (np.array): task
        """
        self.update_w(w)

        if self.create_new_pol(w):
            self.create_policy()

    def update_w(self, w):
        self.w = torch.tensor(w, dtype=torch.float32).to(device)
        self.env.w = w
        self.prev_ws.append(w)
        assert self.w.shape == (self.feature_dim,)

    def create_new_pol(self, w):
        """should we create new set of policies"""
        b = False

        if self.policy_idx == -1:
            b = True

        for prev_w in self.prev_ws:
            l = np.linalg.norm(w / self.feature_scale - prev_w / self.feature_scale)
            if l > 1.0:
                b = True

        return b


if __name__ == "__main__":
    # ============== profile ==============#
    # pip install snakeviz
    # python -m cProfile -o sfgpi.profile rl/rl/agent/sfgpi.py -s time
    # snakeviz sfgpi.profile

    # ============== sweep ==============#
    # wandb sweep rl/rl/sweep_sfgpi.yaml
    import benchmark_env
    import drone_env
    import gym
    from drone_env.envs.script import close_simulation

    env = "multitaskpid-v0"
    # env = "myInvertedDoublePendulum-v4"
    render = True
    auto_start_simulation = False
    if auto_start_simulation:
        close_simulation()

    if env == "multitaskpid-v0":
        env_config = {
            "DBG": False,
            "duration": 1000,
            "simulation": {
                "gui": render,
                "enable_meshes": True,
                "auto_start_simulation": auto_start_simulation,
                # "position": (0, 0, 25),  # initial spawned position
            },
            "observation": {
                "DBG_ROS": False,
                "DBG_OBS": False,
                "noise_stdv": 0.0,
                "scale_obs": True,
                "include_raw_state": False,
                "bnd_constraint": True,
            },
            "action": {
                "DBG_ACT": False,
                "act_noise_stdv": 0.0,
                "thrust_range": [-0.25, 0.25],
            },
            "target": {
                "DBG_ROS": False,
            },
            "tasks": {
                "tracking": {
                    "ori_diff": np.array([0.0, 0.0, 0.0, 0.0]),
                    "ang_diff": np.array([0.0, 0.0, 0.0]),
                    "angvel_diff": np.array([0.0, 0.0, 0.0]),
                    "pos_diff": np.array([0.0, 0.0, 1.0]),
                    "vel_diff": np.array([0.0, 0.0, 0.0]),
                    "vel_norm_diff": np.array([0.0]),
                },
                "constraint": {
                    "action_cost": np.array([0.001]),
                    "pos_ubnd_cost": np.array([0.0, 0.0, 0.1]),  # x,y,z
                    "pos_lbnd_cost": np.array([0.0, 0.0, 0.1]),  # x,y,z
                },
                "success": {"pos": np.array([0.0, 0.0, 10.0])},  # x,y,z
            },
        }
        env_kwargs = {"config": env_config}
    else:
        env = {}

    # task_schedule = "Manual"
    # task_manual = np.array([np.array([0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1])])
    task_schedule = None
    task_manual = None

    default_dict = SFGPIAgent.default_config()
    default_dict.update(
        dict(
            env=env,
            env_kwargs=env_kwargs,
            render=render,
            task_schedule=task_schedule,
            task_manual=task_manual,
        )
    )

    wandb.init(config=default_dict)
    print(wandb.config)

    agent = SFGPIAgent(wandb.config)
    agent.run()
