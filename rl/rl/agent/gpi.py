import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim import Adam

from agent.common.agent import MultiTaskAgent
from agent.common.policy import GaussianPolicy, GMMPolicy
from agent.common.util import (
    assert_shape,
    get_sa_pairs,
    get_sa_pairs_,
    grad_false,
    hard_update,
    soft_update,
    update_params,
)
from agent.common.value_function import TwinnedSFNetwork

torch.autograd.set_detect_anomaly(True)  # detect NaN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


class GPIAgent(MultiTaskAgent):
    """GPI
    Andre Barreto, Transfer in Deep Reinforcement Learning Using
        Successor Features and Generalised Policy Improvement
    """

    @classmethod
    def default_config(cls):
        return dict(
            env="InvertedDoublePendulum-v4",
            env_kwargs={},
            total_timesteps=int(1e5),
            reward_scale=1,
            gamma=0.99,
            tau=5e-3,
            lr=5e-4,
            policy_lr=5e-3,
            alpha_lr=3e-4,
            alpha=1.0,
            net_kwargs={"value_sizes": [64, 64], "policy_sizes": [32, 32]},
            policy_class="GMM",
            policy_regularization=1e-3,
            n_gauss=5,
            mini_batch_size=256,
            replay_buffer_size=1e6,
            multi_step=1,
            updates_per_step=1,
            min_n_experience=int(1024),
            td_target_update_interval=1,
            generate_task_schedule=False,
            task_schedule_stepsize=0.5,
            new_task_threshold=1.0,
            grad_clip=None,
            render=False,
            log_interval=10,
            entropy_tuning=True,
            eval_interval=100,
            seed=0,
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
        self.pol_class = config.get("policy_class", "GMM")
        self.pol_reg = config.get("policy_regularization", 1e-3)
        self.n_gauss = config.get("n_gauss", 5)

        self.net_kwargs = config["net_kwargs"]
        self.updates_per_step = config.get("updates_per_step", 1)
        self.grad_clip = config.get("grad_clip", None)
        self.min_n_experience = self.start_steps = int(
            config.get("min_n_experience", int(1e4))
        )
        self.entropy_tuning = config.get("entropy_tuning", True)
        self.eval_interval = config.get("eval_interval", 1000)

        self.cur_idx = -1
        self.sfs, self.sf_tars, self.sf_optims = [], [], []
        self.pols, self.pol_optims = [], []

        self.prev_ws = []
        self.task_diff = config.get("new_task_threshold", 1.0)
        self.task_schedule = config.get("generate_task_schedule", False)
        if self.task_schedule:
            self.w_schedule = self.generate_schedule(
                config.get("task_schedule_stepsize", 1.0)
            )
            self.w = self.w_schedule[0]
        self.update_task(self.w)
        self.ten_episode_pol_loss = []

        # if self.entropy_tuning:
        #     self.alpha_lr = config.get("alpha_lr", 3e-4)
        #     # target entropy = -|A|
        #     self.target_entropy = -torch.prod(
        #         torch.Tensor(self.env.action_space.shape).to(device)
        #     ).item()
        #     # optimize log(alpha), instead of alpha
        #     self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        #     self.alpha = self.log_alpha.exp()
        #     self.alpha_optimizer = Adam([self.log_alpha], lr=self.alpha_lr)
        # else:
        #     self.alpha = torch.tensor(config.get("alpha", 0.2)).to(device)
        self.alpha = torch.tensor(config.get("alpha", 0.2)).to(device)

    def run(self):
        return super().run()

    def train_episode(self):
        if self.task_schedule:
            self.change_task()
        return super().train_episode()

    def act(self, obs):
        return super().act(obs)

    def get_action(self, obs):
        w = self.w

        obs_ts = self.np2ts(obs)
        w_ts = self.np2ts(w)

        act_ts = self.calc_actions(obs_ts, w_ts)
        act_ts = act_ts.squeeze(1)

        act = self.ts2np(act_ts)
        assert act.shape == (self.action_dim,)
        return act

    def calc_actions(self, obs, w):
        acts = []
        qs = []
        for idx in range(len(self.sfs)):
            act, _ = self.pols[idx](obs)
            sf0, sf1 = self.sfs[idx](obs, act)
            sf = torch.min(sf0, sf1)
            q = torch.sum(w * sf, 1)

            acts.append(act)
            qs.append(q)

        qs = torch.tensor(qs)
        pol_idx = torch.argmax(qs)
        act = acts[pol_idx]

        return act

    def learn(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            for i in range(len(self.sfs)):
                soft_update(self.sf_tars[i], self.sfs[i], self.tau)

        batch = self.replay_buffer.sample(self.mini_batch_size)

        i = self.cur_idx
        sf_optim, sf, sf_tar = self.sf_optims[i], self.sfs[i], self.sf_tars[i]
        pol_optim, pol = self.pol_optims[i], self.pols[i]

        sf_loss, mean_sf = self.calc_sf_loss(batch, sf, sf_tar, pol)
        pol_loss = self.calc_pol_loss(batch, sf_tar, pol)

        update_params(sf_optim[0], sf.SF0, sf_loss[0], self.grad_clip)
        update_params(sf_optim[1], sf.SF1, sf_loss[1], self.grad_clip)
        update_params(pol_optim, pol, pol_loss, self.grad_clip)

        self.ten_episode_pol_loss.append(pol_loss)
        if len(self.ten_episode_pol_loss) > 10:
            self.ten_episode_pol_loss.pop(0)

        if self.learn_steps % self.log_interval == 0:
            metrics = {
                f"loss/SF": sf_loss[0].detach().item(),
                f"loss/policy": pol_loss.detach().item(),
                f"state/mean_SF": mean_sf[0],
            }
            wandb.log(metrics)

    def calc_sf_loss(self, batch, sf, sf_target, pol):
        (states, features, actions, rewards, next_states, dones) = batch

        cur_sf0, cur_sf1 = sf(states, actions)
        cur_sf0, cur_sf1 = cur_sf0.squeeze(), cur_sf1.squeeze()

        target_sf, next_sfv = self.calc_target_sf(
            next_states, features, dones, sf_target, pol
        )

        sf0_loss = F.mse_loss(cur_sf0, target_sf)
        sf1_loss = F.mse_loss(cur_sf1, target_sf)

        mean_sf0 = cur_sf0.detach().mean().item()
        mean_sf1 = cur_sf1.detach().mean().item()

        return (sf0_loss, sf1_loss), (mean_sf0, mean_sf1)

    def calc_target_sf(self, next_states, features, dones, sf_target, pol):
        with torch.no_grad():
            n_sample = next_states.shape[0]
            next_actions, _ = pol(next_states)
            next_sf0, next_sf1 = sf_target(next_states, next_actions)
            next_sf0 = next_sf0.view(n_sample, -1, self.feature_dim)
            next_sf1 = next_sf1.view(n_sample, -1, self.feature_dim)

            next_sf = torch.min(next_sf0, next_sf1)
            next_sf /= self.alpha

            next_sfv = torch.logsumexp(next_sf, 1)
            next_sfv *= self.alpha

            target_sf = (features + (1 - dones) * self.gamma * next_sfv).squeeze(1)
        return target_sf, next_sfv

    def calc_pol_loss(self, batch, sf_target, pol):
        (states, _, _, _, _, _) = batch
        act, logp = pol(states)

        sf0, sf1 = sf_target(states, act)
        sf = torch.min(sf0, sf1)
        q_hat = torch.sum(self.w * sf, 1)
        q_hat = q_hat.detach()

        reg_loss = self.reg_loss(pol)

        loss = torch.mean(logp - 1 / self.alpha * q_hat) + reg_loss
        return loss

    def reg_loss(self, pol):
        reg = getattr(pol, "reg_loss", None)
        if callable(reg):
            loss = pol.reg_loss()
        else:
            loss = 0
        return loss

    def generate_schedule(self, gap):
        l = []
        for _ in range(self.feature_dim):
            x = np.arange(0.0, 1.0 + gap, gap)
            l.append(x)
        g = np.meshgrid(*l)
        rgrid = []
        for i in range(len(g)):
            rgrid.append(g[i].reshape(-1, 1))
        grid = np.concatenate(rgrid, 1)
        grid = np.delete(grid, 0, 0)
        return grid

    def change_task(self):
        """if a current task is mastered then update task according to schedule"""
        if self.master_curtask():
            w = self.w_schedule[self.cur_idx + 1]
            self.update_task(w)

    def master_curtask(self):
        master = False

        var = np.array(self.ten_episode_pol_loss).var()
        if var < 1 and len(self.ten_episode_pol_loss) == 10:
            master = True

        return master

    def update_task(self, w):
        """update task and create corresponding policy and successor feature

        Args:
            w (np.array): task
        """
        self.update_w(w)

        if self.create_new_pol(w):
            self.prev_ws.append(w)
            self.create_sfpol()

    def update_w(self, w):
        self.w = torch.tensor(w, dtype=torch.float32).to(device)
        self.env.w = w
        assert self.w.shape == (self.feature_dim,)

    def create_new_pol(self, w):
        """should we create new set of policies?"""
        b = False

        if len(self.prev_ws) == 0:
            b = True

        for prev_w in self.prev_ws:
            l = np.linalg.norm(w - prev_w)
            if l > self.task_diff:
                b = True

        return b

    def create_sfpol(self):
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

        if self.pol_class == "Gaussian":
            policy = GaussianPolicy(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                sizes=self.net_kwargs.get("policy_sizes", [64, 64]),
            ).to(device)
        elif self.pol_class == "GMM":
            policy = GMMPolicy(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                sizes=self.net_kwargs.get("policy_sizes", [64, 64]),
                n_gauss=self.n_gauss,
                reg=self.pol_reg,
            ).to(device)

        sf_optimizer = (
            Adam(sf.SF0.parameters(), lr=self.lr),
            Adam(sf.SF1.parameters(), lr=self.lr),
        )
        policy_optimizer = Adam(policy.parameters(), lr=self.pol_lr)

        self.sfs.append(sf)
        self.sf_tars.append(sf_target)
        self.sf_optims.append(sf_optimizer)
        self.pols.append(policy)
        self.pol_optims.append(policy_optimizer)
        self.cur_idx += 1


if __name__ == "__main__":
    # ============== profile ==============#
    # pip install snakeviz
    # python -m cProfile -o out.profile rl/rl/agent/gpi.py -s time
    # snakeviz gpi.profile

    # ============== sweep ==============#
    # wandb sweep rl/rl/sweep_gpi.yaml
    import benchmark_env
    import gym

    env = "myInvertedDoublePendulum-v4"
    env_kwargs = {}
    render = True

    default_dict = GPIAgent.default_config()
    default_dict.update(
        dict(
            env=env,
            env_kwargs=env_kwargs,
            render=render,
        )
    )

    wandb.init(config=default_dict)
    print(wandb.config)

    agent = GPIAgent(wandb.config)
    agent.run()
