import datetime
from pathlib import Path
import os.path
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from common.policy import StochasticPolicy
from common.kernel import adaptive_isotropic_gaussian_kernel
from common.replay_buffer import ReplayBuffer
from common.util import assert_shape, get_sa_pairs, get_sa_pairs_
from common.value_function import SFMLP
from common.agent import AbstractAgent
import copy
import pprint

pp = pprint.PrettyPrinter(depth=4)
torch.autograd.set_detect_anomaly(True)  # detect NaN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


class SFGPI(AbstractAgent):
    @classmethod
    def default_config(cls):
        return dict(
            total_timesteps=5e4,
            n_timesteps=200,
            reward_scale=1,
            replay_buffer_size=1e6,
            mini_batch_size=256,
            min_n_experience=2048,
            lr=5e-4,
            alpha=1.0,
            gamma=0.99,
            td_target_update_interval=1,
            value_n_particles=96,
            policy_lr=5e-3,
            action_n_particles=64,
            kernel_update_ratio=0.5,
            render=False,
            net_kwargs={"value_sizes": [32, 32], "policy_sizes": [16, 16]},
            env_kwargs={},
        )

    def __init__(
        self,
        env,
        lr: float = 1e-3,
        gamma: float = 0.99,
        alpha: float = 8,
        td_target_update_interval: int = 1,
        value_n_particles: int = 16,
        policy_lr: float = 1e-3,
        action_n_particles: int = 16,
        kernel_update_ratio: float = 0.5,
        replay_buffer=ReplayBuffer,
        tasks: dict = None,
        net_kwargs: dict = {},
        env_kwargs: dict = {},
        config: dict = {},
        **kwargs,
    ) -> None:
        self.config = config
        super().__init__(env=env, env_kwargs=env_kwargs, **kwargs)

        self.learn_steps = 0
        self.epoch = 0

        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.begin_learn_td = False
        self.td_target_update_interval = int(td_target_update_interval)
        self.value_n_particles = int(value_n_particles)

        self.policy_lr = policy_lr
        self.kernel_fn = adaptive_isotropic_gaussian_kernel
        self.action_n_particles = int(action_n_particles)
        self.kernel_update_ratio = kernel_update_ratio

        self.net_kwargs = net_kwargs
        self.sf_dim = self.env.feature_len

        self.behavior_sfs = []
        self.target_sfs = []
        self.policy_nets = []
        self.optimizers = []
        self.policy_optimizers = []
        self.n_policies = 0
        self.add_policy()

        self.replay_buffer = replay_buffer(self.replay_buffer_size)

        if tasks is not None:
            self.set_tasks(tasks)

    def train(self):
        episode_reward = 0
        n_epochs = int(self.total_timesteps / self.n_timesteps)

        for epoch in range(self.epoch, n_epochs):
            self.epoch = epoch
            state = self.env.reset()
            phi = self.env.get_features()

            episode_reward = 0

            for _ in range(self.n_timesteps):
                tasks = self.env.get_tasks_weights()
                action = self.gpi_policy(state, tasks)

                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                self.replay_buffer.add(
                    (state, next_state, phi, action, reward, done, tasks)
                )

                if self.replay_buffer.size() > self.min_n_experience:
                    batch = self.sample_minibatch()
                    self.td_update(batch)
                    self.policy_update(batch)

                if done:
                    break

                state = next_state
                phi = info["features"]

            wandb.log({"episode_reward": episode_reward})

    def td_update(self, batch):

        if self.begin_learn_td is False:
            print("begin learning q function")
            self.begin_learn_td = True

        if self.learn_steps % self.td_target_update_interval == 0:
            self.update_target_sfs()

        self.learn_steps += 1

        [self.update_sfs(batch, i) for i in range(len(self.behavior_sfs))]

    def policy_update(self, batch):
        [self.update_policies(batch, i) for i in range(len(self.policy_nets))]

    def update_policies(self, batch, policy_idx):
        (
            batch_state,
            batch_next_state,
            batch_phi,
            batch_action,
            batch_reward,
            batch_done,
            batch_task,
        ) = batch

        actions = self.policy_nets[policy_idx].acitons_for(
            observations=batch_state, n_action_samples=self.action_n_particles
        )
        assert_shape(actions, [None, self.action_n_particles, self.action_dim])

        n_updated_actions = int(self.action_n_particles * self.kernel_update_ratio)
        n_fixed_actions = self.action_n_particles - n_updated_actions
        fixed_actions, updated_actions = torch.split(
            actions, [n_fixed_actions, n_updated_actions], dim=1
        )
        assert_shape(fixed_actions, [None, n_fixed_actions, self.action_dim])
        assert_shape(updated_actions, [None, n_updated_actions, self.action_dim])

        s_a = get_sa_pairs_(batch_state, fixed_actions)
        tasks = torch.tensor(self.env.get_tasks_weights()).float().to(device)
        svgd_target_values = self.compute_q_target_from_policy_distr(
            s_a, tasks, policy_idx
        ).view(batch_state.shape[0], -1)
        assert_shape(svgd_target_values, [None, n_fixed_actions])

        squash_correction = torch.sum(torch.log(1 - fixed_actions**2 + EPS), dim=-1)
        log_p = svgd_target_values + squash_correction

        grad_log_p = torch.autograd.grad(
            log_p, fixed_actions, grad_outputs=torch.ones_like(log_p)
        )
        grad_log_p = grad_log_p[0].unsqueeze(2).detach()
        assert_shape(grad_log_p, [None, n_fixed_actions, 1, self.action_dim])

        # kernel function in Equation 13:
        kernel_dict = self.kernel_fn(xs=fixed_actions, ys=updated_actions)
        kappa = kernel_dict["output"].unsqueeze(3)
        assert_shape(kappa, [None, n_fixed_actions, n_updated_actions, 1])

        # stein variational gradient in equation 13:
        action_gradients = torch.mean(
            kappa * grad_log_p + self.alpha * kernel_dict["gradient"], dim=1
        )
        assert_shape(action_gradients, [None, n_updated_actions, self.action_dim])

        # Propagate the gradient through the policy network (Equation 14). action_gradients * df_{\phi}(.,s)/d(\phi)
        gradients = torch.autograd.grad(
            updated_actions,
            self.policy_nets[policy_idx].parameters(),
            grad_outputs=action_gradients,
        )

        # multiply weight since optimizer will differentiate it later so we can apply the gradient
        surrogate_loss = torch.sum(
            torch.stack(
                [
                    torch.sum(w * g.detach())
                    for w, g in zip(
                        self.policy_nets[policy_idx].parameters(), gradients
                    )
                ],
                dim=0,
            )
        )
        assert surrogate_loss.requires_grad == True

        self.policy_optimizers[policy_idx].zero_grad()
        loss = -surrogate_loss
        loss.backward()
        self.policy_optimizers[policy_idx].step()

        metrics = {
            "policy_loss_" + str(policy_idx): loss,
            "svgd_target_" + str(policy_idx): svgd_target_values.mean(),
        }
        wandb.log(metrics)

    def update_target_sfs(self):
        [
            self.target_sfs[i].load_state_dict(self.behavior_sfs[i].state_dict())
            for i in range(len(self.target_sfs))
        ]

    def update_sfs(self, batch, policy_idx):

        (
            batch_state,
            batch_next_state,
            batch_phi,
            batch_action,
            batch_reward,
            batch_done,
            batch_task,
        ) = batch
        batch_reward *= self.reward_scale

        target_psi = self.compute_target_psi_from_uniform_distr(
            batch_next_state,
            policy_idx,
        )
        target_psi /= self.alpha

        assert_shape(target_psi, [None, self.sf_dim, self.value_n_particles])

        nextPsi = torch.logsumexp(target_psi, 2)
        assert_shape(nextPsi, [None, self.sf_dim])

        nextPsi -= np.log(self.value_n_particles)
        nextPsi += self.action_dim * np.log(2)
        nextPsi *= self.alpha

        psiTarget = (batch_phi + (1 - batch_done) * self.gamma * nextPsi).detach()
        assert_shape(psiTarget, [None, self.sf_dim])

        psi = self.behavior_sfs[policy_idx](torch.cat((batch_state, batch_action), -1))
        assert_shape(psi, [None, self.sf_dim])

        self.optimizers[policy_idx].zero_grad()
        loss = F.mse_loss(psi, psiTarget)
        loss.backward()
        self.optimizers[policy_idx].step()

        metrics = {
            "loss_" + str(policy_idx): loss,
            "action0": batch_action.mean(0)[0],
            "action1": batch_action.mean(0)[1],
            "action2": batch_action.mean(0)[2],
            "action3": batch_action.mean(0)[3],
        }
        wandb.log(metrics)

    def compute_target_psi_from_uniform_distr(
        self, s: torch.tensor, i: int
    ) -> torch.tensor:
        a = (
            torch.distributions.uniform.Uniform(-1, 1)
            .sample((self.value_n_particles, self.action_dim))
            .to(device)
        )
        s_a = get_sa_pairs(s, a)
        sf = self.target_sfs[i](s_a)
        return sf.view(s.shape[0], self.sf_dim, -1)

    def compute_q_target_from_policy_distr(
        self, s_a: torch.tensor, w: torch.tensor, i: int
    ) -> torch.tensor:
        sf = self.behavior_sfs[i](s_a)
        q = torch.tensordot(sf, w, dims=([1], [0]))
        return q.view(s_a.shape[0], -1)

    def compute_q_from_policy_distr(
        self, s: torch.tensor, w: torch.tensor, i: int
    ) -> torch.tensor:
        """in state s, task w, the performance of policy i, evaluated as q_mean

        Args:
            s (torch.tensor): state
            w (torch.tensor): task
            i (int): policy index

        Returns:
            torch.tensor: q_mean
        """
        a = self.policy_nets[i].acitons_for(s, self.action_n_particles)
        s_a = get_sa_pairs_(s, a)
        sf = self.behavior_sfs[i](s_a)
        q = torch.tensordot(sf, w, dims=([1], [0]))
        return torch.mean(q, dim=0)

    def gpi_policy(self, s: np.array, w: np.array) -> np.array:
        s = torch.tensor(s).float().unsqueeze(0).to(device)
        w = torch.tensor(w).float().to(device)

        qs = [
            self.compute_q_from_policy_distr(s, w, i)
            for i in range(len(self.behavior_sfs))
        ]
        policy_idx = torch.argmax(torch.tensor(qs))
        return self.policy_nets[policy_idx].get_action(s)

    def set_tasks(self, tasks: dict):
        self.env.tasks.update(tasks)

    def add_policy(self):
        bnet = SFMLP(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sf_dim=self.sf_dim,
            sizes=self.net_kwargs.get("value_sizes", [64, 64]),
        ).to(device)
        tnet = SFMLP(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sf_dim=self.sf_dim,
            sizes=self.net_kwargs.get("value_sizes", [64, 64]),
        ).to(device)
        tnet.load_state_dict(bnet.state_dict())

        policy = StochasticPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sizes=self.net_kwargs.get("policy_sizes", [64, 64]),
        ).to(device)

        optimizer = torch.optim.Adam(bnet.parameters(), lr=self.lr)
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=self.policy_lr)

        self.behavior_sfs.append(bnet)
        self.target_sfs.append(tnet)
        self.policy_nets.append(policy)
        self.optimizers.append(optimizer)
        self.policy_optimizers.append(policy_optimizer)
        self.n_policies += 1

    def sample_minibatch(self):
        batch = self.replay_buffer.sample(self.mini_batch_size, False)
        (
            batch_state,
            batch_next_state,
            batch_phi,
            batch_action,
            batch_reward,
            batch_done,
            batch_task,
        ) = zip(*batch)

        return (
            torch.FloatTensor(np.array(batch_state)).to(device),
            torch.FloatTensor(np.array(batch_next_state)).to(device),
            torch.FloatTensor(np.array(batch_phi)).to(device),
            torch.FloatTensor(np.array(batch_action)).to(device),
            torch.FloatTensor(np.array(batch_reward)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(batch_done)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(batch_task)).to(device),
        )

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError


if __name__ == "__main__":
    from drone_env.envs.multitask_env import MultiTaskEnv, MultiTaskPIDEnv
    from drone_env.envs.script import close_simulation

    auto_start_simulation = True
    if auto_start_simulation:
        close_simulation()

    ENV = MultiTaskPIDEnv
    env_name = ENV.__name__
    env_kwargs = {
        "DBG": False,
        "simulation": {
            "gui": True,
            "enable_meshes": True,
            "auto_start_simulation": auto_start_simulation,
            "position": (0, 0, 20.0),  # initial spawned position
        },
        "observation": {
            "noise_stdv": 0.0,
        },
        "action": {
            "act_noise_stdv": 0.0,
        },
        "tasks": {
            "tracking": {
                "ori_diff": np.array([0.0, 0.0, 0.0, 0.0]),
                "ang_diff": np.array([0.0, 0.0, 0.0]),
                "angvel_diff": np.array([0.0, 0.0, 0.0]),
                "pos_diff": np.array([0.0, 0.0, 1]),
                "vel_diff": np.array([0.0, 0.0, 0.0]),
                "vel_norm_diff": np.array([0.0]),
            },
            "success": np.array([0.0]),
            "action": np.array([0.0]),
        },
    }

    save_path = "~/results/" + "SFGPI/" + env_name + "/" + str(datetime.datetime.now())
    save_path = os.path.expanduser(save_path)

    default_dict = SFGPI.default_config()
    default_dict.update(dict(env_kwargs=env_kwargs, save_path=save_path))

    env = ENV(copy.deepcopy(env_kwargs))

    wandb.init(config=default_dict)
    pp.pprint(wandb.config)
    agent = SFGPI(env=env, **wandb.config)
    agent.train()
