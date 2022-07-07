import datetime
from email import policy
from pathlib import Path
import os.path
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from common.policy import StochasticPolicy
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
            total_timesteps=1e5,
            n_timesteps=200,
            reward_scale=1,
            replay_buffer_size=1e6,
            mini_batch_size=128,
            min_n_experience=1024,
            lr=0.0004924,
            alpha=1.2,
            gamma=0.99,
            td_target_update_interval=1,
            value_n_particles=88,
            policy_lr=0.004866,
            action_n_particles=54,
            render=False,
            net_kwargs={"value_sizes": [64, 64], "policy_sizes": [32, 32]},
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
        replay_buffer=ReplayBuffer,
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
        self.action_n_particles = int(action_n_particles)

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
        self.tasks = self.env.get_tasks_weights()

    def train(self):
        episode_reward = 0
        n_epochs = int(self.total_timesteps / self.n_timesteps)

        for epoch in range(self.epoch, n_epochs):
            self.epoch = epoch
            state = self.env.reset()
            phi = self.env.get_features()

            episode_reward = 0

            for _ in range(self.n_timesteps):
                if self.render:
                    self.env.render()
                action = self.gpi_policy(state, self.tasks)

                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                self.replay_buffer.add((state, next_state, phi, action, reward, done))

                if self.replay_buffer.size() > self.min_n_experience:
                    batch = self.sample_minibatch()
                    # self.td_update(batch)
                    # self.policy_update(batch)

                if done:
                    break

                state = next_state
                phi = info["features"]

            wandb.log({"episode_reward": episode_reward})

    def get_tasks_weights(self):
        tasks = self.tasks.copy()
        feature_weights = np.concatenate([v for k, v in tasks["tracking"].items()])
        return np.concatenate([feature_weights, tasks["success"], tasks["action"]])

    def td_update(self):
        raise NotImplementedError

    def policy_update(self):
        raise NotImplementedError

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

    def update_target_sfs(self):
        [
            self.target_sfs[i].load_state_dict(self.behavior_sfs[i].state_dict())
            for i in range(len(self.target_sfs))
        ]

    def td_update(self, batch):
        if self.begin_learn_td is False:
            print("begin learning q function")
            self.begin_learn_td = True

        if self.learn_steps % self.td_target_update_interval == 0:
            self.update_target_sfs()

        self.learn_steps += 1

        [self.update_sfs(batch, i) for i in range(len(self.behavior_sfs))]

    def update_sfs(self, batch, policy_idx):

        (
            batch_state,
            batch_next_state,
            batch_phi,
            batch_next_phi,
            batch_action,
            batch_reward,
            batch_done,
            batch_task,
        ) = batch
        batch_reward *= self.reward_scale

        target_psi = self.compute_target_psi_from_uniform_distr(
            batch_next_state,
            batch_task,
            policy_idx,
        )
        target_psi /= self.alpha

        assert_shape(target_psi, [None, self.value_n_particles])

        next_Psi = torch.logsumexp(target_psi, 1)
        assert_shape(next_Psi, [None])

        next_Psi -= np.log(self.value_n_particles)
        next_Psi += self.action_dim * np.log(2)
        next_Psi *= self.alpha
        next_Psi = next_Psi.unsqueeze(1)

        target = (
            (batch_phi + (1 - batch_done) * self.gamma * next_Psi).squeeze(1).detach()
        )
        assert_shape(target, [None])

        q_value = self.behavior_sfs[policy_idx](
            torch.cat((batch_state, batch_action), -1)
        ).squeeze()
        assert_shape(q_value, [None])

        self.optimizers[policy_idx].zero_grad()
        loss = F.mse_loss(q_value, target)
        loss.backward()
        self.optimizers[policy_idx].step()

        metrics = {
            "loss_" + str(policy_idx): loss,
            "q_mean_" + str(policy_idx): q_value.mean(),
            "v_mean_" + str(policy_idx): next_value.mean(),
            "target_" + str(policy_idx): target.mean(),
        }
        wandb.log(metrics)

    def gpi_policy(self, s: np.array, w: np.array) -> np.array:
        s = torch.tensor(s).float().unsqueeze(0).to(device)
        w = torch.tensor(w).float().to(device)

        qs = [
            self.compute_q_from_policy_distr(s, w, i)
            for i in range(len(self.behavior_sfs))
        ]
        policy_idx = torch.argmax(torch.tensor(qs))
        return self.policy_nets[policy_idx].get_action(s)

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

    def compute_target_psi_from_uniform_distr(
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
        a = (
            torch.distributions.uniform.Uniform(-1, 1)
            .sample((self.value_n_particles, self.action_dim))
            .to(device)
        )
        s_a = get_sa_pairs(s, a)
        sf = self.target_sfs[i](s_a)
        return sf.view(s.shape[0], -1)

    def compute_target_q_from_uniform_distr(
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
        a = (
            torch.distributions.uniform.Uniform(-1, 1)
            .sample((self.value_n_particles, self.action_dim))
            .to(device)
        )
        s_a = get_sa_pairs(s, a)
        sf = self.target_sfs[i](s_a)
        q = torch.tensordot(sf, w, dims=([1], [0]))
        return q.view(s.shape[0], -1)

    def sample_minibatch(self):
        batch = self.replay_buffer.sample(self.mini_batch_size, False)
        (
            batch_state,
            batch_next_state,
            batch_action,
            batch_reward,
            batch_done,
            batch_task,
        ) = zip(*batch)

        return (
            torch.FloatTensor(np.array(batch_state)).to(device),
            torch.FloatTensor(np.array(batch_next_state)).to(device),
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
    from drone_env.envs.multitask_env import MultiTaskEnv
    from drone_env.envs.script import close_simulation

    auto_start_simulation = False
    if auto_start_simulation:
        close_simulation()

    ENV = MultiTaskEnv
    env_name = ENV.__name__
    env_kwargs = {
        "DBG": True,
        "simulation": {
            "gui": True,
            "enable_meshes": True,
            "auto_start_simulation": auto_start_simulation,
            "position": (0, 0, 30),  # initial spawned position
        },
        "tasks": {
            "tracking": {
                "ori_diff": np.array([0.0, 0.0, 0.0, 0.0]),
                "ang_diff": np.array([0.0, 0.0, 0.0]),
                "angvel_diff": np.array([0.0, 0.0, 0.0]),
                "pos_diff": np.array([0.5, 0.0, 0.0]),
                "vel_diff": np.array([0.5, 0.0, 0.0]),
                "vel_norm_diff": np.array([0.0]),
            },
            "success": np.array([0.0]),
            "action": np.array([0.0]),
        },
    }

    save_path = "~/results/" + "SFGPI/" + env_name + "/" + str(datetime.datetime.now())
    save_path = os.path.expanduser(save_path)

    default_dict = SFGPI.default_config()
    default_dict.update(
        dict(
            env_kwargs=env_kwargs,
            save_path=save_path,
        )
    )

    wandb.init(config=default_dict)
    pp.pprint("experiment configuration: ", wandb.config)

    env = ENV(copy.deepcopy(env_kwargs))
    agent = SFGPI(env=env, **wandb.config)

    # s = agent.env.reset()
    # w = np.arange(s.shape[0])

    # agent.add_policy()
    # a = agent.gpi_policy(s, w)

    agent.train()
