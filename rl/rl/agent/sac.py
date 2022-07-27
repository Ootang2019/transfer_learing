import datetime
from pathlib import Path
import os.path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import wandb
from common.policy import GaussianPolicy
from common.replay_buffer import ReplayBuffer
from common.util import (
    assert_shape,
    hard_update,
    soft_update,
    grad_false,
    to_batch,
    update_params,
)
from common.value_function import TwinnedQNetwork
from common.agent import AbstractAgent

torch.autograd.set_detect_anomaly(True)  # detect NaN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


class SACAgent(AbstractAgent):
    """SAC"""

    @classmethod
    def default_config(cls):
        return dict(
            env="InvertedDoublePendulum-v4",
            env_kwargs={},
            total_timesteps=int(3e6),
            mini_batch_size=256,
            lr=3e-4,
            policy_lr=3e-4,
            net_kwargs={"value_sizes": [64, 64], "policy_sizes": [32, 32]},
            replay_buffer_size=1e6,
            gamma=0.99,
            tau=5e-3,
            alpha=0.2,
            grad_clip=None,
            updates_per_step=1,
            reward_scale=1,
            min_n_experience=int(1e4),
            td_target_update_interval=1,
            render=False,
            log_interval=10,
            seed=0,
        )

    def __init__(
        self,
        config: dict = {},
    ) -> None:
        self.config = config
        super().__init__(**config)

        self.lr = config.get("lr", 1e-3)
        self.policy_lr = config.get("policy_lr", 1e-3)
        self.net_kwargs = config["net_kwargs"]
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 5e-3)
        self.alpha = torch.tensor(config.get("alpha", 0.2)).to(device)
        self.td_target_update_interval = int(config.get("td_target_update_interval", 1))
        self.updates_per_step = config.get("updates_per_step", 0.99)
        self.grad_clip = config.get("grad_clip", None)
        self.begin_learn_td = False
        self.min_n_experience = self.start_steps = int(
            config.get("min_n_experience", int(1e4))
        )

        self.critic = TwinnedQNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sizes=self.net_kwargs.get("value_sizes", [64, 64]),
        ).to(device)
        self.critic_target = (
            TwinnedQNetwork(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                sizes=self.net_kwargs.get("value_sizes", [64, 64]),
            )
            .to(device)
            .eval()
        )
        hard_update(self.critic_target, self.critic)
        grad_false(self.critic_target)

        self.policy = GaussianPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sizes=self.net_kwargs.get("policy_sizes", [64, 64]),
        ).to(device)

        self.q1_optimizer = Adam(self.critic.Q1.parameters(), lr=self.lr)
        self.q2_optimizer = Adam(self.critic.Q2.parameters(), lr=self.lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.policy_lr)

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.total_timesteps:
                break

    def train_episode(self):
        episode_reward = 0
        self.episodes += 1
        episode_steps = 0
        done = False
        state = self.env.reset()

        while not done:
            if self.render:
                self.env.render()

            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            self.replay_buffer.add((state, action, reward, next_state, done))

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            state = next_state

        if self.episodes % self.log_interval == 0:
            wandb.log({"episode_reward": episode_reward})

    def is_update(self):
        return (
            self.replay_buffer.size() > self.mini_batch_size
            and self.steps >= self.start_steps
        )

    def act(self, state):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        # act without randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, action = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def learn(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        batch = self.sample_minibatch()
        weights = 1

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(
            batch, weights
        )
        policy_loss, entropies = self.calc_policy_loss(batch, weights)

        update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)
        update_params(self.q1_optimizer, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(self.q2_optimizer, self.critic.Q2, q2_loss, self.grad_clip)

        if self.learn_steps % self.log_interval == 0:
            log_dict = {
                "loss/Q1": q1_loss.detach().item(),
                "loss/Q2": q2_loss.detach().item(),
                "loss/policy": policy_loss.detach().item(),
                "state/mean_Q1": mean_q1,
                "state/mean_Q2": mean_q2,
                "state/entropy": entropies.detach().mean().item(),
            }
            wandb.log(log_dict)

    def sample_minibatch(self):
        batch = self.replay_buffer.sample(self.mini_batch_size, False)
        return to_batch(*zip(*batch), device)

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # We re-sample actions to calculate expectations of Q.
        sampled_action, entropy, _ = self.policy.sample(states)
        # expectations of Q with clipped double Q technique
        q1, q2 = self.critic(states, sampled_action)
        q = torch.min(q1, q2)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = torch.mean((-q - self.alpha * entropy) * weights)
        return policy_loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach() * weights
        )
        return entropy_loss

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self.gamma * next_q

        return target_q


if __name__ == "__main__":
    # ============== profile ==============#
    # pip install snakeviz
    # python -m cProfile -o out.profile rl/rl/softqlearning/sql.py -s time
    # snakeviz sql.profile

    # ============== sweep ==============#
    # wandb sweep rl/rl/sweep.yaml

    env = "InvertedDoublePendulum-v4"  # Pendulum-v1, LunarLander-v2, InvertedDoublePendulum-v4, MountainCarContinuous-v0, Ant-v4
    env_kwargs = {"continuous": True} if env == "LunarLander-v2" else {}
    render = True

    save_path = "~/results/" + env + "/" + str(datetime.datetime.now())
    save_path = os.path.expanduser(save_path)

    default_dict = SACAgent.default_config()
    default_dict.update(
        dict(
            env=env,
            env_kwargs=env_kwargs,
            save_path=save_path,
            render=render,
        )
    )

    wandb.init(config=default_dict)
    print(wandb.config)

    agent = SACAgent(wandb.config)
    agent.run()
