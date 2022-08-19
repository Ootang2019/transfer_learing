import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import wandb
from agent.common.policy import GaussianPolicy
from agent.common.util import (
    assert_shape,
    hard_update,
    soft_update,
    grad_false,
    update_params,
)
from agent.common.value_function import TwinnedQNetwork
from agent.common.agent import BasicAgent, MultiTaskAgent

torch.autograd.set_detect_anomaly(True)  # detect NaN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


class SACAgent(MultiTaskAgent):
    """SAC
    Tuomas Haarnoja, Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
    see https://github.com/haarnoja/sac/blob/master/sac/algos/sac.py
    and https://github.com/ku2482/soft-actor-critic.pytorch
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
            mini_batch_size=128,
            replay_buffer_size=1e6,
            multi_step=2,
            updates_per_step=1,
            min_n_experience=int(1024),
            td_target_update_interval=1,
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
        self.policy_lr = config.get("policy_lr", 1e-3)
        self.net_kwargs = config["net_kwargs"]
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 5e-3)
        self.td_target_update_interval = int(config.get("td_target_update_interval", 1))
        self.updates_per_step = config.get("updates_per_step", 1)
        self.grad_clip = config.get("grad_clip", None)
        self.min_n_experience = self.start_steps = int(
            config.get("min_n_experience", int(1e4))
        )
        self.entropy_tuning = config.get("entropy_tuning", True)
        self.eval_interval = config.get("eval_interval", 1000)

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

        if self.entropy_tuning:
            self.alpha_lr = config.get("alpha_lr", 3e-4)
            # target entropy = -|A|
            self.target_entropy = -torch.prod(
                torch.Tensor(self.env.action_space.shape).to(device)
            ).item()
            # optimize log(alpha), instead of alpha
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor(config.get("alpha", 0.2)).to(device)

    def run(self):
        return super().run()

    def train_episode(self):
        super().train_episode()
        if self.steps % self.eval_interval == 0:
            self.evaluate()

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

        batch = self.replay_buffer.sample(self.mini_batch_size)

        q1_loss, q2_loss, mean_q1, mean_q2 = self.calc_critic_loss(batch)
        policy_loss, entropies = self.calc_policy_loss(batch)

        update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)
        update_params(self.q1_optimizer, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(self.q2_optimizer, self.critic.Q2, q2_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies)
            update_params(self.alpha_optimizer, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
            wandb.log(
                {
                    "loss/alpha": entropy_loss.detach().item(),
                    "state/alpha": self.alpha.detach().item(),
                }
            )

        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "loss/Q1": q1_loss.detach().item(),
                "loss/Q2": q2_loss.detach().item(),
                "loss/policy": policy_loss.detach().item(),
                "state/mean_Q1": mean_q1,
                "state/mean_Q2": mean_q2,
                "state/entropy": entropies.detach().mean().item(),
            }
            wandb.log(metrics)

    def calc_critic_loss(self, batch):
        (observations, features, actions, rewards, next_observations, dones) = batch
        curr_q1, curr_q2 = self.calc_current_q(observations, actions)
        target_q = self.calc_target_q(rewards, next_observations, dones)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors.
        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)

        return q1_loss, q2_loss, mean_q1, mean_q2

    def calc_policy_loss(self, batch):
        (observations, features, actions, rewards, next_observations, dones) = batch

        # We re-sample actions to calculate expectations of Q.
        sampled_action, entropy, _ = self.policy.sample(observations)
        # expectations of Q with clipped double Q technique
        q1, q2 = self.critic(observations, sampled_action)
        q = torch.min(q1, q2)

        # Policy objective is maximization of (Q + alpha * entropy).
        policy_loss = torch.mean((-q - self.alpha * entropy))
        return policy_loss, entropy

    def calc_entropy_loss(self, entropy):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
        )
        return entropy_loss

    def calc_current_q(self, states, actions):
        curr_q1, curr_q2 = self.critic(states, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, rewards, next_states, dones):

        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies

        target_q = self.reward_scale * rewards + (1.0 - dones) * self.gamma * next_q

        return target_q

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            returns[i] = episode_reward

        mean_return = np.mean(returns)

        wandb.log({"reward/test": mean_return})


if __name__ == "__main__":
    # ============== profile ==============#
    # pip install snakeviz
    # python -m cProfile -o out.profile rl/rl/softqlearning/sac.py -s time
    # snakeviz sac.profile

    # ============== sweep ==============#
    # wandb sweep rl/rl/sweep_sac.yaml
    import benchmark_env
    import drone_env
    import gym
    from drone_env.envs.script import close_simulation

    # env = "InvertedDoublePendulum-v4"  # Pendulum-v1, LunarLander-v2, InvertedDoublePendulum-v4, MountainCarContinuous-v0, Ant-v4
    env = "multitaskpid-v0"
    # env_kwargs = {"continuous": True} if env == "LunarLander-v2" else {}
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

    default_dict = SACAgent.default_config()
    default_dict.update(
        dict(
            env=env,
            env_kwargs=env_kwargs,
            render=render,
        )
    )

    wandb.init(config=default_dict)
    print(wandb.config)

    agent = SACAgent(wandb.config)
    agent.run()
