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
    get_sa_pairs,
)
from agent.common.value_function import TwinnedQNetwork
from agent.common.agent import BasicAgent, MultiTaskAgent

torch.autograd.set_detect_anomaly(True)  # detect NaN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


class NACAgent(MultiTaskAgent):
    """NAC
    Yang Gao, Reinforcement Learning from Imperfect Demonstrations
    """

    @classmethod
    def default_config(cls):
        return dict(
            env="myInvertedDoublePendulum-v4",
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
            replay_buffer_size=1e6,
            multi_step=2,
            prioritized_memory=True,
            updates_per_step=1,
            mini_batch_size=128,
            min_n_experience=int(1024),
            td_target_update_interval=1,
            n_particles=64,
            grad_clip=None,
            render=False,
            log_interval=10,
            entropy_tuning=True,
            alpha_bnd=[0.1, 10],
            normalize_critic=True,
            calc_pol_loss_by_advantage=True,
            eval=True,
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
        self.n_particles = config.get("n_particles", 64)
        self.normalize_critic = config.get("normalize_critic", True)
        self.calc_pol_loss_by_advantage = config.get("calc_pol_loss_by_advantage", True)
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
            self.alpha_bnd = config.get("alpha_bnd", [1e-1, 10])
            self.alpha = self.log_alpha.exp()
            self.alpha = torch.clip(self.alpha, *self.alpha_bnd)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor(config.get("alpha", 0.2)).to(device)

        self.beta = 1 / self.mini_batch_size

        wandb.watch(self.critic)
        wandb.watch(self.policy)

    def run(self):
        return super().run()

    def train_episode(self):
        super().train_episode()
        if self.steps % self.eval_interval == 0:
            self.evaluate()

    def explore(self, state, w):
        # act with randomness
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action

    def exploit(self, state, w):
        # act without randomness
        with torch.no_grad():
            _, _, action = self.policy.sample(state)
        return action

    def learn(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if self.prioritized_memory:
            batch, indices, weights = self.replay_buffer.sample(self.mini_batch_size)
        else:
            batch = self.replay_buffer.sample(self.mini_batch_size)
            weights = 1

        policy_loss, entropies = self.calc_policy_loss(batch, weights)

        (
            q1_loss,
            q2_loss,
            errors,
            mean_q1,
            mean_q2,
            mean_v,
            loss_q,
            loss_pg,
        ) = self.calc_critic_loss(batch, weights)

        update_params(self.q1_optimizer, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(self.q2_optimizer, self.critic.Q2, q2_loss, self.grad_clip)
        update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optimizer, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
            self.alpha = torch.clip(self.log_alpha.exp(), *self.alpha_bnd)

        if self.prioritized_memory:
            self.replay_buffer.update_priority(indices, errors.cpu().numpy())

        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "loss/Q1": q1_loss.detach().item(),
                "loss/Q2": q2_loss.detach().item(),
                "loss/policy": policy_loss.detach().item(),
                "loss/Q": loss_q,
                "loss/PG": loss_pg,
                "state/mean_Q1": mean_q1,
                "state/mean_Q2": mean_q2,
                "state/mean_V": mean_v,
                "state/entropy": entropies.detach().mean().item(),
            }
            wandb.log(metrics)

            if self.entropy_tuning:
                wandb.log(
                    {
                        "loss/alpha": entropy_loss.detach().item(),
                        "state/alpha": self.alpha.detach().item(),
                    }
                )

    def calc_critic_loss(self, batch, weights):
        (observations, features, actions, rewards, next_observations, dones) = batch
        rewards *= self.reward_scale

        # critic loss is mean squared TD errors.
        curr_q1, curr_q2 = self.calc_current_q(observations, actions)
        target_q = self.calc_target_q(rewards, next_observations, dones)

        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        # critic normalization loss
        curr_q1, curr_q2 = self.calc_current_q(observations, actions)
        curr_v = self.calc_v(observations)

        if self.normalize_critic:
            pg_loss1 = self.calc_critic_normalize_loss(
                self.critic.Q1, curr_q1, target_q, curr_v, weights
            )
            pg_loss2 = self.calc_critic_normalize_loss(
                self.critic.Q2, curr_q2, target_q, curr_v, weights
            )
        else:
            pg_loss1 = 0
            pg_loss2 = 0

        # sum all loss
        loss1 = q1_loss + pg_loss1
        loss2 = q2_loss + pg_loss2

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # log to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()
        mean_v = curr_v.detach().mean().item()

        loss_q1 = q1_loss.detach()
        if self.normalize_critic:
            loss_pg1 = pg_loss1.detach()
        else:
            loss_pg1 = pg_loss1

        return loss1, loss2, errors, mean_q1, mean_q2, mean_v, loss_q1, loss_pg1

    def calc_policy_loss(self, batch, weights):
        (observations, features, actions, rewards, next_observations, dones) = batch
        rewards *= self.reward_scale

        # We re-sample actions to calculate expectations of Q.
        sampled_action, entropy, _ = self.policy.sample(observations)
        # expectations of Q with clipped double Q technique
        q1, q2 = self.critic_target(observations, sampled_action)
        q = torch.min(q1, q2)

        if self.calc_pol_loss_by_advantage:
            v = self.calc_v(observations)
            q = q - v

        # Policy objective is maximization of (Q + alpha * entropy).
        policy_loss = torch.mean((-q - self.alpha * entropy) * weights)
        return policy_loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach() * weights
        )
        return entropy_loss

    def calc_current_q(self, states, actions):
        curr_q1, curr_q2 = self.critic(states, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, rewards, next_states, dones):
        with torch.no_grad():
            next_v = self.calc_v(next_states)
        target_q = rewards + (1.0 - dones) * self.gamma * next_v
        return target_q

    def calc_critic_normalize_loss(self, net, curr_q, target_q, curr_v, weights):
        gradient = self.calc_normalized_grad(net, curr_q, target_q, curr_v, weights)
        pg_loss = self.calc_surrogate_loss(net, gradient)
        pg_loss = self.beta * pg_loss
        return pg_loss

    def calc_v(self, obs):
        with torch.no_grad():
            sample_act = (
                torch.distributions.uniform.Uniform(-1, 1)
                .sample((self.n_particles, self.action_dim))
                .to(device)
            )
            sample_obs, sample_act = get_sa_pairs(obs, sample_act)
            q0, q1 = self.critic_target(sample_obs, sample_act)
            q = torch.min(q0, q1)
            q = q.view(-1, self.n_particles)
            v = self.log_sum_exp_q(q)
        return v

    def log_sum_exp_q(self, qs):
        v = torch.logsumexp(qs / self.alpha, 1)
        v = v - np.log(self.n_particles)
        v = v + self.action_dim * np.log(2)
        v = self.alpha * v
        v = v.unsqueeze(1)
        assert_shape(v, [None, 1])
        return v

    def calc_normalized_grad(self, net, q, target_q, v, weights):
        gradient = torch.autograd.grad(
            (q - v),
            net.parameters(),
            grad_outputs=(q - target_q) * weights,
        )
        return gradient

    def calc_surrogate_loss(self, net, gradient):
        # multiply weights and differentiate later to apply gradient
        sur_loss = torch.sum(
            torch.stack(
                [torch.sum(w * g.detach()) for w, g in zip(net.parameters(), gradient)],
                dim=0,
            )
        )
        assert sur_loss.requires_grad == True
        return sur_loss

    def calc_priority_error(self, batch):
        (obs, _, act, rewards, next_obs, dones) = batch
        rewards *= self.reward_scale

        with torch.no_grad():
            curr_q1, curr_q2 = self.calc_current_q(obs, act)
        target_q = self.calc_target_q(rewards, next_obs, dones)

        error = torch.abs(curr_q1 - target_q).item()
        return error


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

    env = "myInvertedDoublePendulum-v4"  # myInvertedDoublePendulum-v4, multitask-v0
    auto_start_simulation = False
    if auto_start_simulation:
        close_simulation()

    env_kwargs = {}
    if env == "multitask-v0":
        from agent.task_config import get_task_schedule

        env_config = {
            "simulation": {
                "gui": True,
                "enable_meshes": True,
                "auto_start_simulation": auto_start_simulation,
                "position": (0, 0, 25),  # initial spawned position
            },
            "observation": {
                "noise_stdv": 0.0,
            },
            "action": {
                "act_noise_stdv": 0.0,
                "thrust_range": [-0.2, 0.2],
            },
            "target": {"type": "FixedGoal"},
        }
        env_kwargs = {"config": env_config}
        task_schedule = get_task_schedule(
            ["roll", "pitch", "yaw", "att", "z", "xz", "yz", "xyz"]
        )

    default_dict = NACAgent.default_config()
    default_dict.update(
        dict(
            env=env,
            env_kwargs=env_kwargs,
            render=False,
            entropy_tuning=True,
            calc_pol_loss_by_advantage=True,
            prioritized_memory=True,
            normalize_critic=True,
            reward_scale=1,
            lr=5e-4,
            policy_lr=5e-3,
            total_timesteps=5e4,
            eval=True,
            evaluate_episodes=3,
        )
    )

    wandb.init(config=default_dict)
    print(wandb.config)

    agent = NACAgent(wandb.config)
    agent.run()
