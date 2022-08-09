import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import wandb
from common.kernel import adaptive_isotropic_gaussian_kernel
from common.policy import StochasticPolicy
from common.util import (
    assert_shape,
    get_sa_pairs,
    get_sa_pairs_,
    hard_update,
    soft_update,
    grad_false,
    update_params,
)
from common.value_function import TwinnedQNetwork
from common.agent import BasicAgent

torch.autograd.set_detect_anomaly(True)  # detect NaN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


class SQLAgent(BasicAgent):
    """soft q learning
    Tuomas Haarnoja, Reinforcement Learning with Deep Energy-Based Policies
    see https://github.com/haarnoja/softqlearning
    """

    @classmethod
    def default_config(cls):
        return dict(
            env="InvertedDoublePendulum-v4",
            env_kwargs={},
            total_timesteps=1e5,
            reward_scale=1,
            replay_buffer_size=1e6,
            mini_batch_size=128,
            min_n_experience=1024,
            lr=0.0005,
            alpha=1.0,
            gamma=0.99,
            tau=5e-3,
            updates_per_step=2,
            td_target_update_interval=1,
            value_n_particles=30,  # 64
            policy_lr=0.005,
            kernel_n_particles=20,  # 54
            kernel_update_ratio=0.5,
            grad_clip=None,
            render=False,
            log_interval=10,
            net_kwargs={"value_sizes": [64, 64], "policy_sizes": [32, 32]},
            seed=0,
        )

    def __init__(
        self,
        config: dict = {},
    ) -> None:
        self.config = config
        super().__init__(**config)

        self.learn_steps = 0
        self.episodes = 0
        self.lr = config.get("lr", 1e-3)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 5e-3)
        self.alpha = torch.tensor(config.get("alpha", 8)).to(device)
        self.td_target_update_interval = int(config.get("td_target_update_interval", 1))
        self.updates_per_step = config.get("updates_per_step", 1)
        self.value_n_particles = int(config.get("value_n_particles", 64))
        self.policy_lr = config.get("policy_lr", 1e-3)
        self.kernel_fn = adaptive_isotropic_gaussian_kernel
        self.kernel_n_particles = int(config.get("kernel_n_particles", 64))
        self.kernel_update_ratio = config.get("kernel_update_ratio", 0.5)
        self.grad_clip = config.get("grad_clip", None)
        self.net_kwargs = config["net_kwargs"]

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

        self.policy = StochasticPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sizes=self.net_kwargs.get("policy_sizes", [64, 64]),
        ).to(device)

        self.q1_optimizer = Adam(self.critic.Q1.parameters(), lr=self.lr)
        self.q2_optimizer = Adam(self.critic.Q2.parameters(), lr=self.lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.policy_lr)

    def run(self):
        return super().run()

    def train_episode(self):
        return super().train_episode()

    def act(self, state):
        return self.policy.get_action(state)

    def learn(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        batch = self.replay_buffer.sample(self.mini_batch_size)

        q1_loss, q2_loss, mean_q1, mean_q2 = self.calc_critic_loss(batch)
        policy_loss = self.calc_policy_loss(batch)

        update_params(self.q1_optimizer, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(self.q2_optimizer, self.critic.Q2, q2_loss, self.grad_clip)
        update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)

        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "loss/Q1": q1_loss.detach().item(),
                "loss/Q2": q2_loss.detach().item(),
                "loss/policy": policy_loss.detach().item(),
                "state/mean_Q1": mean_q1,
                "state/mean_Q2": mean_q2,
            }
            wandb.log(metrics)

    def calc_critic_loss(self, batch):

        (
            batch_state,
            batch_action,
            batch_reward,
            batch_next_state,
            batch_done,
        ) = batch

        batch_reward *= self.reward_scale

        curr_q1, curr_q2 = self.critic(batch_state, batch_action)
        curr_q1, curr_q2 = curr_q1.squeeze(), curr_q2.squeeze()
        assert_shape(curr_q1, [None])

        next_q = self.calc_next_q(batch_next_state)
        target_q, next_v = self.calc_target_q(next_q, batch_reward, batch_done)
        assert_shape(next_q, [None, self.value_n_particles])
        assert_shape(target_q, [None])

        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)
        return q1_loss, q2_loss, mean_q1, mean_q2

    def calc_policy_loss(self, batch):
        """Create a minimization operation for policy update (SVGD)."""
        (batch_state, _, _, _, _) = batch

        (
            fixed_actions,
            updated_actions,
            n_fixed_actions,
            n_updated_actions,
        ) = self.get_fix_update_actions(batch_state)

        grad_log_p = self.calc_grad_log_p(batch_state, fixed_actions)
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
            self.policy.parameters(),
            grad_outputs=action_gradients,
        )

        # multiply weight since optimizer will differentiate it later so we can apply the gradient
        surrogate_loss = torch.sum(
            torch.stack(
                [
                    torch.sum(w * g.detach())
                    for w, g in zip(self.policy.parameters(), gradients)
                ],
                dim=0,
            )
        )
        assert surrogate_loss.requires_grad == True

        loss = -surrogate_loss
        return loss

    def calc_next_q(self, batch_next_state):
        # eqn10: sampling a for importance sampling
        sample_action = (
            torch.distributions.uniform.Uniform(-1, 1)
            .sample((self.value_n_particles, self.action_dim))
            .to(device)
        )

        # eqn10: Q(s,a)
        s, a = get_sa_pairs(batch_next_state, sample_action)
        next_q1, next_q2 = self.critic_target(s, a)

        n_sample = batch_next_state.shape[0]
        next_q1, next_q2 = next_q1.view(n_sample, -1), next_q2.view(n_sample, -1)
        next_q = torch.min(next_q1, next_q2)
        next_q /= self.alpha
        return next_q

    def calc_target_q(self, next_q, batch_reward, batch_done):
        # eqn10
        next_v = torch.logsumexp(next_q, 1)
        assert_shape(next_v, [None])

        # Importance weights add just a constant to the value.
        next_v -= np.log(self.value_n_particles)
        next_v += self.action_dim * np.log(2)
        next_v *= self.alpha
        next_v = next_v.unsqueeze(1)

        # eqn11: \hat Q(s,a)
        target_q = (
            (batch_reward + (1 - batch_done) * self.gamma * next_v).squeeze(1).detach()
        )
        return target_q, next_v

    def get_fix_update_actions(self, batch_state):
        actions = self.policy.acitons_for(
            observations=batch_state, n_action_samples=self.kernel_n_particles
        )
        assert_shape(actions, [None, self.kernel_n_particles, self.action_dim])

        n_updated_actions = int(self.kernel_n_particles * self.kernel_update_ratio)
        n_fixed_actions = self.kernel_n_particles - n_updated_actions
        fixed_actions, updated_actions = torch.split(
            actions, [n_fixed_actions, n_updated_actions], dim=1
        )
        assert_shape(fixed_actions, [None, n_fixed_actions, self.action_dim])
        assert_shape(updated_actions, [None, n_updated_actions, self.action_dim])
        return fixed_actions, updated_actions, n_fixed_actions, n_updated_actions

    def calc_grad_log_p(self, batch_state, fixed_actions):
        s, a = get_sa_pairs_(batch_state, fixed_actions)
        svgd_q0, svgd_q1 = self.critic(s, a)

        n_sample = batch_state.shape[0]
        svgd_q0, svgd_q1 = svgd_q0.view(n_sample, -1), svgd_q1.view(n_sample, -1)
        svgd_q = torch.min(svgd_q0, svgd_q1)

        # Target log-density. Q_soft in Equation 13:
        squash_correction = torch.sum(torch.log(1 - fixed_actions**2 + EPS), dim=-1)
        log_p = svgd_q + squash_correction

        grad_log_p = torch.autograd.grad(
            log_p, fixed_actions, grad_outputs=torch.ones_like(log_p)
        )
        grad_log_p = grad_log_p[0].unsqueeze(2).detach()
        return grad_log_p


if __name__ == "__main__":
    # ============== profile ==============#
    # pip install snakeviz
    # python -m cProfile -o out.profile rl/rl/softqlearning/sql.py -s time
    # snakeviz sql.profile

    # ============== sweep ==============#
    # wandb sweep rl/rl/sweep_sql.yaml

    env = "InvertedDoublePendulum-v4"  # Pendulum-v1, LunarLander-v2, InvertedDoublePendulum-v4, MountainCarContinuous-v0, Ant-v4
    env_kwargs = {"continuous": True} if env == "LunarLander-v2" else {}
    render = True

    default_dict = SQLAgent.default_config()
    default_dict.update(
        dict(
            env=env,
            env_kwargs=env_kwargs,
            render=render,
        )
    )

    wandb.init(config=default_dict)
    print(wandb.config)

    agent = SQLAgent(wandb.config)
    agent.run()
