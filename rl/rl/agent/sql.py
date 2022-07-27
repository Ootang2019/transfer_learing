import datetime
from pathlib import Path
import os.path
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
    to_batch,
    update_params,
)
from common.value_function import QNetwork
from common.agent import AbstractAgent

torch.autograd.set_detect_anomaly(True)  # detect NaN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


class SQLAgent(AbstractAgent):
    """soft q learning with continuous action space"""

    @classmethod
    def default_config(cls):
        return dict(
            env="InvertedDoublePendulum-v4",
            env_kwargs={},
            total_timesteps=1e5,
            n_timesteps=200,
            reward_scale=1,
            replay_buffer_size=1e6,
            mini_batch_size=128,
            min_n_experience=1024,
            lr=0.0005,
            alpha=1.0,
            gamma=0.99,
            td_target_update_interval=1,
            value_n_particles=64,
            policy_lr=0.005,
            kernel_n_particles=54,
            kernel_update_ratio=0.5,
            render=False,
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
        self.alpha = torch.tensor(config.get("alpha", 8)).to(device)
        self.begin_learn_td = False
        self.td_target_update_interval = int(config.get("td_target_update_interval", 1))
        self.value_n_particles = int(config.get("value_n_particles", 16))
        self.policy_lr = config.get("policy_lr", 1e-3)
        self.kernel_fn = adaptive_isotropic_gaussian_kernel
        self.kernel_n_particles = int(config.get("kernel_n_particles", 16))
        self.kernel_update_ratio = config.get("kernel_update_ratio", 0.5)
        self.net_kwargs = config["net_kwargs"]

        self.behavior_net = QNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sizes=self.net_kwargs.get("value_sizes", [64, 64]),
        ).to(device)
        self.target_net = QNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sizes=self.net_kwargs.get("value_sizes", [64, 64]),
        ).to(device)
        hard_update(self.target_net, self.behavior_net)

        self.policy = StochasticPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sizes=self.net_kwargs.get("policy_sizes", [64, 64]),
        ).to(device)

        self.optimizer = Adam(self.behavior_net.parameters(), lr=self.lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.policy_lr)

    def is_update(self):
        return (
            self.replay_buffer.size() > self.mini_batch_size
            and self.steps >= self.start_steps
        )

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

            action = self.policy.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            self.replay_buffer.add((state, action, reward, next_state, done))

            if self.is_update():
                batch = self.sample_minibatch()
                self.td_update(batch)
                self.svgd_update(batch)

            state = next_state

        wandb.log({"episode_reward": episode_reward})

    def sample_minibatch(self):
        batch = self.replay_buffer.sample(self.mini_batch_size, False)
        return to_batch(*zip(*batch), device)

    def td_update(self, batch):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            hard_update(self.target_net, self.behavior_net)

        (
            batch_state,
            batch_action,
            batch_reward,
            batch_next_state,
            batch_done,
        ) = batch
        batch_reward *= self.reward_scale

        # eqn10: sampling a for importance sampling
        sample_action = (
            torch.distributions.uniform.Uniform(-1, 1)
            .sample((self.value_n_particles, self.action_dim))
            .to(device)
        )

        # eqn10: Q(s,a)
        s_a = get_sa_pairs(batch_next_state, sample_action)
        q_value_target = self.target_net(s_a).view(batch_next_state.shape[0], -1)
        q_value_target /= self.alpha

        assert_shape(q_value_target, [None, self.value_n_particles])

        q_value = self.behavior_net(
            torch.cat((batch_state, batch_action), -1)
        ).squeeze()
        assert_shape(q_value, [None])

        # eqn10
        next_value = torch.logsumexp(q_value_target, 1)
        assert_shape(next_value, [None])

        # Importance weights add just a constant to the value.
        next_value -= np.log(self.value_n_particles)
        next_value += self.action_dim * np.log(2)
        next_value *= self.alpha
        next_value = next_value.unsqueeze(1)

        # eqn11: \hat Q(s,a)
        target = (
            (batch_reward + (1 - batch_done) * self.gamma * next_value)
            .squeeze(1)
            .detach()
        )
        assert_shape(target, [None])

        # eqn 11
        loss = F.mse_loss(q_value, target)
        update_params(self.optimizer, self.behavior_net, loss)

        metrics = {
            "loss/q": loss,
            "state/mean_q": q_value.mean(),
            "state/std_q": q_value.std(),
            "state/mean_v": next_value.mean(),
            "state/target": target.mean(),
        }
        wandb.log(metrics)

    def svgd_update(self, batch):
        """Create a minimization operation for policy update (SVGD)."""
        (batch_state, _, _, _, _) = batch

        actions = self.policy.acitons_for(
            observations=batch_state, n_action_samples=self.kernel_n_particles
        )
        assert_shape(actions, [None, self.kernel_n_particles, self.action_dim])

        # a_i: fixed actions
        # a_j: updated actions
        n_updated_actions = int(self.kernel_n_particles * self.kernel_update_ratio)
        n_fixed_actions = self.kernel_n_particles - n_updated_actions
        fixed_actions, updated_actions = torch.split(
            actions, [n_fixed_actions, n_updated_actions], dim=1
        )
        assert_shape(fixed_actions, [None, n_fixed_actions, self.action_dim])
        assert_shape(updated_actions, [None, n_updated_actions, self.action_dim])

        s_a = get_sa_pairs_(batch_state, fixed_actions)
        svgd_target_values = self.behavior_net(s_a).view(batch_state.shape[0], -1)

        # Target log-density. Q_soft in Equation 13:
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
        update_params(self.policy_optimizer, self.policy, loss)

        metrics = {
            "loss/policy": loss,
        }
        wandb.log(metrics)

    def save_model(self):
        path = self.save_path + "/" + str(self.episodes)
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "episodes": self.episodes,
                "model_state_dict": self.behavior_net.state_dict(),
            },
            path + "/value_net.pth",
        )
        torch.save(
            {
                "episodes": self.episodes,
                "model_state_dict": self.policy.state_dict(),
            },
            path + "/policy_net.pth",
        )

    def load_model(self, path):
        self.behavior_net = QNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sizes=self.net_kwargs.get("value_sizes", [64, 64]),
        ).to(device)
        self.target_net = QNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sizes=self.net_kwargs.get("value_sizes", [64, 64]),
        ).to(device)
        self.policy = StochasticPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            device=device,
            sizes=self.net_kwargs.get("policy_sizes", [64, 64]),
        ).to(device)

        value_checkpoint = torch.load(path + "/value_net.pth")
        policy_checkpoint = torch.load(path + "/policy_net.pth")
        self.behavior_net.load_state_dict(value_checkpoint["model_state_dict"])
        self.target_net.load_state_dict(self.behavior_net.state_dict())
        self.policy.load_state_dict(policy_checkpoint["model_state_dict"])
        self.episodes = value_checkpoint["episodes"]


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

    default_dict = SQLAgent.default_config()
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

    agent = SQLAgent(wandb.config)
    agent.run()
    agent.save_config(wandb.config)
    agent.save_model()
