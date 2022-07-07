import datetime
from pathlib import Path
import os.path
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from common.kernel import adaptive_isotropic_gaussian_kernel
from common.policy import StochasticPolicy
from common.replay_buffer import ReplayBuffer
from common.util import assert_shape, get_sa_pairs, get_sa_pairs_
from common.value_function import MLP
from common.agent import AbstractAgent

torch.autograd.set_detect_anomaly(True)  # detect NaN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


class SQLAgent(AbstractAgent):
    """soft q learning with continuous action space"""

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
            kernel_n_particles=54,
            kernel_update_ratio=0.5,
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
        kernel_n_particles: int = 16,
        kernel_update_ratio: float = 0.5,
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
        self.kernel_n_particles = int(kernel_n_particles)
        self.kernel_update_ratio = kernel_update_ratio

        self.net_kwargs = net_kwargs
        self.behavior_net = MLP(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sizes=net_kwargs.get("value_sizes", [64, 64]),
        ).to(device)

        self.target_net = MLP(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sizes=net_kwargs.get("value_sizes", [64, 64]),
        ).to(device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())

        self.policy = StochasticPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            device=device,
            sizes=net_kwargs.get("policy_sizes", [64, 64]),
        ).to(device)

        self.optimizer = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.policy_lr
        )

    def train(self):
        episode_reward = 0
        n_epochs = int(self.total_timesteps / self.n_timesteps)

        for epoch in range(self.epoch, n_epochs):
            self.epoch = epoch
            state = self.env.reset()
            episode_reward = 0

            for _ in range(self.n_timesteps):
                if self.render:
                    self.env.render()
                action = self.policy.get_action(state)

                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                self.replay_buffer.add((state, next_state, action, reward, done))

                if self.replay_buffer.size() > self.min_n_experience:
                    batch = self.sample_minibatch()
                    self.td_update(batch)
                    self.svgd_update(batch)

                if done:
                    break

                state = next_state

            wandb.log({"episode_reward": episode_reward})
            # wandb.watch(self.behavior_net)
            # wandb.watch(self.policy)

    def update_target_net(self):
        self.target_net.load_state_dict(self.behavior_net.state_dict())

    def sample_minibatch(self):
        batch = self.replay_buffer.sample(self.mini_batch_size, False)
        (
            batch_state,
            batch_next_state,
            batch_action,
            batch_reward,
            batch_done,
        ) = zip(*batch)

        batch_state = torch.FloatTensor(np.array(batch_state)).to(device)
        batch_next_state = torch.FloatTensor(np.array(batch_next_state)).to(device)
        batch_action = torch.FloatTensor(np.array(batch_action)).to(device)
        batch_reward = torch.FloatTensor(np.array(batch_reward)).unsqueeze(1).to(device)
        batch_done = torch.FloatTensor(np.array(batch_done)).unsqueeze(1).to(device)
        return batch_state, batch_next_state, batch_action, batch_reward, batch_done

    def td_update(self, batch):
        if self.begin_learn_td is False:
            print("begin learning q function")
            self.begin_learn_td = True

        if self.learn_steps % self.td_target_update_interval == 0:
            self.update_target_net()

        self.learn_steps += 1

        (
            batch_state,
            batch_next_state,
            batch_action,
            batch_reward,
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
        self.optimizer.zero_grad()
        loss = F.mse_loss(q_value, target)
        loss.backward()
        self.optimizer.step()

        metrics = {
            "loss": loss,
            "q_mean": q_value.mean(),
            "q_std": q_value.std(),
            "v_mean": next_value.mean(),
            "target": target.mean(),
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

        self.policy_optimizer.zero_grad()
        loss = -surrogate_loss
        loss.backward()
        self.policy_optimizer.step()

        metrics = {
            "policy_loss": loss,
        }
        wandb.log(metrics)

    def save_model(self):
        path = self.save_path + "/" + str(self.epoch)
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.behavior_net.state_dict(),
            },
            path + "/value_net.pth",
        )
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.policy.state_dict(),
            },
            path + "/policy_net.pth",
        )

    def load_model(self, path):
        self.behavior_net = MLP(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            sizes=self.net_kwargs.get("value_sizes", [64, 64]),
        ).to(device)
        self.target_net = MLP(
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
        self.epoch = value_checkpoint["epoch"]


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

    agent = SQLAgent(**wandb.config)
    agent.train()
    agent.save_config(wandb.config)
    agent.save_model()
