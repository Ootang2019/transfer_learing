from stat import SF_IMMUTABLE
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from agent.common.policy import GMMPolicy
from agent.common.value_function import TwinnedSFNetwork
from agent.common.replay_buffer import MultiStepMemory
from agent.common.agent import BasicAgent
from agent.common.util import (
    assert_shape,
    get_sa_pairs,
    get_sa_pairs_,
    hard_update,
    soft_update,
    grad_false,
    update_params,
)
import wandb

EPS = 1e-2

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SuccessorAgent(BasicAgent):
    @classmethod
    def default_config(cls):
        return dict(
            env="myPendulum-v0",
            env_kwargs={},
            total_timesteps=1e5,
            reward_scale=1,
            replay_buffer_size=1e6,
            mini_batch_size=256,
            min_n_experience=1024,
            lr=0.001,
            sp_lr=0.001,
            alpha=0.2,
            gamma=0.99,
            tau=5e-3,
            n_particles=64,
            n_gauss=10,
            grad_clip=None,
            updates_per_step=1,
            td_target_update_interval=1,
            render=False,
            log_interval=10,
            net_kwargs={"value_sizes": [64, 64], "policy_sizes": [64, 64]},
            seed=0,
        )

    def __init__(
        self,
        config: dict = {},
    ) -> None:
        self.config = config
        super().__init__(**config)

        self.lr = config.get("lr", 1e-3)
        self.sp_lr = config.get("sp_lr", 1e-3)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 5e-3)
        self.alpha = torch.tensor(config.get("alpha", 0.2)).to(device)
        self.n_particles = int(config.get("n_particles", 64))
        self.n_gauss = int(config.get("n_gauss", 5))
        self.td_target_update_interval = int(config.get("td_target_update_interval", 1))
        self.updates_per_step = config.get("updates_per_step", 1)
        self.grad_clip = config.get("grad_clip", None)
        self.net_kwargs = config["net_kwargs"]

        self.learn_steps = 0
        self.episodes = 0

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.feature_dim = self.env.feature_space.shape[0]
        self.w = np.array(self.env.w)

        self.replay_buffer = MultiStepMemory(
            int(self.replay_buffer_size),
            self.env.observation_space.shape,
            self.env.feature_space.shape,
            self.env.action_space.shape,
            device,
            self.gamma,
            self.multi_step,
        )

        self.sf = TwinnedSFNetwork(
            observation_dim=self.observation_dim,
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            sizes=self.net_kwargs.get("value_sizes", [64, 64]),
        ).to(device)
        self.sf_target = (
            TwinnedSFNetwork(
                observation_dim=self.observation_dim,
                feature_dim=self.feature_dim,
                action_dim=self.action_dim,
                sizes=self.net_kwargs.get("value_sizes", [64, 64]),
            )
            .to(device)
            .eval()
        )
        hard_update(self.sf_target, self.sf)
        grad_false(self.sf_target)

        self.sp = GMMPolicy(
            state_dim=self.observation_dim,
            feature_dim=self.feature_dim,
            sizes=self.net_kwargs.get("policy_sizes", [32, 32]),
            n_gauss=self.n_gauss,
        ).to(device)

        self.sf1_optimizer = Adam(self.sf.SF1.parameters(), lr=self.lr)
        self.sf2_optimizer = Adam(self.sf.SF2.parameters(), lr=self.lr)
        self.sp_optimizer = Adam(self.sp.parameters(), lr=self.sp_lr)

    def run(self):
        return super().run()

    def train_episode(self):
        episode_reward = 0
        self.episodes += 1
        episode_steps = 0
        done = False
        state, _ = self.env.reset(return_info=True)
        feature = np.zeros(5)

        while not done:
            if self.render:
                self.env.render()

            action = self.act(state)
            next_state, reward, done, info = self.env.step(action)
            next_feature = info["features"]
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            self.replay_buffer.append(
                state,
                feature,
                action,
                reward,
                next_state,
                masked_done=done,
                episode_done=done,
            )

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            state = next_state
            feature = next_feature

        if self.episodes % self.log_interval == 0:
            wandb.log({"reward/train": episode_reward})

    def act(self, obs):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.get_action(obs)
        return action

    def get_action(self, obs):
        w = self.w
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
        if isinstance(w, np.ndarray):
            w = torch.tensor(w, dtype=torch.float32).to(device)

        act, _ = self.sp.act(obs, w)
        act = act.cpu().detach().numpy()
        assert act.shape == (self.action_dim,)
        return act

    def learn(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(self.sf_target, self.sf, self.tau)

        batch = self.replay_buffer.sample(self.mini_batch_size)

        sf1_loss, sf2_loss, mean_sf1, mean_sf2 = self.calc_sf_loss(batch)
        policy_loss = self.calc_policy_loss(batch)

        update_params(self.sf1_optimizer, self.sf.SF1, sf1_loss, self.grad_clip)
        update_params(self.sf2_optimizer, self.sf.SF2, sf2_loss, self.grad_clip)
        update_params(self.sp_optimizer, self.sp, policy_loss, self.grad_clip)

        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "loss/SF1": sf1_loss.detach().item(),
                "loss/SF2": sf2_loss.detach().item(),
                "loss/policy": policy_loss.detach().item(),
                "state/mean_SF1": mean_sf1,
                "state/mean_SF2": mean_sf2,
            }
            wandb.log(metrics)

    def calc_sf_loss(self, batch):
        (
            batch_state,
            batch_feature,
            batch_action,
            batch_reward,
            batch_next_state,
            batch_done,
        ) = batch

        batch_reward *= self.reward_scale

        cur_sf1, cur_sf2 = self.sf(batch_state, batch_action)
        cur_sf1, cur_sf2 = cur_sf1.squeeze(), cur_sf2.squeeze()

        next_sf = self.calc_next_sf(batch_next_state)
        target_sf, next_sfv = self.calc_target_sf(next_sf, batch_feature, batch_done)

        mean_sf1 = cur_sf1.detach().mean().item()
        mean_sf2 = cur_sf2.detach().mean().item()

        sf1_loss = F.mse_loss(cur_sf1, target_sf)
        sf2_loss = F.mse_loss(cur_sf2, target_sf)

        return sf1_loss, sf2_loss, mean_sf1, mean_sf2

    def calc_next_sf(self, batch_next_state):
        sample_action = (
            torch.distributions.uniform.Uniform(-1, 1)
            .sample((self.n_particles, self.action_dim))
            .to(device)
        )

        s, a = get_sa_pairs(batch_next_state, sample_action)
        next_sf1, next_sf2 = self.sf_target(s, a)

        n_sample = batch_next_state.shape[0]
        next_sf1 = next_sf1.view(n_sample, -1, self.feature_dim)
        next_sf2 = next_sf2.view(n_sample, -1, self.feature_dim)
        next_sf = torch.min(next_sf1, next_sf2)
        next_sf /= self.alpha
        return next_sf

    def calc_target_sf(self, next_sf, batch_feature, batch_done):
        next_sfv = torch.logsumexp(next_sf, 1)
        next_sfv -= np.log(self.n_particles)
        next_sfv += self.action_dim * np.log(2)
        next_sfv *= self.alpha

        target_sf = (
            (batch_feature + (1 - batch_done) * self.gamma * next_sfv)
            .squeeze(1)
            .detach()
        )
        return target_sf, next_sfv

    def calc_policy_loss(self, batch):
        (batch_state, _, _, _, _, _) = batch
        chi, logp = self.sp.get_chi(batch_state)

        sf1, sf2 = self.sf_target.forward_chi(batch_state, chi)
        sf = torch.min(sf1, sf2)

        reg_loss = self.sp.reg_loss_t
        loss = torch.mean(logp - 1 / self.alpha * sf) + reg_loss
        return loss

    def squash_correction(self, inp):
        return torch.sum(torch.log(1 - torch.tanh(inp) ** 2 + EPS), 1)


if __name__ == "__main__":
    # ============== profile ==============#
    # pip install snakeviz
    # python -m cProfile -o out.profile rl/rl/agent/sa.py -s time
    # snakeviz sa.profile

    # ============== sweep ==============#
    # wandb sweep rl/rl/sweep_sa.yaml
    import gym
    import benchmark_env

    env = "myInvertedDoublePendulum-v4"
    env_kwargs = {}
    render = False

    default_dict = SuccessorAgent.default_config()
    default_dict.update(
        dict(
            env=env,
            env_kwargs=env_kwargs,
            render=render,
        )
    )

    wandb.init(config=default_dict)
    print(wandb.config)

    agent = SuccessorAgent(wandb.config)
    agent.run()
