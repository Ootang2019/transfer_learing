import gym
import torch
import numpy as np
import wandb
from rltorch.memory import MultiStepMemory
from agent.common.replay_buffer import MyMultiStepMemory
from agent.common.util import np2ts, ts2np, check_action
import warnings

warnings.simplefilter("once", UserWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AbstractAgent:
    def __init__(
        self,
        env: str,
        env_kwargs: dict = {},
        total_timesteps=1e6,
        render: bool = False,
        log_interval=10,
        seed=0,
        **kwargs,
    ):

        if isinstance(env, str):
            self.env = gym.make(env, **env_kwargs)
        else:
            self.env = env

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.render = render
        self.log_interval = log_interval

        self.steps = 0
        self.learn_steps = 0
        self.episodes = 0
        self.total_timesteps = int(total_timesteps)

    def run(self):
        raise NotImplementedError


class BasicAgent(AbstractAgent):
    def __init__(
        self,
        env: str,
        env_kwargs: dict = {},
        total_timesteps=1000000,
        reward_scale: float = 1,
        replay_buffer_size: int = 1000000,
        gamma: float = 0.99,
        multi_step: int = 1,
        mini_batch_size: int = 128,
        min_n_experience: int = 1024,
        updates_per_step: int = 1,
        render: bool = False,
        log_interval=10,
        seed=0,
        **kwargs,
    ):
        super().__init__(
            env,
            env_kwargs,
            total_timesteps,
            render,
            log_interval,
            seed,
            **kwargs,
        )

        self.reward_scale = reward_scale
        self.replay_buffer_size = int(replay_buffer_size)
        self.mini_batch_size = int(mini_batch_size)
        self.min_n_experience = self.start_steps = int(min_n_experience)
        self.updates_per_step = updates_per_step
        self.gamma = gamma
        self.multi_step = multi_step

        self.replay_buffer = MultiStepMemory(
            self.replay_buffer_size,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            device,
            self.gamma,
            self.multi_step,
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
        observation = self.env.reset()

        while not done:
            if self.render:
                self.env.render()

            action = self.act(observation)
            next_observation, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            if episode_steps >= self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done
            self.replay_buffer.append(
                observation,
                action,
                reward,
                next_observation,
                masked_done,
                episode_done=done,
            )

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            observation = next_observation

        if self.episodes % self.log_interval == 0:
            wandb.log({"reward/train": episode_reward})

    def act(self, observation):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def is_update(self):
        return (
            len(self.replay_buffer) > self.mini_batch_size
            and self.steps >= self.start_steps
        )


class MultiTaskAgent(BasicAgent):
    def __init__(
        self,
        env: str,
        env_kwargs: dict = {},
        total_timesteps=1000000,
        reward_scale: float = 1,
        replay_buffer_size: int = 1000000,
        gamma: float = 0.99,
        multi_step: int = 1,
        mini_batch_size: int = 128,
        min_n_experience: int = 1024,
        updates_per_step: int = 1,
        render: bool = False,
        log_interval=10,
        eval=True,
        eval_interval=50,
        seed=0,
        **kwargs,
    ):
        super().__init__(
            env,
            env_kwargs,
            total_timesteps,
            reward_scale,
            replay_buffer_size,
            gamma,
            multi_step,
            mini_batch_size,
            min_n_experience,
            updates_per_step,
            render,
            log_interval,
            seed,
            **kwargs,
        )
        self.learn_steps = 0
        self.episodes = 0

        self.eval_interval = eval_interval

        if self.env.max_episode_steps is not None:
            self.max_episode_steps = self.env.max_episode_steps
        elif self.env._max_episode_steps is not None:
            self.max_episode_steps = self.env._max_episode_steps
        else:
            self.max_episode_steps = int(1e10)

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.feature_dim = self.env.feature_space.shape[0]
        self.w = np.array(self.env.w)
        self.feature_scale = np.array(self.env.w)

        self.replay_buffer = MyMultiStepMemory(
            int(self.replay_buffer_size),
            self.env.observation_space.shape,
            self.env.feature_space.shape,
            self.env.action_space.shape,
            device,
            self.gamma,
            self.multi_step,
        )
        self.obs_info = {}

    def train_episode(self):
        episode_reward = 0
        self.episodes += 1
        episode_steps = 0
        done = False
        observation, info = self.env.reset()
        feature, self.obs_info = info["features"], info

        while not done:
            if self.render:
                try:
                    self.env.render()
                except:
                    warnings.warn("env has no rendering method")

            action = self.act(observation)
            next_observation, reward, done, info = self.env.step(action)
            next_feature, self.obs_info = info["features"], info
            self.steps += 1
            episode_steps += 1
            episode_reward += reward
            masked_done = False if episode_steps >= self.max_episode_steps else done

            self.replay_buffer.append(
                observation,
                feature,
                action,
                reward,
                next_observation,
                masked_done,
                episode_done=done,
            )

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            observation = next_observation
            feature = next_feature

        if self.episodes % self.log_interval == 0:
            wandb.log({"reward/train": episode_reward})

        if self.steps % self.eval_interval == 0:
            self.evaluate()

    def evaluate(self):
        episodes = 3
        mode = "exploit"
        returns = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            observation, info = self.env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                action = self.act(observation, mode)
                next_observation, reward, done, _ = self.env.step(action)
                episode_reward += reward
                observation = next_observation
            returns[i] = episode_reward

        mean_return = np.mean(returns)

        wandb.log({"reward/test": mean_return})

    def act(self, obs, mode="explore"):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.get_action(obs, mode)
        return action

    def get_action(self, obs, mode):
        obs_ts, w_ts = np2ts(obs), np2ts(self.w)

        if mode == "explore":
            act_ts = self.explore(obs_ts, w_ts)
        elif mode == "exploit":
            act_ts = self.exploit(obs_ts, w_ts)

        act = ts2np(act_ts)
        act = check_action(act, self.action_dim)
        return act

    def explore(self, obs, w):
        raise NotImplementedError

    def exploit(self, obs, w):
        raise NotImplementedError

    def change_task(self):
        raise NotImplementedError

    def master_current_task(self):
        raise NotImplementedError
