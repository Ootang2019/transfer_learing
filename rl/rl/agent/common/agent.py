import gym
import torch
import numpy as np
import wandb
from rltorch.memory import MultiStepMemory

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
        state = self.env.reset()

        while not done:
            if self.render:
                self.env.render()

            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            if episode_steps >= self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done
            self.replay_buffer.append(
                state, action, reward, next_state, masked_done, episode_done=done
            )

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            state = next_state

        if self.episodes % self.log_interval == 0:
            wandb.log({"reward/train": episode_reward})

    def act(self, state):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def is_update(self):
        return (
            len(self.replay_buffer) > self.mini_batch_size
            and self.steps >= self.start_steps
        )
