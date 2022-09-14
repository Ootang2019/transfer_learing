import copy
import warnings
from multiprocessing.sharedctypes import Value

import gym
import numpy as np
import torch
import wandb
from rltorch.memory import MultiStepMemory

from agent.common.replay_buffer import MyMultiStepMemory, MyPrioritizedMemory
from agent.common.util import check_dim, check_output_action, np2ts, to_batch, ts2np

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
        obs = self.env.reset()

        while not done:
            if self.render:
                self.env.render()

            action = self.act(obs)
            next_obs, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            if episode_steps >= self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done
            self.replay_buffer.append(
                obs,
                action,
                reward,
                next_obs,
                masked_done,
                episode_done=done,
            )

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            obs = next_obs

        if self.episodes % self.log_interval == 0:
            wandb.log({"reward/train": episode_reward})

    def act(self, obs):
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
        prioritized_memory: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing: float = 0.0001,
        mini_batch_size: int = 128,
        min_n_experience: int = 1024,
        updates_per_step: int = 1,
        train_nround: int = 3,
        render: bool = False,
        log_interval=10,
        eval=True,
        eval_interval=10,
        evaluate_episodes=10,
        task_schedule=None,
        enable_curriculum_learning=False,
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

        self.eval = eval
        self.eval_interval = eval_interval
        self.evaluate_episodes = evaluate_episodes

        if self.env.max_episode_steps is not None:
            self.max_episode_steps = self.env.max_episode_steps
        elif self.env._max_episode_steps is not None:
            self.max_episode_steps = self.env._max_episode_steps
        else:
            self.max_episode_steps = int(1e10)

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.feature_dim = self.env.feature_space.shape[0]

        self.task_idx = 0
        self.prev_ws = []
        self.learn_all_tasks = False
        self.task_schedule = task_schedule
        self.train_nround = train_nround
        self.n_round = 0
        if task_schedule is not None:
            self.budget_per_task = self.total_timesteps / (
                len(self.task_schedule) * self.train_nround
            )
        else:
            self.budget_per_task = self.total_timesteps
        self.update_task()

        self.enable_curriculum_learning = enable_curriculum_learning
        if self.enable_curriculum_learning:
            try:
                self.goal_range = self.env.target_type.get_range_dict()
                self.update_goal_range(scale=1 / self.train_nround)
            except:
                print("env target type has no range_dict attribute")
                self.goal_range = {}
        else:
            self.goal_range = {}

        self.prioritized_memory = prioritized_memory
        if self.prioritized_memory:
            self.replay_buffer = MyPrioritizedMemory(
                int(self.replay_buffer_size),
                self.env.observation_space.shape,
                self.env.feature_space.shape,
                self.env.action_space.shape,
                device,
                self.gamma,
                self.multi_step,
                alpha=alpha,
                beta=beta,
                beta_annealing=beta_annealing,
            )
        else:
            self.replay_buffer = MyMultiStepMemory(
                int(self.replay_buffer_size),
                self.env.observation_space.shape,
                self.env.feature_space.shape,
                self.env.action_space.shape,
                device,
                self.gamma,
                self.multi_step,
            )

        self.learn_steps = 0
        self.episodes = 0
        self.success_rate = 0

    def train_episode(self):
        if self.task_schedule is not None:
            self.change_task()

        episode_reward = 0
        self.episodes += 1
        episode_steps = 0
        done = False
        obs, feature = self.reset_env()

        while not done:
            self.render_env()

            action = self.act(obs)
            next_obs, reward, done, info = self.env.step(action)
            next_feature = info["features"]
            masked_done = False if episode_steps >= self.max_episode_steps else done

            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            self.save_to_buffer(
                done, obs, feature, action, next_obs, reward, masked_done
            )
            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            obs = next_obs
            feature = next_feature

        if self.episodes % self.log_interval == 0:
            self.log_training(episode_reward, episode_steps, info)

        if self.eval and (self.episodes % self.eval_interval == 0):
            self.evaluate()

        self.post_episode_process()

    def render_env(self):
        if self.render:
            try:
                self.env.render()
            except:
                warnings.warn("env has no rendering method")

    def log_training(self, episode_reward, episode_steps, info):
        wandb.log({"reward/train_reward": episode_reward})

        try:
            success_rate = int(info["terminal_info"]["success"]) / episode_steps
            crash_rate = int(info["terminal_info"]["fail"]) / episode_steps
            wandb.log({"reward/train_success_rate": success_rate})
            wandb.log({"reward/train_crash_rate": crash_rate})
        except:
            pass

    def save_to_buffer(self, done, obs, feature, action, next_obs, reward, masked_done):
        if self.prioritized_memory:
            batch = to_batch(
                obs, feature, action, reward, next_obs, masked_done, device
            )
            error = self.calc_priority_error(batch)
            self.replay_buffer.append(
                obs,
                feature,
                action,
                reward,
                next_obs,
                masked_done,
                error,
                done,
            )
        else:
            self.replay_buffer.append(
                obs,
                feature,
                action,
                reward,
                next_obs,
                masked_done,
                done,
            )

    def reset_env(self):
        try:
            obs, info = self.env.reset()
            feature = info["features"]
        except ValueError:
            obs = self.env.reset()
            feature = np.zeros(self.feature_dim)
        return obs, feature

    def evaluate(self):
        episodes = self.evaluate_episodes
        if episodes == 0:
            return

        returns = np.zeros((episodes,), dtype=np.float32)
        success = 0

        for i in range(episodes):
            obs, info = self.reset_env()
            episode_reward = 0.0
            done = False
            while not done:
                action = self.act(obs, "exploit")
                next_obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                obs = next_obs

            returns[i] = episode_reward
            success += int(self.is_success(info))

        self.log_evaluation(episodes, returns, success)

    def log_evaluation(self, episodes, returns, success):
        mean_return = np.mean(returns)
        self.success_rate = success / episodes
        wandb.log({"evaluate/reward": mean_return})
        wandb.log({"evaluate/surrccess_rate": self.success_rate})

    def is_success(self, info):
        success = False
        try:
            if info["terminal_info"]["success"]:
                success = True
        except:
            pass
        return int(success)

    def act(self, obs, mode="explore"):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.get_action(obs, mode)

        action = check_output_action(action, self.action_dim)
        return action

    def get_action(self, obs, mode):
        obs, w = np2ts(obs), np2ts(self.w)
        obs = check_dim(obs, self.observation_dim)

        with torch.no_grad():
            if mode == "explore":
                act_ts = self.explore(obs, w)
            elif mode == "exploit":
                act_ts = self.exploit(obs, w)

        act = ts2np(act_ts)
        return act

    def post_episode_process(self):
        pass

    def calc_priority_error(self, batch):
        raise NotImplementedError

    def explore(self):
        raise NotImplementedError

    def exploit(self):
        raise NotImplementedError

    def change_task(self):
        """if a current task is mastered then update task according to schedule"""
        if self.master_current_task():
            if self.task_idx < len(self.task_schedule) - 1:
                self.task_idx += 1
                self.prev_ws.append(self.env.w)
                print(f"master current task and switch to task{self.task_idx}")
            else:
                self.learn_all_tasks = True
                print("all tasks learned")
        elif self.task_budget_exhaust():
            if self.task_idx < len(self.task_schedule) - 1:
                self.task_idx += 1
                self.prev_ws.append(self.env.w)
                print(f"current task budget exhaust and switch to task{self.task_idx}")
            else:
                self.task_idx = 0
                self.n_round += 1
                self.prev_ws.append(self.env.w)
                self.post_all_tasks_process()
                print(f"current task budget exhaust and switch to task{self.task_idx}!")

        self.update_task()

    def post_all_tasks_process(self):
        if self.enable_curriculum_learning:
            m = np.power(self.train_nround, 1 / (self.train_nround - 1))
            self.update_goal_range(m)

    def update_goal_range(self, scale=1):
        self.goal_range = {k: np.array(v) * scale for k, v in self.goal_range.items()}
        try:
            goal_range = self.modify_goal_range(self.goal_range)
            self.env.target_type.sample_new_goal(goal_range)
        except:
            print("env target_type has no method: sample_new_goal()")
            pass

    def modify_goal_range(self, original_goal_range, max_z=-5, initial_z=-25):
        goal_range = copy.deepcopy(original_goal_range)
        z_range = goal_range["z_range"]

        z_range += initial_z

        if z_range[1] > max_z:
            z_range[1] = max_z

        goal_range["z_range"] = z_range
        return goal_range

    def update_task(self):
        if self.task_schedule is not None:
            try:
                task = self.task_schedule[self.task_idx]
                self.update_task_by_task_dict(task)
            except:
                w = self.task_schedule[self.task_idx]
                self.update_task_by_task_weight(w)
        else:
            try:
                task = self.env.task
                self.update_task_by_task_dict(task)
            except:
                w = self.env.w
                self.update_task_by_task_weight(w)

    def update_task_by_task_dict(self, task):
        """update task by task dictionary"""
        w = self.task_to_w(task)
        self.w = torch.tensor(w, dtype=torch.float32).to(device)
        assert self.w.shape == (self.feature_dim,)

    def task_to_w(self, task):
        self.env.update_tasks(task)
        w = self.env.get_tasks_weights()
        return w

    def update_task_by_task_weight(self, w):
        """update task by task weights"""
        self.env.w = w
        self.w = torch.tensor(w, dtype=torch.float32).to(device)
        assert self.w.shape == (self.feature_dim,)

    def master_current_task(self):
        """is current task mastered?"""
        master = False
        if self.success_rate >= 0.9:
            master = True
            self.success_rate = 0
        return master

    def task_budget_exhaust(self):
        next_task_step = int(self.budget_per_task * (len(self.prev_ws) + 1))
        print("steps", self.steps)
        print("current task", self.task_idx)
        print("switch next task at", next_task_step)
        return self.steps >= next_task_step
