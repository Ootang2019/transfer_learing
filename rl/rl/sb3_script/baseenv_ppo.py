from drone_env.envs import BaseEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import util

if __name__ == "__main__":

    env = util.subprocvecenv_handle(n_envs=8, env=BaseEnv, env_config={})
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1e6)
    model.save("ppo_test")
