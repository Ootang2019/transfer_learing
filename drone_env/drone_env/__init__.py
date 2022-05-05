""" register gym environment """
from gym.envs.registration import register

register(
    id="base-v0",
    entry_point="drone_env.envs:BaseEnv",
)
