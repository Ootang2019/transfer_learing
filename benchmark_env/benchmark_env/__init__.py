from gym.envs.registration import register

register(
    id="myPendulum-v0",
    entry_point="benchmark_env.envs:MyPendulumEnv",
)
register(
    id="myInvertedDoublePendulum-v4",
    entry_point="benchmark_env.envs:MyInvertedDoublePendulumEnv",
)
