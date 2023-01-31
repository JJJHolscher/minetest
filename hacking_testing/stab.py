gimport pickle
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete
from gym.wrappers import Monitor, TimeLimit
from minetest_env import Minetest
from stable_baselines3 import PPO
from typing import Optional, Dict, Any
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gym
import gc
import sys
if "./vpt-minetest" not in sys.path:
    sys.path.append("./vpt-minetest")
from agent import MineRLAgent
from run_vpt_agent import minetest_to_minerl_obs, minerl_to_minetest_action


model, weights, video_dir, minetest_path, max_steps, show, seed, show_agent_pov = "2x.model", "foundation-model-2x.weights", "videos", "../bin/minetest", 100000, False, 32, False



class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env, discretes=27):
        self.env = env
        self.discretes = discretes
        sizes = []
        self.vals = []
        for i, v in env.action_space.spaces.items():
            self.vals.append(len(sizes))
            if isinstance(v, Discrete):
                sizes.append(v.n)
            elif isinstance(v, Box):
                for _ in v.low:
                    sizes.append(discretes)
                
        self.action_space = MultiDiscrete(sizes)  # TODO
    
    def action(self, act):
        return {k: (np.asarray(act[i:i+len(v.low)]) / self.discretes * (v.high - v.low) + v.low
                    if isinstance(v, Box) else act[i]).astype(v.dtype)
                for i, (k, v) in zip(self.vals, self.env.action_space.spaces.items())}

def make_env(
    minetest_path: str,
    rank: int,
    seed: int = 0,
    max_steps: int = 1e9,
    env_kwargs: Optional[Dict[str, Any]] = None,
):
    env_kwargs = env_kwargs or {}

    def _init():
        # Make sure that each Minetest instance has
        # - different server and client ports
        # - different and deterministic seeds
        env = Minetest(
            env_port=5555 + rank,
            server_port=30000 + rank,
            # seed=seed + rank,
            world_dir=f"../worlds/myworld{rank}",
            minetest_executable=minetest_path,
            # xvfb_headless=True,
            config_path="../minetest.conf",
            **env_kwargs,
        )
        env.reset_world = True
        env = TimeLimit(env, max_episode_steps=max_steps)
        env = DiscreteActions(env)
        return env

    return _init

# Env settings
seed = 42
max_steps = 1000
env_kwargs = {"display_size": (1024, 600), "fov": 72}

# Create a vectorized environment
num_envs = 2  # Number of envs to use (<= number of avail. cpus)
# vec_env_cls = SubprocVecEnv
vec_env_cls = DummyVecEnv
venv = vec_env_cls(
    [
        make_env(minetest_path=minetest_path, rank=i, seed=seed, max_steps=max_steps, env_kwargs=env_kwargs)
        for i in range(num_envs)
    ],
)

print("---Loading model---")
agent_parameters = pickle.load(open(model, "rb"))
policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
agent = MineRLAgent(
    venv,
    policy_kwargs=policy_kwargs,
    pi_head_kwargs=pi_head_kwargs,
    show_agent_perspective=show_agent_pov,
)
agent.load_weights(weights)

from wandb.integration.sb3 import WandbCallback
# ppo = PPO("CnnPolicy", venv, verbose=1, callback=WandbCallback())
ppo = PPO("CnnPolicy", venv, verbose=1)
ppo.learn(total_timesteps=25000)

# print("---Launching Minetest enviroment---")
# obs = minetest_to_minerl_obs(env.reset())
# done = False
# while not done:
#     minerl_action = agent.get_action(obs)
#     minetest_action = minerl_to_minetest_action(minerl_action, env)
#     obs, reward, done, info = env.step(minetest_action)
#     obs = minetest_to_minerl_obs(obs)
#     if show:
#         env.render()
# env.close()
