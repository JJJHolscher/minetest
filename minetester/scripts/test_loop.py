#!/usr/bin/env python3
import os

from minetester import Minetest

env = Minetest(
    seed=42,
    start_minetest=True,
    sync_port=30010,
    sync_dtime=0.05,
    headless=True,
    start_xvfb=True,
    clientmods=["random_v0"],
    servermods=["info"]
)

render = True
obs = env.reset()
done = False
while not done:
    try:
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        print(rew, done, info)
        if render:
            env.render()
    except KeyboardInterrupt:
        break
env.close()
