import os
import subprocess
import uuid

import gym
import matplotlib.pyplot as plt
import numpy as np
import zmq
from gym.spaces import Box, Dict, Discrete
from proto_python.client import dumb_inputs_pb2 as dumb_inputs

# TODO read from the minetest.conf file
DISPLAY_SIZE = (1024, 600)
FOV_Y = 72  # degrees
FOV_X = FOV_Y * DISPLAY_SIZE[0] / DISPLAY_SIZE[1]
MAX_MOUSE_MOVE_X = 180 / FOV_X * DISPLAY_SIZE[0]
MAX_MOUSE_MOVE_Y = 180 / FOV_Y * DISPLAY_SIZE[1]

KEYS = [
    "FORWARD",
    "BACKWARD",
    "LEFT",
    "RIGHT",
    "JUMP",
    "SNEAK",
    "DIG",  # left mouse
    "MIDDLE",  # middle mouse
    "PLACE",  # right mouse
    "DROP",
    "HOTBAR_NEXT",
    "HOTBAR_PREVIOUS",
    "SLOT1",
    "SLOT2",
    "SLOT3",
    "SLOT4",
    "SLOT5",
    "SLOT6",
    "SLOT7",
    "SLOT8",
    # these keys open the inventory/menu
    "ESC",
    "INVENTORY",
    # "AUX1",
    # these keys lead to errors:
    # "CHAT", "CMD",
    # these keys are probably uninteresting
    # "zoom",
    # "autoforward",
    # "pitchmove",
    # "freemove",
    # "fastmove",
    # "noclip",
    # "screenshot",
]


def start_minetest_server(
    minetest_path: str = "bin/minetest",
    config_path: str = "minetest.conf",
    log_dir: str = "log",
    world: str = None,
):
    if world:
        world_path = world
    else:
        world_path = str(uuid.uuid4())
    cmd = [
        minetest_path,
        "--server",
        "--world",
        world_path,
        "--gameid",
        "minetest",
        "--config",
        config_path,
    ]
    os.makedirs(log_dir, exist_ok=True)
    stdout_file = os.path.join(log_dir, "server_stdout.log")
    stderr_file = os.path.join(log_dir, "server_stderr.log")
    with open(stdout_file, "w") as out, open(stderr_file, "w") as err:
        server_process = subprocess.Popen(cmd, stdout=out, stderr=err)
    return server_process


def start_minetest_client(
    minetest_path: str = "bin/minetest",
    log_dir: str = "log",
    client_port: int = 5555,
    cursor_img: str = "cursors/mouse_cursor_white_16x16.png",
    client_name: str = "MinetestAgent",
    headless: bool = True
):
    cmd = [
        minetest_path,
        "--name",
        client_name,
        "--password",
        "1234",
        "--address",
        "0.0.0.0",  # listen to all interfaces
        "--port",
        "30000",  # TODO this should be the same as in minetest.conf
        "--go",
        "--dumb",
        "--client-address",
        "tcp://localhost:" + str(client_port),
        "--record",
        "--noresizing",
    ]
    if cursor_img:
        cmd.extend(["--cursor-image", cursor_img])
    if headless:
        cmd = ["xvfb-run", "-a", "-n", "1", "-s" "-screen 0 1024x768x16"] + cmd

    os.makedirs(log_dir, exist_ok=True)
    stdout_file = os.path.join(log_dir, "client_stdout.log")
    stderr_file = os.path.join(log_dir, "client_stderr.log")
    with open(stdout_file, "w") as out, open(stderr_file, "w") as err:
        client_process = subprocess.Popen(cmd, stdout=out, stderr=err)
    return client_process


class Minetest(gym.Env):
    metadata = {"render.modes": ["rgb_array", "human"]}

    def __init__(
        self,
        socket_port: int = 5555,
        minetest_executable: os.PathLike = None, 
        log_dir: os.PathLike = None,
        config_path: os.PathLike = None,
        headless: bool = True
    ):
        # Define action and observation space
        self.action_space = Dict(
            {
                **{key.lower(): Discrete(2) for key in KEYS},
                **{
                    "mouse": Box(
                        np.array([-MAX_MOUSE_MOVE_X, -MAX_MOUSE_MOVE_Y]),
                        np.array([MAX_MOUSE_MOVE_X, MAX_MOUSE_MOVE_Y]),
                        shape=(2,),
                        dtype=int,
                    ),
                },
            },
        )
        self.observation_space = Box(0, 255, shape=(*DISPLAY_SIZE, 3), dtype=np.uint8)

        root_dir = os.path.dirname(os.path.dirname(__file__))
        if minetest_executable is None:
            minetest_executable = os.path.join(root_dir, "bin", "minetest")
        if log_dir is None:
            log_dir = os.path.join(root_dir, "log")
        if config_path is None:
            config_path = os.path.join(root_dir, "minetest.conf")

        # Start Minetest server and client
        self.server_process = start_minetest_server(
            minetest_executable, config_path, log_dir, "newworld"
        )
        self.client_process = start_minetest_client(minetest_executable, log_dir, socket_port, headless=headless)

        # Setup ZMQ
        self.socket_port = socket_port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.socket_port}")

        self.last_obs = None
        self.render_fig = None
        self.render_img = None

    def reset(self):
        print("Waiting for obs...")
        byte_obs = self.socket.recv()
        obs = np.frombuffer(byte_obs, dtype=np.uint8).reshape(
            DISPLAY_SIZE[1],
            DISPLAY_SIZE[0],
            3,
        )
        self.last_obs = obs
        print("Received obs: {}".format(obs.shape))
        return obs

    def step(self, action):
        # make mouse action serializable
        pb_action = dumb_inputs.InputAction()
        pb_action.mouseDx, pb_action.mouseDy = action["mouse"]
        for key, v in action.items():
            if key == "mouse":
                continue
            pb_action.keyEvents.append(
                dumb_inputs.KeyboardEvent(
                    key=key,
                    eventType=dumb_inputs.PRESS if v else dumb_inputs.RELEASE,
                ),
            )

        print("Sending action: {}".format(action))
        self.socket.send(pb_action.SerializeToString())

        # TODO more robust check for whether a server/client is alive while receiving observations
        for process in [self.server_process, self.client_process]:
            if process.poll() is not None:
                return self.last_obs, 0.0, True, {}
        
        print("Waiting for obs...")
        byte_next_obs = self.socket.recv()
        next_obs = np.frombuffer(byte_next_obs, dtype=np.uint8).reshape(
            DISPLAY_SIZE[1],
            DISPLAY_SIZE[0],
            3,
        )
        self.last_obs = next_obs
        print("Received obs: {}".format(next_obs.shape))
        # TODO receive rewards etc.
        rew = 0.0
        done = False
        info = {}
        return next_obs, rew, done, info

    def render(self, render_mode: str = "human"):
        if render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")',
            )
            return
        if render_mode == "human":
            if self.render_img is None:
                # Setup figure
                plt.rcParams["toolbar"] = "None"
                plt.rcParams["figure.autolayout"] = True

                self.render_fig = plt.figure(
                    num="Minetest",
                    figsize=(3 * DISPLAY_SIZE[0] / DISPLAY_SIZE[1], 3),
                )
                self.render_img = self.render_fig.gca().imshow(
                    self.last_obs,
                )
                self.render_fig.gca().axis("off")
                self.render_fig.gca().margins(0, 0)
                self.render_fig.gca().autoscale_view()
            else:
                self.render_img.set_data(self.last_obs)
            plt.draw(), plt.pause(1e-3)
        elif render_mode == "rgb_array":
            return self.last_obs

    def close(self):
        if self.render_fig is not None:
            plt.close()
        self.socket.close()
        self.client_process.kill()
        self.server_process.kill()
