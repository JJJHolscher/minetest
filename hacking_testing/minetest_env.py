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


class Minetest(gym.Env):
    metadata = {"render.modes": ["rgb_array", "human"]}

    def __init__(
        self,
        socket_port: int = 5555,
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
                    key=key, eventType=dumb_inputs.PRESS if v else dumb_inputs.RELEASE,
                ),
            )

        print("Sending action: {}".format(action))
        self.socket.send(pb_action.SerializeToString())
        print("Waiting for obs...")
        byte_next_obs = self.socket.recv()
        # next_obs = np.frombuffer(byte_next_obs, dtype=np.uint8).reshape(
        #     DISPLAY_SIZE[1],
        #     DISPLAY_SIZE[0],
        #     3,
        # )
        next_obs = None
        self.last_obs = next_obs
        # print("Received obs: {}".format(next_obs.shape))
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
