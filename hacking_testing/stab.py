import pickle
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from gym.spaces import Dict as DictOAI, Box, Discrete, MultiDiscrete
from gym.wrappers import Monitor, TimeLimit, FrameStack
from minetest_env import Minetest
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from typing import Optional, Dict, Any, List, Tuple
from stable_baselines3.common.distributions import (
    Distribution,
    MultiCategoricalDistribution,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from gym_wrappers import (DictToMultiDiscreteActionSpace,
                          HiddenStateObservationSpace, ObservationToCPU,
                          ObservationToInfos)
from stable_baselines3.common.type_aliases import Schedule
import torch
import gym
import gc
import sys
if "./vpt-minetest" not in sys.path:
    sys.path.append("./vpt-minetest")
from agent import MineRLAgent
from run_vpt_agent import minetest_to_minerl_obs, minerl_to_minetest_action


model, weights, video_dir, minetest_path, max_steps, show, seed, show_agent_pov = "2x.model", "foundation-model-2x.weights", "videos", "../bin/minetest", 100000, False, 32, False


# class DiscreteActions(gym.ActionWrapper):
#     def __init__(self, env, discretes=64):
#         self.env = env
#         self.discretes = discretes
#         sizes = []
#         self.vals = []
#         for i, v in env.action_space.spaces.items():
#             self.vals.append(len(sizes))
#             if isinstance(v, Discrete):
#                 sizes.append(v.n)
#             elif isinstance(v, Box):
#                 for _ in v.low:
#                     sizes.append(discretes)
                
#         self.action_space = MultiDiscrete(sizes)  # TODO
    
#     def action(self, act):
#         return {k: (np.asarray(act[i:i+len(v.low)]) / self.discretes * (v.high - v.low) + v.low
#                     if isinstance(v, Box) else act[i]).astype(v.dtype)
#                 for i, (k, v) in zip(self.vals, self.env.action_space.spaces.items())}

def _env_action_to_agent(self, minerl_action_transformed, action_transformer, action_mapper, to_torch=False, check_if_null=False):
    """
    Turn action from MineRL to model's action.

    Note that this will add batch dimensions to the action.
    Returns numpy arrays, unless `to_torch` is True, in which case it returns torch tensors.

    If `check_if_null` is True, check if the action is null (no action) after the initial
    transformation. This matches the behaviour done in OpenAI's VPT work.
    If action is null, return "None" instead
    """
    minerl_action = action_transformer.env2policy(minerl_action_transformed)
    if check_if_null:
        if np.all(minerl_action["buttons"] == 0) and np.all(minerl_action["camera"] == action_transformer.camera_zero_bin):
            return None

    # Add batch dims if not existant
    if minerl_action["camera"].ndim == 1:
        minerl_action = {k: v[None] for k, v in minerl_action.items()}
    action = action_mapper.from_factored(minerl_action)
    if to_torch:
        action = {k: th.from_numpy(v).to(self.device) for k, v in action.items()}
    return action

class DictToMultiDiscreteActionSpace(gym.Wrapper):
    """Converts Dict to MultiDiscrete action space for MineRL envs"""

    def __init__(self, env, action_transformer, action_mapper):
        super().__init__(env)
        print(self.env.action_space)

        if not isinstance(self.env.action_space, DictOAI):
            raise ValueError("Original action space is not of type gym.Dict.")

        # assert minerl_agent is not None

        # self.minerl_agent = minerl_agent
        self.action_transformer = action_transformer
        self.action_mapper = action_mapper

        # first dimension = camera, second dimension = buttons
        self.action_space = MultiDiscrete([121, 8641])

        # check action space conversion
        # random_env_action = self.env.action_space.sample()
        # agent_action = self.minerl_agent._env_action_to_agent(random_env_action)
        # array_action = self.to_array_action(agent_action)
        # assert array_action.squeeze() in self.action_space

        # agent_action2 = self.to_agent_action(array_action)
        # env_action = self.minerl_agent._agent_action_to_env(agent_action2)
        # env_action["ESC"] = env_action["swapHands"] = env_action["pickItem"] = np.array(
            # 0
        # )
        # assert env_action in self.env.action_space

    def to_array_action(self, agent_action):
        if isinstance(agent_action["camera"], np.ndarray):
            array_action = np.concatenate(
                (agent_action["camera"], agent_action["buttons"]), -1
            )
        elif isinstance(agent_action["camera"], th.Tensor):
            array_action = th.cat(
                (agent_action["camera"], agent_action["buttons"]), dim=-1
            )
        return array_action

    def to_agent_action(self, array_action):
        agent_action = {"camera": array_action[..., 0], "buttons": array_action[..., 1]}
        return agent_action

    def step(self, action):
        if len(action.shape) < 2:
            action = action[np.newaxis, :]
        # transform array action to agent action to MineRL action
        agent_action = self.to_agent_action(action)
        minerl_action = self.minerl_agent._agent_action_to_env(agent_action)

        # TODO implement policy that controls the remaining actions (especially ESC):
        minerl_action["ESC"] = np.array(0)
        minerl_action["swapHands"] = minerl_action["pickItem"] = np.array(0)

        obs, reward, terminated, info = self.env.step(minerl_action)

        return obs, reward, terminated, info

def make_env(
    minetest_path: str,
    rank: int,
    seed: int = 0,
    max_steps: int = 1e9,
    env_kwargs: Optional[Dict[str, Any]] = None,
    action_transformer=None, action_mapper=None
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
            # xvfb_headless=False,
            config_path="../minetest.conf",
            **env_kwargs,
        )
        env.reset_world = True
        env = TimeLimit(env, max_episode_steps=max_steps)
        # env = DiscreteActions(env)
        # env = ObservationToInfos(env)
        env = DictToMultiDiscreteActionSpace(env, 
            action_transformer=action_transformer,
            action_mapper=action_mapper,)
        env = FrameStack(env, 128)
        # env = HiddenStateObservationSpace(env, minerl_agent)
        # env = ObservationToCPU(env)

        return env

    return _init

# Env settings
seed = 42
max_steps = 4
env_kwargs = {"display_size": (1024, 600), "fov": 72}

print("---Loading model---")
agent_parameters = pickle.load(open(model, "rb"))
policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
agent = MineRLAgent(
    None,
    policy_kwargs=policy_kwargs,
    pi_head_kwargs=pi_head_kwargs,
    show_agent_perspective=show_agent_pov,
)
# print(vars(agent).keys())
# agent.load_weights(weights)
agent_kwargs = dict(
    policy_kwargs=policy_kwargs,
    pi_head_kwargs=pi_head_kwargs,
    show_agent_pov=show_agent_pov,
)

# Create a vectorized environment
num_envs = 2  # Number of envs to use (<= number of avail. cpus)
# vec_env_cls = SubprocVecEnv
vec_env_cls = DummyVecEnv
venv = vec_env_cls(
    [
        make_env(minetest_path=minetest_path, rank=i, seed=seed, max_steps=max_steps, env_kwargs=env_kwargs,
                 action_transformer=agent.action_transformer, action_mapper=agent.action_mapper)
        for i in range(num_envs)
    ],
)

from wandb.integration.sb3 import WandbCallback

class MinecraftActorCriticPolicy(ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms wrapping OpenAI's VPT models.
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param minerl_agent: MineRL agent to be wrapped
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        minerl_agent: MineRLAgent,
        # policy_kwargs: Dict = {},
        # pi_head_kwargs: Dict = {},
        # show_agent_pov: bool = False,
        **kwargs
    ):

        # self.minerl_agent = minerl_agent
        # print(policy_kwargs)
        self.minerl_agent = minerl_agent
        # self.minerl_agent = MineRLAgent(
        #     None,
        #     # venv,
        #     policy_kwargs=policy_kwargs,
        #     pi_head_kwargs=pi_head_kwargs,
        #     show_agent_perspective=show_agent_pov,
        # )

        super(MinecraftActorCriticPolicy, self).__init__(
            observation_space, action_space, lr_schedule, **kwargs
        )

        self.ortho_init = False

    def forward(
        self, observation: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        # unpack observation
        obs, first, state_in = self.unpack_dict_obs(observation)

        # inference
        (pi_logits, vpred, _), state_out = self.minerl_agent.policy(
            obs, first, state_in
        )

        # update MineRLAgent's hidden state (important: only do this in forward()!)
        self.minerl_agent.hidden_state = state_out

        # action sampling
        action = self.action_net.sample(pi_logits, deterministic=deterministic)

        value = self.value_net.denormalize(vpred)[:, 0]
        log_prob = self.action_net.logprob(action, pi_logits)

        # convert agent action into array so it can pass through the SB3 functions
        array_action = torch.cat((action["camera"], action["buttons"]), dim=-1)

        return array_action.squeeze(1), value, log_prob

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        # Setup action and value heads
        self.action_net = self.minerl_agent.policy.pi_head
        self.value_net = self.minerl_agent.policy.value_head
        # print(self.minerl_agent.policy)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.
            minerl_agent.policy.net.img_process.cnn.stacks[0].firstconv  # .pi_head
            .parameters()
            , lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def evaluate_actions(
        self, obs: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        # convert array actions to agent actions
        agent_actions = {"camera": actions[..., 0], "buttons": actions[..., 1]}

        # unpack observation
        img_obs, first, state_in = self.unpack_dict_obs(obs)

        # inference
        (pi_logits, vpred, _), state_out = self.minerl_agent.policy(
            img_obs, first, state_in
        )

        value = self.value_net.denormalize(vpred)[:, 0]
        log_prob = self.action_net.logprob(agent_actions, pi_logits)
        entropy = self.action_net.entropy(pi_logits)

        return value, log_prob, entropy

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        :param obs:
        :return: the estimated values.
        """

        # unpack observation
        img_obs, first, state_in = self.unpack_dict_obs(obs)

        # inference
        (_, latent_vf), state_out = self.minerl_agent.policy.net(
            img_obs, state_in, {"first": first}
        )
        value = self.value_net(latent_vf)

        return value

    def get_distribution(self, obs: Dict[str, torch.Tensor]) -> Distribution:
        """
        Get the current policy distribution given the observations.
        :param obs:
        :return: the action distribution.
        """
        # unpack observation
        img_obs, first, state_in = self.unpack_dict_obs(obs)

        # inference
        (latent_pi, _), state_out = self.minerl_agent.policy.net(
            img_obs,
            state_in,
            {"first": first},
        )
        # features = self.extract_features(obs)
        # latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        # convert mean agent actions to mean array actions
        mean_array_actions = (
            torch.cat((mean_actions["camera"], mean_actions["buttons"]), -1)
            .squeeze(0)
            .squeeze(0)
        )

        print(self.action_dist, mean_array_actions.shape)
        if isinstance(self.action_dist, MultiCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_array_actions)
        else:
            raise ValueError("Invalid action distribution")

    def unpack_dict_obs(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        """
        Unpack the observation dictionary
        :param obs:
        :return: the agent image observation, first input tensor and the hidden state
        """

        first = torch.zeros((obs.shape[0], 16)).to(self.minerl_agent.device)
        first[:, 0] = 1
        first = first.bool()
        state = self.minerl_agent.policy.net.initial_state(obs.shape[0])
        return {"img": obs}, first, state

policy_kwargs = dict(minerl_agent=agent)  # **agent_kwargs)
# ppo = PPO("CnnPolicy", venv, verbose=1, callback=WandbCallback())
# ppo = PPO("CnnPolicy", venv, verbose=1, batch_size=4, n_steps=8)
ppo = PPO(MinecraftActorCriticPolicy, venv, verbose=1, policy_kwargs=policy_kwargs,
          batch_size=2, n_steps=1)
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
