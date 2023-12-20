from copy import deepcopy
from typing import  Optional, Tuple, Union
from time import time
from base.space_converter import SpaceConverter
from exceptions import *
from experiment_scripts.rl.rec_env_rl_libs_wrapper import RecEnvRlLibsWrapper
from experiment_scripts.rl.space_converters.sequential_space_converter import SequentialSpaceConverter
from env.rec_env import RecEnv
from gymnasium.envs.registration import EnvSpec
from gymnasium import Env
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Space, Sequence, Box, Discrete, Dict as DictSpace, Tuple as TupleSpace, MultiDiscrete

from gym.envs.registration import EnvSpec as OldEnvSpec
from gym import Env as OldEnv
from gym.core import ObsType as OldObsType, ActType as OldActType
from gym.spaces import Space as OldSpace, Sequence as OldSequence, Box as OldBox, MultiDiscrete as OldMultiDiscrete, Discrete as OldDiscrete, Dict as OldDictSpace, Tuple as OldTupleSpace

from utils.utils import epsilonify
from .space_converters_zoo import create_space_converter_sequence
import numpy as np

def gym_to_gymnasium_space(space: OldSpace) -> Space:
    if type(space) == OldBox:
        return Box(space.low, space.high, shape=space.shape, dtype=space.dtype)
    elif type(space) == OldDiscrete:
        return Discrete(space.n, start=space.start)
    elif type(space) == OldMultiDiscrete:
        return MultiDiscrete(space.nvec, dtype=space.dtype)
    elif type(space) == OldTupleSpace:
        return TupleSpace([gym_to_gymnasium_space(subspace) for subspace in space.spaces])
    elif type(space) == OldDictSpace:
        return DictSpace({subspace_key:gym_to_gymnasium_space(subspace) for subspace_key, subspace in space.items()})
    else:
        raise NotImplementedError(type(space))

class RecEnvRlLibsGymnasiumGymWrapper(Env):

    def __init__(self, rec_env:RecEnvRlLibsWrapper):
        self._wrapped_rec_env = deepcopy(rec_env)
        self.observation_space = gym_to_gymnasium_space(self._wrapped_rec_env.observation_space)
        self.action_space = gym_to_gymnasium_space(self._wrapped_rec_env.action_space)
        self.reward_range = Box(self._wrapped_rec_env.reward_range.low, self._wrapped_rec_env.reward_range.high)
        self.spec = EnvSpec(
            "RecEnvGymnasium-0", None, max_episode_steps=self._wrapped_rec_env.spec.max_episode_steps
        )

    def reset(self, *, seed=None, options=None) -> Tuple[ObsType, dict]:
        obs = self._wrapped_rec_env.reset()
        return obs, {"eval_env": self.eval_env}
    
    def step(self, action: ActType):
        #t = time()
        next_observation, reward, _, info = self._wrapped_rec_env.step(action)
        #print("Global env step took", time()-t, "seconds")
        return next_observation, reward, info["terminated"] or info["truncated"], info["truncated"] or info["terminated"], info
    
    def space_convert_obs(self, obs, action=None, reward=None):
        return self._wrapped_rec_env.space_convert_obs(obs, action=action, reward=reward)
    
    def close(self):
        pass

    def render(self, *args, **kwargs):
        pass
    
    def space_convert_act(self, action, obs=None, reward=None):
        return self._wrapped_rec_env.space_convert_act(action, obs=obs, reward=reward)

    @property
    def T(self):
        return self._wrapped_rec_env.T
    
    @property
    def members_with_controllable_assets(self):
        return self._wrapped_rec_env.members_with_controllable_assets
    
    @property
    def env_name(self):
        return self._wrapped_rec_env.env_name
    
    @property
    def eval_env(self):
        return self._wrapped_rec_env._eval_env
    
    @property
    def space_converters(self):
        return self._wrapped_rec_env._space_converters
    
    @property
    def wrapped_rec_env(self):
        return self._wrapped_rec_env
    
    @property
    def obs_z_score_mean(self):
        return self._wrapped_rec_env._obs_z_score_mean
    
    @property
    def obs_z_score_std(self):
        return self._wrapped_rec_env._obs_z_score_std
    
    @obs_z_score_mean.setter
    def obs_z_score_mean(self, new_obs_z_score_mean):
        self._wrapped_rec_env.obs_z_score_mean = new_obs_z_score_mean

    @obs_z_score_std.setter
    def obs_z_score_std(self, new_obs_z_score_std):
        self._wrapped_rec_env.obs_z_score_std = new_obs_z_score_std

    @property
    def rew_z_score_mean(self):
        return self._wrapped_rec_env._rew_z_score_mean
    
    @property
    def type_solver(self):
        return self._wrapped_rec_env.type_solver
    
    @property
    def rew_z_score_std(self):
        return self._wrapped_rec_env._rew_z_score_std
    
    @rew_z_score_mean.setter
    def rew_z_score_mean(self, new_rew_z_score_mean):
        self._wrapped_rec_env.rew_z_score_mean = new_rew_z_score_mean

    @rew_z_score_std.setter
    def rew_z_score_std(self, new_rew_z_score_std):
        self._wrapped_rec_env.rew_z_score_std = new_rew_z_score_std

    