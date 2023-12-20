from copy import deepcopy
from typing import  Optional, Tuple, Union
from gym import Env
from gym.core import ObsType, ActType
from gym.spaces import Space, Sequence, Box, Dict as DictSpace, Tuple as TupleSpace, Discrete
from base.space_converter import SpaceConverter
from exceptions import *
from experiment_scripts.rl.space_converters.sequential_space_converter import SequentialSpaceConverter
from env.rec_env import RecEnv
from gym.envs.registration import EnvSpec

from utils.utils import epsilonify
from .space_converters_zoo import create_space_converter_sequence
import numpy as np
from time import time

class RecEnvRlLibsWrapper(Env):

    def __init__(self, rec_env:RecEnv, space_converter_ids:str, eval_env=False, observe_current_peaks=False, members_with_controllable_assets=[], **kwargs):
        self._wrapped_rec_env = deepcopy(rec_env)

        """
            Create initial observation space
        """
        # Change tuple keys to string keys
        # Replace sequences by inner space
        self._reward_space = DictSpace(
            {
                "metering_period_cost": Box(-100000.0, 100000.0),
                "peak_period_cost": Box(-100000.0, 100000.0),
                "controllable_assets_costs": Box(-100000.0, 100000.0)
            }
        )
        self._observe_current_peaks = observe_current_peaks
        self._members_with_controllable_assets = members_with_controllable_assets
        self._space_converters:SequentialSpaceConverter = create_space_converter_sequence(
            rec_env, rec_env.observation_space, rec_env.action_space, self._reward_space, space_converter_sequence_ids=space_converter_ids
        )
        self.observation_space = self._space_converters.convert_observation_space()
        self.action_space = self._space_converters.convert_action_space()
        self.reward_range = Box(-10000, 10000)
        
        self.spec = EnvSpec(
            "RecEnv-0", None, max_episode_steps=self._wrapped_rec_env.T
        )
        self._previous_reward = None
        self._previous_observation = None
        self._previous_action=None
        self._eval_env = eval_env
        self._deep_copied_space_converters = True
        self._obs_z_score_mean = None
        self._obs_z_score_std = None
        self._rew_z_score_mean = None
        self._rew_z_score_std = None
        self._resetted = False

    def _normalize_obs(self, observation):
        if self._obs_z_score_mean  is not None and self._obs_z_score_std is not None:
            if type(observation) in (tuple, list):
                j = 0
                new_observation = list(observation)
                for i in range(len(new_observation)):
                    if new_observation[i].dtype in (np.float16, np.float32, np.float64):
                        new_observation[i] = (new_observation[i] - self._obs_z_score_mean[j]) / self._obs_z_score_std[j]
                        j += 1
                new_observation = tuple(new_observation)
            elif type(observation) == dict:
                new_observation = dict()
                for key in observation.keys():
                    new_observation[key] = (observation[key] - self._obs_z_score_mean[key]) / self._obs_z_score_std[key]
            else:
                new_observation = (observation - self._obs_z_score_mean) / self._obs_z_score_std
            #print("DO NORMALIZE", self.eval_env)
            return new_observation
        else:
            return observation

    def reset(self) -> Tuple[ObsType, dict]:
        observation = self._wrapped_rec_env.reset()
        
        if not self._deep_copied_space_converters:
            self._space_converters = deepcopy(self._space_converters)
        self._space_converters.reset()
        previous_reward = {
            "metering_period_cost": 0.0,
            "peak_period_cost": 0.0,
            "controllable_assets_costs": 0.0
        }
        self._previous_reward = dict(previous_reward)
        previous_action = {
            action:0.0 for action in dict(self._wrapped_rec_env.action_space).keys()
        }
        previous_observation = dict(observation)
        self._previous_observation = dict(previous_observation)
        new_observation, new_action, new_reward = self._space_converters.convert(
            observation=previous_observation,
            action=previous_action,
            reward=previous_reward,
            original_action=previous_action,
            original_observation=previous_observation,
            original_reward=previous_reward,
            infos=dict(),
            metering_period_counter=0,
            peak_period_counter=0
        )
        self._previous_converted_observation = new_observation
        self._previous_converted_action = new_action
        self._previous_converted_reward = new_reward
        self._previous_infos = dict()
        new_observation = self._normalize_obs(new_observation)
        self._resetted=True
        return new_observation
    
    def space_convert_obs(self, obs, action=None, reward=None):
        new_obs = None
        if action is None and reward is None:
            previous_action = {
                action:0.0 for action in dict(self._wrapped_rec_env.action_space).keys()
            }
            previous_reward = {
                "metering_period_cost": 0.0,
                "peak_period_cost": 0.0,
                "controllable_assets_costs": 0.0
            }
        elif action is not None and reward is not None:
            previous_action = action
            previous_reward = reward
        else:
            raise NotImplementedError("These settings do not make sense")
        new_obs, _, _ = self._space_converters.convert(
            observation=obs,
            action=previous_action,
            reward=previous_reward,
            original_action=previous_action,
            original_observation=obs,
            original_reward=previous_reward,
            infos=dict(),
            metering_period_counter=0,
            peak_period_counter=0
        )
        return new_obs
    
    def space_convert_act(self, action, obs=None, reward=None):
        new_act = None
        if obs is None and reward is None:
            previous_obs = self._wrapped_rec_env._compute_current_observation()
            previous_reward = {
                "metering_period_cost": 0.0,
                "peak_period_cost": 0.0,
                "controllable_assets_costs": 0.0
            }
        elif obs is not None and reward is not None:
            previous_obs = obs
            previous_reward = reward
        else:
            raise NotImplementedError("These settings do not make sense")
        _, new_act, _ = self._space_converters.convert(
            action=action,
            observation=previous_obs,
            reward=previous_reward,
            original_action=action,
            original_observation=obs,
            original_reward=previous_reward,
            infos=dict(),
            metering_period_counter=0,
            peak_period_counter=0
        )
        return new_act
    
    def step(self, action: ActType):
        #if not self._eval_env:
            #print(id(self), self._wrapped_rec_env._wrapped_rec_env._t)
        if type(self.action_space) == Box:
            action = np.clip(action, np.round(self.action_space.low, 6), np.round(self.action_space.high, 6))
        elif type(self.action_space) != Discrete and type(self.action_space[0] == Box):
            action[0] = np.clip(action, np.round(self.action_space[0].low, 6), np.round(self.action_space[0].high, 6))
        _, original_action, _ = self._space_converters.convert(
            observation=self._previous_converted_observation,
            action=action,
            reward=self._previous_converted_reward,
            original_action=None,
            original_observation=self._previous_observation,
            original_reward=self._previous_reward,
            backward=True,
            infos=self._previous_infos,
            metering_period_counter=self._previous_observation["metering_period_counter"],
            peak_period_counter=self._previous_observation.get("peak_period_counter", 0)
        )
        #t = time()
        next_original_observation, cost, terminated, truncated, info = self._wrapped_rec_env.step(original_action)
        #print(terminated, truncated)
        #print("Internal step took", time() - t)
        if not terminated:
            original_reward = {
                "metering_period_cost": info["costs"]["metering_period_cost"],
                "peak_period_cost": info["costs"].get("peak_period_cost", 0.0),
                "controllable_assets_cost": info["costs"]["controllable_assets_cost"] * 0.0
            }
        else:
            original_reward = {
                "metering_period_cost": 100000,
                "peak_period_cost": 100000,
                "controllable_assets_cost": 100000
            }
        previous_observation = self._previous_observation
        self._previous_reward = original_reward
        self._previous_observation = next_original_observation
        self._previous_action = original_action
        #if self._eval_env and self._previous_observation["peak_period_counter"] == self._wrapped_rec_env.Delta_P-1:
            #print(original_reward, info)
            #print(self._previous_observation["metering_period_counter"], cost, self._previous_observation.get("peak_period_counter", 0), next_original_observation["peak_period_counter"] == self._wrapped_rec_env.Delta_P)
            #print(info)
        next_observation, _, new_cost = self._space_converters.convert(
            observation=next_original_observation,
            action=original_action,
            reward=original_reward,
            original_observation=next_original_observation,
            original_action=original_action, 
            original_reward=original_reward,
            eval_env=self._eval_env,
            infos=info,
            metering_period_counter=previous_observation["metering_period_counter"],
            peak_period_counter=previous_observation.get("peak_period_counter", 0),
            reward_z_score_infos=((self._rew_z_score_mean, self._rew_z_score_std) if (info["is_peak_period_cost_triggered"] or info["is_metering_period_cost_triggered"]) else None),
            current_observation=previous_observation
        )
        if new_cost is not None:
            if type(new_cost) == dict:
                new_cost = {
                    k:(v if v is not None else 0.0) for k,v in new_cost.items()
                }
                new_cost = sum(list(new_cost.values()))
            elif type(new_cost) in (list, tuple):
                new_cost = [
                    (v if v is not None else 0.0) for v in new_cost
                ]
                new_cost = sum(new_cost)
        self._previous_observation = self._wrapped_rec_env._compute_current_observation()
        self._previous_converted_observation = next_observation
        self._previous_converted_action = action
        self._previous_converted_reward = new_cost
        self._previous_infos = info
        next_observation = self._normalize_obs(next_observation)
        new_info = dict(info)
        if new_cost is None:
            reward = 0.0
            new_info["is_peak_period_cost_triggered"] = False
            new_info["is_metering_period_cost_triggered"] = False
        else:
            reward = -new_cost
            if (info["is_peak_period_cost_triggered"] or info["is_metering_period_cost_triggered"]) and self._rew_z_score_mean is not None and self._rew_z_score_std is not None:
                reward = (reward - self._rew_z_score_mean) / self._rew_z_score_std
        self._resetted=False
        return next_observation, reward, terminated or truncated, {**new_info, **{"terminated": terminated,  "truncated": truncated, "eval_env": self._eval_env}}
    
    @property
    def eval_env(self):
        return self._eval_env
    
    @property
    def type_solver(self):
        return self._wrapped_rec_env.type_solver

    @property
    def env_name(self):
        return self._wrapped_rec_env.env_name
    
    @property
    def T(self):
        return self._wrapped_rec_env.T
    
    @property
    def members_with_controllable_assets(self):
        return self._members_with_controllable_assets
    
    @property
    def space_converters(self):
        return self._space_converters
    
    @property
    def wrapped_rec_env(self):
        return self._wrapped_rec_env
    
    @property
    def obs_z_score_mean(self):
        return self._obs_z_score_mean
    
    @property
    def obs_z_score_std(self):
        return self._obs_z_score_std
    
    @obs_z_score_mean.setter
    def obs_z_score_mean(self, new_obs_z_score_mean):
        self._obs_z_score_mean = new_obs_z_score_mean

    @obs_z_score_std.setter
    def obs_z_score_std(self, new_obs_z_score_std):
        self._obs_z_score_std = new_obs_z_score_std

    @property
    def rew_z_score_mean(self):
        return self._rew_z_score_mean
    
    @property
    def rew_z_score_std(self):
        return self._rew_z_score_std
    
    @rew_z_score_mean.setter
    def rew_z_score_mean(self, new_rew_z_score_mean):
        self._rew_z_score_mean = new_rew_z_score_mean

    @rew_z_score_std.setter
    def rew_z_score_std(self, new_rew_z_score_std):
        self._rew_z_score_std = new_rew_z_score_std

    def render(self, *args, **kwargs):
        pass