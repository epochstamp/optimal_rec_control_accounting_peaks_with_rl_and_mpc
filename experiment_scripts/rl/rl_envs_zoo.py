from copy import deepcopy
from typing import Union
import numpy as np
from env.rec_env import RecEnv
from env.rec_env_global_bill_trigger import RecEnvGlobalBillTrigger
from env.rec_env_global_bill_wrapper import RecEnvGlobalBillWrapper
from experiment_scripts.rl.compute_optimal_z_score import compute_optimal_z_score
from experiment_scripts.rl.env_wrappers.rec_env_global_bill_members_agg import RecEnvGlobalBillMembersAgg
from experiment_scripts.rl.rec_env_rl_libs_gymnasium_gym_wrapper import RecEnvRlLibsGymnasiumGymWrapper
from experiment_scripts.rl.rec_env_rl_libs_wrapper import RecEnvRlLibsWrapper
from projectors.clip_projector import Clip_Projector
from utils.utils import create_clip_battery_action
from ray.tune.registry import register_env
from experiment_scripts.generic.trigger_zoo import metering_period_trigger_global_bill_functions, peak_period_trigger_global_bill_functions, period_trigger_global_bill_functions
from gymnasium.wrappers import EnvCompatibility
from gymnasium.spaces import Box, Discrete, Tuple as TupleSpace, Dict as DictSpace
from gym.spaces import Box as OldBox, Discrete as OldDiscrete, Tuple as OldTupleSpace, Dict as OldDictSpace
from gymnasium.envs.registration import register as register_gymnasium
from gym.envs.registration import register as register_gym

dont_include_that_space_converter_if_included = {
    "resize_and_pad_meters": {
        "observe_last_meter_only", "minimal_obs", "extreme_minimal_obs"
    },
    "flatten": {
        "flatten_and_boxify", "flatten_and_separate", "flatten_and_boxify_separate", "flatten_and_boxify_separate_dict", "flatten_and_boxify_separate_dict_repeat_exogenous"
    },
    "observe_last_meter_only":{
        "resize_and_pad_meters", "minimal_obs", "extreme_minimal_obs"
    },
    "observe_last_exogenous_variables_only":{
        "resize_and_pad_exogenous", "minimal_obs", "extreme_minimal_obs"
    }
}

battery_specs_envs = {
    "rec_2": {"PVB":
        [{
            "charge_as": "charge",
            "discharge_as": "discharge",
            "soc_as": "soc",
            "minsoc": 0,
            "maxsoc": 1,
            "discharge_efficiency": 1.0,
            "charge_efficiency": 1.0
        }]},
    "rec_6": {
        "B":
        [{
            "charge_as": "charge",
            "discharge_as": "discharge",
            "soc_as": "soc",
            "minsoc": 0,
            "maxsoc": 300,
            "discharge_efficiency": 0.88,
            "charge_efficiency": 0.88
        }]
    },

}

GLOBAL_OBS_OPTIM_SCORE = None
GLOBAL_REW_OPTIM_SCORE = None

def create_env_creator(rec_env_instance: Union[RecEnv, RecEnvGlobalBillWrapper], space_converter_ids, eval_env=False, gymnasium_wrap=False, infos_battery_specs=None, members_with_controllable_assets=[], gamma=1, obs_optim=False, rew_optim=False, num_optim_rollouts=1, obs_optim_score=None, rew_optim_score=None, verbose=False):
    
    def env_creator(*env_config_args, obs_optim_score=obs_optim_score, rew_optim_score=rew_optim_score, **env_config):
        
        
        battery_specs = battery_specs_envs[rec_env_instance.env_name] if infos_battery_specs is None else infos_battery_specs["battery_specs"]
        projector = Clip_Projector(
            rec_env_instance,
            create_clip_battery_action(
                battery_specs,
                rec_env_instance.Delta_C
            )
        )
        rec_env_instance.projector = projector
        #action_space_converter = action_converter_creator[env_config.get("action_space_converter_mode", "none")](rec_env)
        rec_env_wrapper = RecEnvRlLibsWrapper(
            rec_env_instance,
            space_converter_ids=space_converter_ids,
            eval_env=eval_env,
            members_with_controllable_assets=members_with_controllable_assets
        )
        if gymnasium_wrap:
            rec_env_wrapper = RecEnvRlLibsGymnasiumGymWrapper(rec_env_wrapper)
        if ((obs_optim and obs_optim_score is None) or (rew_optim and rew_optim_score is None)) and not eval_env:
            if verbose:
                print(gamma, obs_optim, rew_optim, num_optim_rollouts, obs_optim_score, rew_optim_score)
            obs_optim_score, rew_optim_score = compute_optimal_z_score(
                deepcopy(rec_env_wrapper),
                num_rollouts=num_optim_rollouts,
                gamma=gamma,
                include_obs=obs_optim,
                include_rew=rew_optim,
                verbose=verbose
            )
        if obs_optim:
            z_score_mean_obs, z_score_std_obs = obs_optim_score
            rec_env_wrapper.obs_z_score_mean = z_score_mean_obs
            rec_env_wrapper.obs_z_score_std = z_score_std_obs
            if type(rec_env_wrapper.observation_space) == TupleSpace:
                rec_env_wrapper.observation_space = TupleSpace(
                    tuple([
                        (Box(low=-np.inf, high=np.inf, shape=space.shape) if type(space) == Box else space) for space in rec_env_wrapper.observation_space.spaces
                    ])
                )
            else:
                rec_env_wrapper.observation_space = Box(low=-np.inf, high=np.inf, shape=rec_env_wrapper.observation_space.shape)
        if rew_optim and not rec_env_wrapper.eval_env:
            z_score_mean_rew, z_score_std_rew = rew_optim_score
            rec_env_wrapper.rew_z_score_mean = z_score_mean_rew
            rec_env_wrapper.rew_z_score_std = z_score_std_rew
        if env_config.get("return_optimal_scores", False):
            return rec_env_wrapper, obs_optim_score, rew_optim_score
        else:
            return rec_env_wrapper
    return env_creator

def create_peak_costs(rec_env:Union[RecEnv, RecEnvGlobalBillWrapper]):
    return {
        "current_offtake_peak_cost": rec_env.current_offtake_peak_cost,
        "current_injection_peak_cost": rec_env.current_injection_peak_cost,
        "historical_offtake_peak_cost": rec_env.historical_offtake_peak_cost,
        "historical_injection_peak_cost": rec_env.historical_injection_peak_cost
    }

def create_zero_peak_costs(rec_env:Union[RecEnv, RecEnvGlobalBillWrapper]):
    return {
        "current_offtake_peak_cost": 0,
        "current_injection_peak_cost": 0,
        "historical_offtake_peak_cost": 0,
        "historical_injection_peak_cost": 0
    }

def create_quasi_zero_peak_costs(rec_env:Union[RecEnv, RecEnvGlobalBillWrapper]):
    return {
        "current_offtake_peak_cost": 1e-12,
        "current_injection_peak_cost": 1e-12,
        "historical_offtake_peak_cost": 1e-12,
        "historical_injection_peak_cost": 1e-12
    }

rl_envs = dict()

rl_envs["rl"] = {
    "get_peak_costs": create_peak_costs
}

rl_envs["rl_dense"] = {
    **rl_envs["rl"],
    **{
        "metering_period_trigger": metering_period_trigger_global_bill_functions["everytime_trigger"],
        "peak_period_trigger": peak_period_trigger_global_bill_functions["everytime_trigger"],
    }
}

rl_envs["rl_metering_dense"] = {
    **rl_envs["rl"],
    **{
        "metering_period_trigger": metering_period_trigger_global_bill_functions["metering_period_trigger"],
        "peak_period_trigger": peak_period_trigger_global_bill_functions["metering_period_trigger"]
    }
}

rl_envs["rl_semidense"] = {
    **rl_envs["rl"],
    **{
        "metering_period_trigger": metering_period_trigger_global_bill_functions["metering_period_trigger"],
        "peak_period_trigger": peak_period_trigger_global_bill_functions["peak_period_trigger"],
    }
}

rl_envs["rl_dense_2nd"] = {
    **rl_envs["rl"],
    **{
        "metering_period_trigger": period_trigger_global_bill_functions["peak_period_2nd_trigger"],
        "peak_period_trigger": period_trigger_global_bill_functions["peak_period_2nd_trigger"],
    }
}

rl_envs["rl_dense_4th"] = {
    **rl_envs["rl"],
    **{
        "metering_period_trigger": period_trigger_global_bill_functions["peak_period_4th_trigger"],
        "peak_period_trigger": period_trigger_global_bill_functions["peak_period_4th_trigger"],
    }
}

rl_envs["rl_dense_8th"] = {
    **rl_envs["rl"],
    **{
        "metering_period_trigger": period_trigger_global_bill_functions["peak_period_8th_trigger"],
        "peak_period_trigger": period_trigger_global_bill_functions["peak_period_8th_trigger"],
    }
}

rl_envs["rl_dense_16th"] = {
    **rl_envs["rl"],
    **{
        "metering_period_trigger": period_trigger_global_bill_functions["peak_period_16th_trigger"],
        "peak_period_trigger": period_trigger_global_bill_functions["peak_period_16th_trigger"],
    }
}

rl_envs["rl_dense_32th"] = {
    **rl_envs["rl"],
    **{
        "metering_period_trigger": period_trigger_global_bill_functions["peak_period_32th_trigger"],
        "peak_period_trigger": period_trigger_global_bill_functions["peak_period_32th_trigger"],
    }
}

rl_envs["rl_commodity"] = {
    "get_peak_costs": create_zero_peak_costs
}

rl_envs["rl_commodity_dense"] = {
    **rl_envs["rl_commodity"],
    **{
        "metering_period_trigger": metering_period_trigger_global_bill_functions["everytime_trigger"],
        "peak_period_trigger": peak_period_trigger_global_bill_functions["everytime_trigger"],
    }
}

rl_envs["rl_commodity_peak_force"] = {
    "get_peak_costs": create_quasi_zero_peak_costs
}

rl_envs["rl_commodity_peak_force_semidense"] = {
    **rl_envs["rl_commodity_peak_force"],
    **{
        "metering_period_trigger": metering_period_trigger_global_bill_functions["metering_period_trigger"],
        "peak_period_trigger": peak_period_trigger_global_bill_functions["peak_period_trigger"],
    }
}

rl_envs["rl_commodity_peak_force_metering_dense"] = {
    **rl_envs["rl_commodity_peak_force"],
    **{
        "metering_period_trigger": metering_period_trigger_global_bill_functions["metering_period_trigger"],
        "peak_period_trigger": peak_period_trigger_global_bill_functions["metering_period_trigger"],
    }
}

rl_envs["rl_commodity_peak_force_dense"] = {
    **rl_envs["rl_commodity_peak_force"],
    **{
        "metering_period_trigger": metering_period_trigger_global_bill_functions["everytime_trigger"],
        "peak_period_trigger": peak_period_trigger_global_bill_functions["everytime_trigger"],
    }
}

rl_envs["rl_agg"] = {
    **rl_envs["rl"],
    **{
        "agg_meters": True
    }
}

rl_envs["rl_agg_dense"] = {
    **rl_envs["rl_agg"],
    **{
        "metering_period_trigger": metering_period_trigger_global_bill_functions["everytime_trigger"],
        "peak_period_trigger": peak_period_trigger_global_bill_functions["everytime_trigger"]
    }
}

rl_envs["rl_agg_commodity_peak_force"] = {
    **rl_envs["rl_commodity_peak_force"],
    **{
        "agg_meters": True
    }
}

rl_envs["rl_agg_commodity_peak_force_dense"] = {
    **rl_envs["rl_agg_commodity_peak_force"],
     **{
        "metering_period_trigger": metering_period_trigger_global_bill_functions["everytime_trigger"],
        "peak_period_trigger": peak_period_trigger_global_bill_functions["everytime_trigger"]
    }
}





def get_rl_env_kwargs(id_rl_env: str, rec_env:RecEnv):
    metering_period_trigger = None
    peak_period_trigger = None
    rl_envs_kwargs = dict(rl_envs[id_rl_env])
    if "metering_period_trigger" in rl_envs_kwargs:
        metering_period_trigger = rl_envs_kwargs["metering_period_trigger"](rec_env.Delta_M, rec_env.Delta_P)
        peak_period_trigger = rl_envs_kwargs["peak_period_trigger"](rec_env.Delta_M, rec_env.Delta_P)
        rl_envs_kwargs.pop("metering_period_trigger")
        rl_envs_kwargs.pop("peak_period_trigger")
    if "get_peak_costs" in rl_envs_kwargs:
        rl_envs_kwargs = {
            **rl_envs_kwargs,
            **rl_envs_kwargs["get_peak_costs"](rec_env)
        }
        rl_envs_kwargs.pop("get_peak_costs")
    return rl_envs_kwargs, {"metering_period_trigger": metering_period_trigger, "peak_period_trigger": peak_period_trigger}

def create_rl_env_creators(id_rl_env: str, id_rl_env_eval: str, rec_env_train: RecEnv, rec_env_eval: RecEnv, space_converter: str, gymnasium_wrap=False, time_iter=False, infos_rec_env_train=None, infos_rec_env_eval=None, rec_env_valid: RecEnv = None, infos_rec_env_valid=None, members_with_controllable_assets=[], gym_register=False, return_rec_env_train_creator=False, gamma=1, obs_optim=False, rew_optim=False, num_optim_rollouts=1, verbose=False):
    
    rl_train_envs_kwargs, rl_train_kwargs = get_rl_env_kwargs(id_rl_env, rec_env_train)
    rec_env_rl_train = rec_env_train.clone(
        **rl_train_envs_kwargs
    )
    rl_eval_envs_kwargs, rl_eval_kwargs = get_rl_env_kwargs(id_rl_env_eval, rec_env_eval)
    rec_env_rl_eval = rec_env_eval.clone(
        **rl_eval_envs_kwargs
    )
    metering_period_trigger = rl_train_kwargs["metering_period_trigger"]
    peak_period_trigger = rl_train_kwargs["peak_period_trigger"]
    metering_period_trigger_eval = rl_eval_kwargs["metering_period_trigger"]
    peak_period_trigger_eval = rl_eval_kwargs["peak_period_trigger"]

    if metering_period_trigger_eval is None and metering_period_trigger is not None:
        metering_period_trigger_eval = metering_period_trigger
        peak_period_trigger_eval = peak_period_trigger
    if metering_period_trigger is not None:
        
        rec_env_rl_train = RecEnvGlobalBillTrigger(
            rec_env_rl_train, metering_period_cost_trigger=metering_period_trigger, peak_period_cost_trigger=peak_period_trigger
        )
        rec_env_rl_train.global_bill_adaptative_optimiser.time_optim=time_iter
        rec_env_rl_train.global_bill_adaptative_optimiser.n_cpus = 1
    if metering_period_trigger_eval is not None:
        
        rec_env_rl_eval = RecEnvGlobalBillTrigger(
            rec_env_rl_eval, metering_period_cost_trigger=metering_period_trigger_eval, peak_period_cost_trigger=peak_period_trigger_eval
        )
        rec_env_rl_eval.global_bill_adaptative_optimiser.time_optim=time_iter
        rec_env_rl_eval.global_bill_adaptative_optimiser.n_cpus = 1

    if "agg_meters" in rl_train_envs_kwargs:
        rec_env_rl_train = RecEnvGlobalBillMembersAgg(
            rec_env=rec_env_rl_train,
            return_true_global_bill=False
        )
        rec_env_rl_eval = RecEnvGlobalBillMembersAgg(
            rec_env=rec_env_rl_eval,
            return_true_global_bill=True
        )
        

    train_rl_env_id = rec_env_train.env_name + "_train"
    eval_rl_env_id = rec_env_eval.env_name + "_eval"
    space_converter_lst = [space_converter for space_converter in space_converter.split("#") if space_converter != "no_converter"]
    prefix_space_converter = []
    space_converters_to_check_prefix = ["force_add_missing_obs", "observe_last_meter_only", "observe_last_exogenous_variables_only"]
    if "dense" in id_rl_env and "dense" not in id_rl_env_eval:
        #TODO : TEMPORARILY DISABLED. STILL GIVES THE GLOBAL BILL TOUGH
        pass#space_converters_to_check_prefix += ["keep_only_original_costs_eval"]
    for space_converter_to_check in space_converters_to_check_prefix:
        if (space_converter_to_check not in space_converter_lst and 
            dont_include_that_space_converter_if_included.get(space_converter_to_check, set()).intersection(set(space_converter_lst)) == set()
        ):
            prefix_space_converter = prefix_space_converter + [space_converter_to_check]
    suffix_space_converter = []
    space_converters_to_check_suffix = ["flatten"]
    for space_converter_to_check in space_converters_to_check_suffix:
        if (space_converter_to_check not in space_converter_lst and 
            dont_include_that_space_converter_if_included.get(space_converter_to_check, set()).intersection(set(space_converter_lst)) == set()
        ):
            suffix_space_converter += [space_converter_to_check]
    space_converter_lst = prefix_space_converter + space_converter_lst + suffix_space_converter
    space_converter_lst_eval = list(space_converter_lst)
    if "commodity" in id_rl_env and "commodity" not in id_rl_env_eval and "peak_force" not in id_rl_env:
        space_converter_lst[space_converter_lst.index("force_add_missing_obs")] = "force_add_missing_obs_with_zero_peak"
        space_converter_lst_eval[space_converter_lst_eval.index("force_add_missing_obs")] = "force_add_missing_obs_with_zero_peak"
    rec_env_rl_valid_creator = None
    rec_env_rl_train_creator = create_env_creator(rec_env_rl_train, list(space_converter_lst), eval_env=False, gymnasium_wrap=gymnasium_wrap, infos_battery_specs=infos_rec_env_train, members_with_controllable_assets=members_with_controllable_assets, gamma=gamma, obs_optim=obs_optim, rew_optim=rew_optim, num_optim_rollouts=num_optim_rollouts, verbose=verbose, obs_optim_score=None, rew_optim_score=None)
    _, obs_optim_score, rew_optim_score = rec_env_rl_train_creator(return_optimal_scores=True)
    rec_env_rl_train_creator = create_env_creator(rec_env_rl_train, list(space_converter_lst), eval_env=False, gymnasium_wrap=gymnasium_wrap, infos_battery_specs=infos_rec_env_train, members_with_controllable_assets=members_with_controllable_assets, gamma=gamma, obs_optim=obs_optim, rew_optim=rew_optim, num_optim_rollouts=num_optim_rollouts, verbose=verbose, obs_optim_score=obs_optim_score, rew_optim_score=rew_optim_score)
    if rec_env_valid is not None:
        if type(rec_env_rl_eval) == RecEnvGlobalBillTrigger:
            rec_env_valid = RecEnvGlobalBillTrigger(
                rec_env_valid.clone(), metering_period_cost_trigger=metering_period_trigger_eval, peak_period_cost_trigger=peak_period_trigger_eval
            )
            rec_env_valid.global_bill_adaptative_optimiser.time_optim=time_iter
            rec_env_valid.global_bill_adaptative_optimiser.n_cpus = 1
        rec_env_rl_valid_creator = create_env_creator(rec_env_valid, list(space_converter_lst_eval), eval_env=True, gymnasium_wrap=gymnasium_wrap, infos_battery_specs=infos_rec_env_eval, members_with_controllable_assets=members_with_controllable_assets, gamma=gamma, obs_optim=obs_optim, rew_optim=rew_optim, num_optim_rollouts=num_optim_rollouts, verbose=verbose, obs_optim_score=obs_optim_score, rew_optim_score=rew_optim_score)
    
    register_env(train_rl_env_id, rec_env_rl_train_creator)
    rec_env_rl_eval_creator = create_env_creator(rec_env_rl_eval, list(space_converter_lst_eval), eval_env=True, gymnasium_wrap=gymnasium_wrap, infos_battery_specs=infos_rec_env_eval, members_with_controllable_assets=members_with_controllable_assets, gamma=gamma, obs_optim=obs_optim, rew_optim=rew_optim, num_optim_rollouts=num_optim_rollouts, verbose=verbose, obs_optim_score=obs_optim_score, rew_optim_score=rew_optim_score)
    register_env(eval_rl_env_id, rec_env_rl_eval_creator)
    if gym_register:
        if gymnasium_wrap:
            register_gymnasium(train_rl_env_id, rec_env_rl_train_creator)
            register_gymnasium(eval_rl_env_id, rec_env_rl_eval_creator)
        else:
            register_gym(train_rl_env_id, rec_env_rl_train_creator)
            register_gym(eval_rl_env_id, rec_env_rl_eval_creator)
    
    if return_rec_env_train_creator:
        return train_rl_env_id, eval_rl_env_id, rec_env_rl_eval_creator, rec_env_rl_valid_creator, rec_env_rl_train_creator
    else:
        return train_rl_env_id, eval_rl_env_id, rec_env_rl_eval_creator, rec_env_rl_valid_creator
    