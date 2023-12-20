

from base.space_converter import SpaceConverter
from typing import List, Dict
from env.rec_env import RecEnv
from experiment_scripts.rl.space_converters.numpify_values import NumpifyValues
from .space_converters import *
from .space_converters.sequential_space_converter import SequentialSpaceConverter


space_converter_sequences: Dict[str, List[SpaceConverter]] = {
    "no_converter": [],
    "keep_only_original_costs_eval": [KeepOnlyOriginalCostsEval],
    "observe_last_exogenous_variables_only": [ObserveLastExogenousVariablesOnly],
    "remove_global_bill_costs_eval_only": [RemoveMeteringPeriodCostsOnlyEval, RemovePeakPeriodCostsOnlyEval],
    "flatten": [FlattenAll, NumpifyValues],
    "flatten_and_boxify": [FlattenAllAndBoxify, NumpifyValues],
    "force_add_missing_obs": [ForceAddPreviousPeriodCosts, ForceAddPeakCounter],
    "sum_meters": [SumMeters],
    "net_meters": [NetMeters],
    "force_add_missing_obs_with_zero_peak": [ForceAddPreviousPeriodWithZeroPeakCosts, ForceAddZeroPeakCounter],
    "resize_and_pad_meters": [ResizeAndPadMetersObservations],
    "resize_and_pad_exogenous": [ResizeAndPadExogenousObservations],
    "observe_last_meter_only": [ObserveLastMeterOnly],
    "add_current_peaks": [AddCurrentPeaksObservations],
    "add_max_peaks": [AddMaxCurrentPeaksObservations],
    "squeeze_battery": [BatteryActionSqueeze],
    "scale_battery": [ActionScaler],
    "squeeze_and_scale_battery": [BatteryActionSqueeze, ActionScaler],
    "squeeze_and_beta_scale_battery": [BatteryActionSqueeze, BetaActionScaler],
    "squeeze_and_zero_centered_scale_battery": [BatteryActionSqueeze, ZeroCenteredActionScaler],
    "squeeze_and_zero_centered_scale_battery_10": [BatteryActionSqueeze, ZeroCenteredActionScaler10],
    "squeeze_and_zero_centered_scale_battery_100": [BatteryActionSqueeze, ZeroCenteredActionScaler100],
    "minimal_obs": [RemoveDiscreteObservations, RemoveMeterObservations],
    "extreme_minimal_obs": [RemoveDiscreteObservations, RemoveMeterObservations, RemoveControllableAssetsObservations, AddControllableAssetsActionsToObservations],
    "binary_masks": [AddDeltasBinaryMask],
    "remove_prices": [RemovePriceObservations],
    "remove_exogenous": [RemoveExogenousObservations],
    "remove_prices_meters": [RemovePriceObservations, RemoveMeterObservations],
    "net_prod_cons": [NetProdCons],
    "discretize_action": [Discretize_Action],
    "resize_and_pad_exogenous_members_variables_half_peak": [ResizeAndPadExogenousExogenousMembersVariablesHalfPeakPeriod],
    "flatten_and_separate": [FlattenAllAndSeparateStateExogenous],
    "flatten_and_boxify_separate": [FlattenAllAndBoxifySeparateStateExogenous],
    "curr_min_max_meters": [CurrMinMaxMeters],
    "sum_min_max_meters": [SumMinMaxMeters],
    "no_counters": [RemoveDiscreteObservations],
    "sep_sum_min_max_meters": [SepSumMinMaxMeters],
    "sep_sum_min_max_curr_meters": [SepSumMinMaxCurrMeters],
    "observe_peaks": [AddCurrentPeaksObservations],
    "cancel_intermediate_costs": [CancelIntermediateCosts],
    "surrogate_zero_energy_cost": [SurrogateZeroEnergyCostObs, SurrogateZeroEnergyCostRew],
    "surrogate_zero_energy_cost_with_peaks": [SurrogateZeroEnergyCostObsWithPeaks, SurrogateZeroEnergyCostRewWithPeaks],
    "sum_net_meters": [SumNetMeters],
    "sum_current_peaks": [SumCurrentPeaks],
    "add_remaining_t": [AddRemainingT],
    "add_remaining_t_ratio": [AddRemainingTRatio],
    "add_remaining_pp": [AddRemainingPeakPeriods],
    "add_remaining_pp_ratio": [AddRemainingPeakPeriodsRatio],
    "flatten_and_boxify_separate_dict": [FlattenAllAndSeparateBoxifyStateExogenousDict],
    "flatten_and_boxify_separate_dict_repeat_exogenous": [FlattenAllAndSeparateBoxifyStateExogenousDictRepeatExogenous],
    "combine_meters_and_exogenous": [CombineMetersAndExogenous]
}



def create_space_converter_sequence(
    rec_env: RecEnv, original_observation_space, original_action_space, original_reward_space, space_converter_sequence_ids: List[str] = []
) -> SequentialSpaceConverter:
    space_converters = list()
    current_observation_space = original_observation_space
    current_action_space = original_action_space
    current_reward_space = original_reward_space
    space_converter_sequence = []
    for space_converter_id in space_converter_sequence_ids:
        space_converter_sequence += space_converter_sequences[space_converter_id]
    for space_converter_cls in space_converter_sequence:
        space_converter_kwargs = space_converter_cls.get_kwargs_from_env_and_previous_converters(
            rec_env,
            space_converters
        )
        space_converter: SpaceConverter = space_converter_cls(
            current_action_space,
            current_observation_space,
            current_reward_space,
            original_action_space,
            original_observation_space,
            original_reward_space,
            **space_converter_kwargs
        )
        space_converters.append(
            space_converter
        )
        current_observation_space = space_converter.convert_observation_space()
        current_action_space = space_converter.convert_action_space()
        current_reward_space = space_converter.convert_reward_space()
    return SequentialSpaceConverter(
            original_action_space,
            original_observation_space,
            original_reward_space,
            original_action_space,
            original_observation_space,
            original_reward_space,
            space_converters=space_converters
        )