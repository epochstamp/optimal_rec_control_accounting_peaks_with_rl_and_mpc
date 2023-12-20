ENVIRONMENTS="--env short_horizon_rec_7_from_rec_28_summer_end_red_stochastic_50_3"
FORECASTS="--exogenous-data-provider pseudo_forecast_rec_7_100_sample --exogenous-data-provider pseudo_forecast_rec_7_0999_sample --exogenous-data-provider pseudo_forecast_rec_7_095_sample --exogenous-data-provider pseudo_forecast_rec_7_085_sample --exogenous-data-provider pseudo_forecast_rec_7_050_sample"
MPC_POLICIES="--mpc-policy perfect_foresight_mpc --mpc-policy perfect_foresight_mpc_commodity_peak_force --mpc-policy perfect_foresight_mpc_commodity_peak_force_optimised"
GAMMAS="--gamma 0.99993"
GAMMAS_POLICIES="--gamma-policy 0.99993"
DELTA_P="--Delta-P 30 --Delta-P 45 --Delta-P 60 --Delta-P 90"
RESCALE_GAMMAS="--rescaled-gamma-mode rescale_terminal"
SMALL_PEN_CTRL_ACTIONS="--small-pen-ctrl-actions 0.0"
SOLUTION_CHAINED_OPTIMISATION_FLAGS="--solution-chained-optimisation-flags fresh-optimisation"
N_SAMPLES="--n-samples 32"
NB_RANDOM_SEED="16"
if [ -z "$GLOBALSCRATCH" ]
then
      ROOT_DIR=$HOME
else
      ROOT_DIR=$GLOBALSCRATCH
fi
python -m slurm_utils.rec_param_mpc_sweeper --wandb-offline --time-policy \
--wandb-project mpc_study_rec_7_stochastic --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED $N_SAMPLES $ENVIRONMENTS $FORECASTS --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 4 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 1 --max-K 99 --step-K 1 \
--max-n-jobs 100 --id-job 1_rec_7_stochastic_mpc_small_K --mem-per-cpu 2500 --root-dir $ROOT_DIR --job-dir-name rec_7_stochastic_end_mpc_jobs --time-limits 0 1 15 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --solver cplex &&
python -m slurm_utils.rec_param_mpc_sweeper --wandb-offline --time-policy \
--wandb-project mpc_study_rec_7_stochastic --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED $N_SAMPLES $ENVIRONMENTS $FORECASTS --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 4 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 100 --max-K 250 --step-K 16 \
--max-n-jobs 100 --id-job 2_rec_7_stochastic_mpc_medium_K --mem-per-cpu 2500 --root-dir $ROOT_DIR --job-dir-name rec_7_stochastic_end_mpc_jobs --time-limits 0 1 30 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --solver cplex &&
python -m slurm_utils.rec_param_mpc_sweeper --wandb-offline --time-policy \
--wandb-project mpc_study_rec_7_stochastic --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED $N_SAMPLES $ENVIRONMENTS $FORECASTS --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 4 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 266 --max-K 714 --step-K 64 \
--max-n-jobs 100 --id-job 3_rec_7_stochastic_mpc_high_K --mem-per-cpu 2500 --root-dir $ROOT_DIR --job-dir-name rec_7_stochastic_end_mpc_jobs --time-limits 0 2 00 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --solver cplex &&
python -m slurm_utils.rec_param_mpc_sweeper --wandb-offline --time-policy \
--wandb-project mpc_study_rec_7_stochastic --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED $N_SAMPLES $ENVIRONMENTS $FORECASTS --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 4 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 714 --max-K 721 --step-K 1 \
--max-n-jobs 100 --id-job 4_rec_7_stochastic_mpc_very_high_K --mem-per-cpu 2500 --root-dir $ROOT_DIR --job-dir-name rec_7_stochastic_end_mpc_jobs --time-limits 0 3 00 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --solver cplex