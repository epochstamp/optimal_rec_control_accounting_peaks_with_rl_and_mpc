ENVIRONMENTS="--env rec_28_summer_end_red_stochastic_25 --env rec_28_summer_end_red_stochastic_50 --env rec_28_summer_end_red_stochastic_75"
FORECASTS="--exogenous-data-provider pseudo_forecast_rec_28_summer_end_100 --exogenous-data-provider pseudo_forecast_rec_28_summer_end_099 --exogenous-data-provider pseudo_forecast_rec_28_summer_end_085 --exogenous-data-provider pseudo_forecast_rec_28_summer_end_050 --exogenous-data-provider pseudo_forecast_rec_28_summer_end_025"
MPC_POLICIES="--mpc-policy perfect_foresight_mpc --mpc-policy perfect_foresight_mpc_commodity_peak_force"
GAMMAS="--gamma 0.999992"
GAMMAS_POLICIES="--gamma-policy 0.0"
DELTA_P="--Delta-P 120 --Delta-P 360"
RESCALE_GAMMAS="--rescaled-gamma-mode rescale_terminal"
SMALL_PEN_CTRL_ACTIONS="--small-pen-ctrl-actions 0.0"
SOLUTION_CHAINED_OPTIMISATION_FLAGS="--solution-chained-optimisation-flags fresh-optimisation"
NB_RANDOM_SEED="10"
if [ -z "$GLOBALSCRATCH" ]
then
      ROOT_DIR=$HOME
else
      ROOT_DIR=$GLOBALSCRATCH
fi
python -m slurm_utils.rec_param_mpc_sweeper \
--wandb-project mpc_study_rec_28_summer_end_stochastic --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED $ENVIRONMENTS $FORECASTS --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 1 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 1 --max-K 99 --step-K 1 \
--max-n-jobs 999 --id-job 1_rec_28_summer_end_stochastic_mpc_small_K --mem-per-cpu 4096 --root-dir $ROOT_DIR --job-dir-name rec_28_summer_stochastic_end_mpc_jobs --time-limits 0 1 00 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --time-policy --solver cplex --solver-verbose &&
python -m slurm_utils.rec_param_mpc_sweeper \
--wandb-project mpc_study_rec_28_summer_end_stochastic --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED $ENVIRONMENTS $FORECASTS --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 1 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 100 --max-K 500 --step-K 16 \
--max-n-jobs 999 --id-job 2_rec_28_summer_end_stochastic_mpc_medium_K --mem-per-cpu 4096 --root-dir $ROOT_DIR --job-dir-name rec_28_summer_stochastic_end_mpc_jobs --time-limits 0 2 15 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --time-policy --solver cplex --solver-verbose &&
python -m slurm_utils.rec_param_mpc_sweeper \
--wandb-project mpc_study_rec_28_summer_end_stochastic --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED $ENVIRONMENTS $FORECASTS --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 2 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 516 --max-K 1012 --step-K 16 \
--max-n-jobs 999 --id-job 3_rec_28_summer_end_stochastic_mpc_high_K --mem-per-cpu 4096 --root-dir $ROOT_DIR --job-dir-name rec_28_summer_stochastic_end_mpc_jobs --time-limits 0 4 15 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --time-policy --solver cplex --solver-verbose &&
python -m slurm_utils.rec_param_mpc_sweeper \
--wandb-project mpc_study_rec_28_summer_end_stochastic --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED $ENVIRONMENTS $FORECASTS --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 2 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 1012 --max-K 1512 --step-K 50 \
--max-n-jobs 999 --id-job 3_rec_28_summer_end_stochastic_mpc_high_K --mem-per-cpu 4096 --root-dir $ROOT_DIR --job-dir-name rec_28_summer_stochastic_end_mpc_jobs --time-limits 0 6 15 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --time-policy --solver cplex --solver-verbose &&
python -m slurm_utils.rec_param_mpc_sweeper \
--wandb-project mpc_study_rec_28_summer_end_stochastic --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED $ENVIRONMENTS $FORECASTS --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 4 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 1562 --max-K 2062 --step-K 100 \
--max-n-jobs 999 --id-job 4_rec_28_summer_end_stochastic_mpc_very_high_K --mem-per-cpu 4096 --root-dir $ROOT_DIR --job-dir-name rec_28_summer_stochastic_end_mpc_jobs --time-limits 0 8 15 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --time-policy --solver cplex --solver-verbose &&
python -m slurm_utils.rec_param_mpc_sweeper \
--wandb-project mpc_study_rec_28_summer_end_stochastic --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED $ENVIRONMENTS $FORECASTS --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 4 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 2161 --max-K 2881 --step-K 144 \
--max-n-jobs 999 --id-job 3_rec_28_summer_end_stochastic_mpc_very_very_high_K --mem-per-cpu 4096 --root-dir $ROOT_DIR --job-dir-name rec_28_summer_stochastic_end_mpc_jobs --time-limits 0 13 15 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --time-policy --solver cplex --solver-verbose