MPC_POLICIES="--mpc-policy perfect_foresight_mpc --mpc-policy perfect_foresight_mpc_commodity_peak_force"
GAMMAS="--gamma 0.99993"
GAMMAS_POLICIES="--gamma-policy 0.99993"
DELTA_P="--Delta-P 5"
RESCALE_GAMMAS="--rescaled-gamma-mode rescale_terminal"
SMALL_PEN_CTRL_ACTIONS="--small-pen-ctrl-actions 0.0"
SOLUTION_CHAINED_OPTIMISATION_FLAGS="--solution-chained-optimisation-flags fresh-optimisation"
NB_RANDOM_SEED="1"
if [ -z "$GLOBALSCRATCH" ]
then
      ROOT_DIR=$HOME
else
      ROOT_DIR=$GLOBALSCRATCH
fi
python -m slurm_utils.rec_param_mpc_sweeper --wandb-offline \
--wandb-project mpc_study_rec_7 --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED --env short_horizon_rec_7_from_rec_28_summer_end --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 1 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 1 --max-K 99 --step-K 1 \
--max-n-jobs 999 --id-job 1_short_horizon_rec_7_from_rec_28_summer_end_mpc_small_K --mem-per-cpu 2500 --root-dir $ROOT_DIR --job-dir-name rec_7_mpc_jobs --time-limits 0 0 15 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --time-policy --solver cplex &&
python -m slurm_utils.rec_param_mpc_sweeper --wandb-offline \
--wandb-project mpc_study_rec_7 --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED --env short_horizon_rec_7_from_rec_28_summer_end --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 1 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 100 --max-K 250 --step-K 16 \
--max-n-jobs 999 --id-job 2_short_horizon_rec_7_from_rec_28_summer_end_mpc_medium_K --mem-per-cpu 2500 --root-dir $ROOT_DIR --job-dir-name rec_7_mpc_jobs --time-limits 0 0 30 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --time-policy --solver cplex &&
python -m slurm_utils.rec_param_mpc_sweeper --wandb-offline \
--wandb-project mpc_study_rec_7 --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED --env short_horizon_rec_7_from_rec_28_summer_end --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 1 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 266 --max-K 714 --step-K 16 \
--max-n-jobs 999 --id-job 3_short_horizon_rec_7_from_rec_28_summer_end_mpc_high_K --mem-per-cpu 2500 --root-dir $ROOT_DIR --job-dir-name rec_7_mpc_jobs --time-limits 0 1 15 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --time-policy --solver cplex &&
python -m slurm_utils.rec_param_mpc_sweeper --wandb-offline \
--wandb-project mpc_study_rec_7 --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED --env short_horizon_rec_7_from_rec_28_summer_end --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 1 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 715 --max-K 721 --step-K 1 \
--max-n-jobs 999 --id-job 4_short_horizon_rec_7_from_rec_28_summer_end_mpc_high_K --mem-per-cpu 2500 --root-dir $ROOT_DIR --job-dir-name rec_7_mpc_jobs --time-limits 0 1 30 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --time-policy --solver cplex