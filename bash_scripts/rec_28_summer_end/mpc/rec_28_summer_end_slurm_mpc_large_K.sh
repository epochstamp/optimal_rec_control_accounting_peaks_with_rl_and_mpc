MPC_POLICIES="--mpc-policy perfect_foresight_mpc --mpc-policy perfect_foresight_mpc_commodity_peak_force"
GAMMAS="--gamma 0.999992"
GAMMAS_POLICIES="--gamma-policy 0.0"
RESCALE_GAMMAS="--rescaled-gamma-mode no_rescale --rescaled-gamma-mode rescale_terminal --rescaled-gamma-mode rescale_delayed_terminal"
SMALL_PEN_CTRL_ACTIONS="--small-pen-ctrl-actions 0.0"
SOLUTION_CHAINED_OPTIMISATION_FLAGS="--solution-chained-optimisation-flags solution-chained-optimisation"
if [ -z "$GLOBALSCRATCH" ]
then
      ROOT_DIR=$HOME
else
      ROOT_DIR=$GLOBALSCRATCH
fi
python -m slurm_utils.rec_param_mpc_sweeper \
--wandb-project mpc_study_rec_28_summer_end --wandb-offline --env rec_28_summer_end --Delta-M 4 --Delta-P 120 --Delta-P-prime 0 --n-cpus 2 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 1028 --max-K 2528 --step-K 50 \
--max-n-jobs 499 --id-job 2_rec_28_summer_end_mpc_high_K --mem-per-cpu 8192 --partitions batch --root-dir $ROOT_DIR --job-dir-name rec_28_summer_end_mpc_jobs --time-limits 0 5 00 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --time-policy --solver cplex --solver-verbose