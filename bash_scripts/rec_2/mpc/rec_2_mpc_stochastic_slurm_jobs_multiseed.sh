ENVIRONMENTS="--env rec_2_red_stochastic_50_3"
FORECASTS=" --exogenous-data-provider pseudo_forecast_rec_2_100_sample --exogenous-data-provider pseudo_forecast_rec_2_085_sample --exogenous-data-provider pseudo_forecast_rec_2_050_sample"
MPC_POLICIES="--mpc-policy perfect_foresight_mpc --mpc-policy perfect_foresight_mpc_commodity_peak_force"
GAMMAS="--gamma 0.9995"
GAMMAS_POLICIES="--gamma-policy 0.9995"
DELTA_P="--Delta-P 5"
RESCALE_GAMMAS="--rescaled-gamma-mode rescale_terminal"
SMALL_PEN_CTRL_ACTIONS="--small-pen-ctrl-actions 0.0"
SOLUTION_CHAINED_OPTIMISATION_FLAGS="--solution-chained-optimisation-flags fresh-optimisation"
NB_RANDOM_SEED="2048"
if [ -z "$GLOBALSCRATCH" ]
then
      ROOT_DIR=$HOME
else
      ROOT_DIR=$GLOBALSCRATCH
fi
python -m slurm_utils.rec_param_mpc_sweeper --wandb-offline --output 'mpc_outputs/mpc_output.txt' --partitions batch \
--wandb-project mpc_study_rec_2_stochastic --n-samples 1 --nb-random-seed $NB_RANDOM_SEED $ENVIRONMENTS $FORECASTS --Delta-M 4 $DELTA_P --Delta-P-prime 0 --n-cpus 1 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K 1 --max-K 101 --step-K 1 \
--max-n-jobs 599 --n-tasks 1 --id-job rec_2_stochastic_mpc --mem-per-cpu 2000 --root-dir $ROOT_DIR --job-dir-name rec_2_stochastic_mpc --time-limits 0 0 1 0 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $SOLUTION_CHAINED_OPTIMISATION_FLAGS --solver cplex