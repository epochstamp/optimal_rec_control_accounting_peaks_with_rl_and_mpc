MPC_POLICIES="--mpc-policy perfect_foresight_mpc --mpc-policy perfect_foresight_mpc_commodity_peak_force"
GAMMAS="--gamma 0.999992"
GAMMAS_POLICIES="--gamma-policy 0.0 --gamma-policy 1.0"
RESCALE_GAMMAS="--rescaled-gamma-flags rescaled-gamma --rescaled-gamma-flags no-rescaled-gamma"
SMALL_PEN_CTRL_ACTIONS="--small-pen-ctrl-actions 1e-8"
DISABLE_NET_MUTEX_FLAG="--disable-net-cons-prod-mutex-flags no-disable-net-cons-prod-mutex"
python -m slurm_utils.rec_param_mpc_sweeper \
--wandb-project mpc_study_rec_28_summer_end --env rec_28_summer_end --Delta-M 48 --Delta-P 30 --Delta-P-prime 0 --n-cpus 3 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K $1 --max-K $2 --step-K $3 \
--max-n-jobs 999 --id-job rec_28_summer_end_mpc --job-dir-name rec_28_summer_end_mpc_jobs --time-limits $4 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS $DISABLE_NET_MUTEX_FLAG