MPC_POLICIES="--mpc-policy perfect_foresight_mpc --mpc-policy perfect_foresight_mpc_commodity --mpc-policy perfect_foresight_mpc_commodity_peak_force"
GAMMAS="--gamma 0.999983"
GAMMAS_POLICIES="--gamma-policy 0.0 --gamma-policy 1.0"
RESCALE_GAMMAS="--rescaled-gamma-flags rescaled-gamma --rescaled-gamma-flags no-rescaled-gamma"
SMALL_PEN_CTRL_ACTIONS="--small-pen-ctrl-actions 0.001 --small-pen-ctrl-actions 0.01"
python -m slurm_utils.rec_param_mpc_sweeper \
--wandb-project mpc_study_rec_6_from_rec_28_data --env rec_6 --Delta-M 96 --Delta-P 30 --Delta-P-prime 0 --n-cpus 3 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K $1 --max-K $2 --step-K $3 \
--max-n-jobs 999 --id-job rec_6_from_rec_28_data_mpc --job-dir-name rec_6_mpc_jobs --time-limits $4 --multiprice-flags no-multiprice $SMALL_PEN_CTRL_ACTIONS