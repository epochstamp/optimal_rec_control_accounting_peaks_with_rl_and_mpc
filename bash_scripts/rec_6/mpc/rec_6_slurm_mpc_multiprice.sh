MPC_POLICIES="--mpc-policy perfect_foresight_mpc --mpc-policy perfect_foresight_mpc_commodity"
GAMMAS="--gamma 0.99"
GAMMAS_POLICIES="--gamma-policy 0.0 --gamma-policy 1.0"
RESCALE_GAMMAS="--rescaled-gamma-flags rescaled-gamma --rescaled-gamma-flags no-rescaled-gamma"
python -m slurm_utils.rec_param_mpc_sweeper \
--wandb-project mpc_study_rec_6 --env rec_6 --Delta-M 41 --Delta-P 7 --Delta-P-prime 0 --n-cpus 3 $MPC_POLICIES $GAMMAS $GAMMAS_POLICIES $RESCALE_GAMMAS --min-K $1 --max-K $2 --step-K $3 \
--max-n-jobs 999 --id-job rec_6_mpc --job-dir-name rec_6_mpc_jobs --time-limits 0 0 45 0 --multiprice-flags multiprice --small-pen-ctrl-actions 1e-5