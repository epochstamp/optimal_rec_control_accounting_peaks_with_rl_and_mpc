python -m experiment_scripts.mpc.mpc_experiment --stdout --env rec_6_from_rec_28_data --Delta-M 96 --Delta-P 30 --mpc-policy perfect_foresight_mpc --Delta-P-prime 0 --n-cpus 3 --K $1 --remove-current-peak-costs --remove-historical-peak-costs  ${@:2}