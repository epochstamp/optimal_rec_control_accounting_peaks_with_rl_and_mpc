python -m experiment_scripts.mpc.mpc_experiment --stdout --env rec_28_summer_end --Delta-M 48 --Delta-P 30 --mpc-policy perfect_foresight_mpc --Delta-P-prime 0 --n-cpus 3 --K $1 --remove-current-peak-costs --remove-historical-peak-costs  ${@:2}