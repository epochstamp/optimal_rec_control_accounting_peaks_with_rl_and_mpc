python -m experiment_scripts.mpc.mpc_experiment --stdout --env rec_6 --Delta-M 41 --Delta-P 7 --mpc-policy perfect_foresight_mpc_commodity --Delta-P-prime 0 --n-cpus 3 --K $1 --remove-historical-peak-costs ${@:2}