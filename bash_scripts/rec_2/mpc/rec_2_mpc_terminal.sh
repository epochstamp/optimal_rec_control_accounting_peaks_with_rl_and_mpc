python -m experiment_scripts.mpc.mpc_experiment --stdout --env rec_2 --Delta-M 4 --Delta-P 5 --mpc-policy perfect_foresight_mpc --Delta-P-prime 0 --n-cpus 3 --K $1 --remove-historical-peak-costs ${@:2}