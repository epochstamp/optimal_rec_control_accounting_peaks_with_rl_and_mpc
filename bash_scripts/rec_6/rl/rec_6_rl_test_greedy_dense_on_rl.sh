python -m experiment_scripts.rl.rl_experiment --stdout --env rec_6 --Delta-M 41 --Delta-P 7 --Delta-P-prime 0 --n-cpus 8 --rl-env rl_greedy_dense --rl-env-eval rl --remove-historical-peak-costs --ne 64 --bs 64 --gc 4 --n-sgds 5 --kl-coeff 0.3 --lambda-gae 0.9 --space-converter squeeze_and_scale_battery#sum_meters --mean-std-filter-mode obs_and_rew --time-iter --n-iters 1 --model-config rec_6_transformer_default