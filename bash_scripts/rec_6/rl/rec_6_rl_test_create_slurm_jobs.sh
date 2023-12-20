RL_ENVS="--rl-env rl_greedy --rl-env rl_dense --rl-env rl_semidense --rl-env rl_greedy_dense --rl-env rl"
SPACE_CONVERTERS="--space-converter squeeze_and_scale_battery --space-converter squeeze_and_scale_battery#sum_meters"
MEAN_STD_FILTER_MODES="--mean-std-filter-mode no_filter --mean-std-filter-mode only_obs --mean-std-filter-mode obs_and_rew"
LAMBDA_GAES="--lambda-gae 1.0 --lambda-gae 0.99"
GCS="--gc 1 --gc 2 --gc 4"
N_SGDS="--n-sgds 10 --n-sgds 15"
KL_COEFF="--kl-coeff 0.2 --kl-coeff 0.3"
MODEL_CONFIGS="--model-config rec_6_relu --model-config rec_6_default --model-config rec_6_transformer_default"
python -m slurm_utils.rec_param_rl_sweeper \
--wandb-project rl_study_rec_6 --env rec_6 --Delta-M 41 --Delta-P 7 --Delta-P-prime 0 --n-cpus 4 $RL_ENVS --rl-env-eval rl \
--ne 64 --bs 64 $GCS $N_SGDS $KL_COEFF $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS --n-iters 100 --random-seed 1245 --nb-random-seed 5 \
--max-n-jobs 999 --id-job rec_6_rl --job-dir-name rec_6_rl_jobs --time-limits 0 4 0 --multiprice-flags no-multiprice