ENVIRONMENTS="--env rec_2_red_stochastic_50_3"
SPACE_CONVERTERS="--space-converter squeeze_and_zero_centered_scale_battery#remove_prices#net_meters#flatten_and_boxify --space-converter squeeze_and_zero_centered_scale_battery#remove_prices#resize_and_pad_meters#net_meters#sum_min_max_meters#flatten_and_boxify_separate --space-converter squeeze_and_zero_centered_scale_battery#remove_prices#resize_and_pad_meters#net_meters#flatten_and_boxify_separate"
MEAN_STD_FILTER_MODES="--mean-std-filter-mode obs_and_rew_multi_optim"
N_EVAL="--ne-eval 128"
LAMBDA_GAES="--lambda-gae 0.9 --lambda-gae 0.99"
RL="--rl-env rl --rl-env rl_commodity_peak_force"
#RL_METERING_DENSE="--rl-env rl_metering_dense --rl-env rl_commodity_peak_force_metering_dense"
RL_DENSE="--rl-env rl_dense --rl-env rl_commodity_peak_force_dense"
GCS="--gc 2.0 --gc 1.0"
LRS="--lr 9e-5#1e-5#600"
N_SGDS="--n-sgds 10 --n-sgds 5"
KL_COEFF="--kl-coeff 0"
CLIP_PARAM="--clip-param 0.1"
VF_COEFF="--vf-coeff 1.0 --vf-coeff 0.01"
GAMMAS_POLICY="--gamma-policy 0.95"
GAMMAS="--gamma 0.9995"
MODEL_CONFIGS="--model-config rec_2_default_separated_vf --model-config rec_2_custom"
#RL_ENV="--rl-env rl --rl-env rl_dense"
ENTROPY_COEFF="--entropy-coeff 0.001"
ACTION_WEIGHTS_DIVIDER="--action-weights-divider 1.0"
ACTION_CUSTOM_DIST="--action-dist default"
NB_RANDOM_SEED="16"
N_ITERS="600"
if [ -z "$GLOBALSCRATCH" ]
then
      ROOT_DIR=$HOME
else
      ROOT_DIR=$GLOBALSCRATCH
fi
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline \
--wandb-project rl_study_rec_2_stochastic $ENVIRONMENTS --Delta-M 4 --Delta-P 5 --Delta-P-prime 0 --n-cpus 1 --n-cpus-extras 0 --mem-per-cpu 2500 $RL --rl-env-eval rl \
--ne 128 --bs 128 $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1346 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 499 --root-dir $ROOT_DIR  --id-job 1_rec_2_red_stochastic_1_rl --job-dir-name rec_2_red_stochastic_rl_jobs --time-limits 0 16 30 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter --tar-gz-results --sha-folders &&
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline \
--wandb-project rl_study_rec_2_stochastic $ENVIRONMENTS --Delta-M 4 --Delta-P 5 --Delta-P-prime 0 --n-cpus 1 --n-cpus-extras 0 --mem-per-cpu 2500 $RL_DENSE --rl-env-eval rl \
--ne 128 --bs 128 $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1346 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 499 --root-dir $ROOT_DIR  --id-job 2_1_rec_2_red_stochastic_1_rl_dense --job-dir-name rec_2_red_stochastic_rl_jobs --time-limits 0 17 30 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter --tar-gz-results --sha-folders