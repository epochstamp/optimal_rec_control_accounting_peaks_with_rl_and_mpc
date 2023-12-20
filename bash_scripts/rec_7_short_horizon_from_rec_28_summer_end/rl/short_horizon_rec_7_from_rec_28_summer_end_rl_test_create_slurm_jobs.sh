SPACE_CONVERTERS="--space-converter squeeze_and_zero_centered_scale_battery#binary_masks#flatten_and_boxify --space-converter squeeze_and_zero_centered_scale_battery#remove_prices#resize_and_pad_meters#net_meters#binary_masks#flatten_and_separate --space-converter squeeze_and_zero_centered_scale_battery#remove_prices#net_meters#binary_masks#flatten_and_separate"
MEAN_STD_FILTER_MODES="--mean-std-filter-mode obs_and_rew_optim"
DELTAS_P="--Delta-P 30"
#DELTAS_P="--Delta-P 30 --Delta-P 90"
N_EVAL="--ne-eval 1"
#RL="--rl-env rl --rl-env rl_commodity_peak_force"
#RL_METERING_DENSE="--rl-env rl_metering_dense --rl-env rl_commodity_peak_force_metering_dense"
#RL_DENSE="--rl-env rl_dense --rl-env rl_commodity_peak_force_dense"
RL="--rl-env rl"
RL_METERING_DENSE="--rl-env rl_metering_dense"
RL_DENSE="--rl-env rl_dense"
LAMBDA_GAES="--lambda-gae 0.9"
GCS="--gc 2.0 --gc 4.0"
LRS="--lr 3e-7 --lr 5e-7#1e-7#500"
NE="--ne 64"
BS="--bs 512"
N_SGDS="--n-sgds 5"
KL_COEFF="--kl-coeff 0.3"
KL_TARGET="--kl-target 0.1"
CLIP_PARAM="--clip-param 0.2"
VF_COEFF="--vf-coeff 1.0 --vf-coeff 1e-3"
GAMMAS_POLICY="--gamma-policy 0.99"
GAMMAS="--gamma 0.99993"
MODEL_CONFIGS="--model-config short_horizon_rec_7_from_rec_28_summer_end_custom --model-config short_horizon_rec_7_from_rec_28_summer_end_default"
#RL_ENV="--rl-env rl --rl-env rl_dense"
ENTROPY_COEFF="--entropy-coeff 0.0001"
ACTION_WEIGHTS_DIVIDER="--action-weights-divider 1.0"
ACTION_CUSTOM_DIST="--action-dist default"
NB_RANDOM_SEED="4"
N_ITERS="750"
if [ -z "$GLOBALSCRATCH" ]
then
      ROOT_DIR=$HOME
else
      ROOT_DIR=$GLOBALSCRATCH
fi
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline \
--wandb-project rl_study_rec_7 --env short_horizon_rec_7_from_rec_28_summer_end --Delta-M 4 $DELTAS_P --Delta-P-prime 0 --n-cpus 1 --n-cpus-extras 0 --mem-per-cpu 3000 $RL --rl-env-eval rl \
$NE $BS $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $KL_TARGET $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 499 --root-dir $ROOT_DIR  --id-job 1_short_horizon_rec_7_from_rec_28_summer_end_1_rl --job-dir-name short_horizon_rec_7_from_rec_28_summer_end_rl_jobs --time-limits 0 26 30 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter --tar-gz-results --sha-folders &&
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline \
--wandb-project rl_study_rec_7 --env short_horizon_rec_7_from_rec_28_summer_end --Delta-M 4 $DELTAS_P --Delta-P-prime 0 --n-cpus 1 --n-cpus-extras 0 --mem-per-cpu 3000 $RL_METERING_DENSE --rl-env-eval rl \
$NE $BS $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $KL_TARGET $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 499 --root-dir $ROOT_DIR  --id-job 2_short_horizon_rec_7_from_rec_28_summer_end_1_rl_metering_dense --job-dir-name short_horizon_rec_7_from_rec_28_summer_end_rl_jobs --time-limits 0 33 30 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter --tar-gz-results --sha-folders &&
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline \
--wandb-project rl_study_rec_7 --env short_horizon_rec_7_from_rec_28_summer_end --Delta-M 4 $DELTAS_P --Delta-P-prime 0 --n-cpus 1 --n-cpus-extras 0 --mem-per-cpu 3000 $RL_DENSE --rl-env-eval rl \
$NE $BS $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $KL_TARGET $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 499 --root-dir $ROOT_DIR  --id-job 3_short_horizon_rec_7_from_rec_28_summer_end_1_rl_dense --job-dir-name short_horizon_rec_7_from_rec_28_summer_end_rl_jobs --time-limits 0 39 30 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter --tar-gz-results --sha-folders