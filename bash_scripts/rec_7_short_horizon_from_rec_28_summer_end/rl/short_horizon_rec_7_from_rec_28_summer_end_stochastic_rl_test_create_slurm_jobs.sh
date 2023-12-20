ENVIRONMENTS="--env short_horizon_rec_7_from_rec_28_summer_end_red_stochastic_50_3"
SPACE_CONVERTERS="--space-converter squeeze_and_zero_centered_scale_battery#remove_prices#net_meters#flatten_and_boxify --space-converter squeeze_and_zero_centered_scale_battery#combine_meters_and_exogenous#remove_prices#net_meters#add_remaining_t#flatten_and_boxify --space-converter squeeze_and_zero_centered_scale_battery#remove_exogenous#net_meters#add_remaining_t#flatten_and_boxify"
MEAN_STD_FILTER_MODES="--mean-std-filter-mode obs_and_rew_multi_optim"
DELTAS_P="--Delta-P 45"
N_EVAL="--ne-eval 128"
RL="--rl-env rl"
RL_RETAIL="--rl-env rl_commodity_peak_force"
#RL_METERING_DENSE="--rl-env rl_metering_dense --rl-env rl_commodity_peak_force_metering_dense"
RL_DENSE="--rl-env rl_dense"
RL_RETAIL_DENSE="--rl-env rl_commodity_peak_force_dense"
#"--rl-env rl_dense --rl-env rl_commodity_peak_force_dense"
LAMBDA_GAES="--lambda-gae 0.9 --lambda-gae 0.95"
GCS="--gc 2.0"
LRS="--lr 5e-5 --lr 9e-5#3e-6#200"
NE="--ne 96"
BS="--bs 360"
N_SGDS="--n-sgds 5"
KL_COEFF="--kl-coeff 0.001 --kl-coeff 0"
CLIP_PARAM="--clip-param 0.1 --clip-param 0.3"
VF_COEFF="--vf-coeff 1e-3 --vf-coeff 1.0"
KL_TARGET="--kl-target 1e-3"
GAMMAS_POLICY="--gamma-policy 0.99"
GAMMAS="--gamma 0.99993"
MODEL_CONFIGS="--model-config short_horizon_rec_7_from_rec_28_summer_end_default_separated_vf"
#RL_ENV="--rl-env rl --rl-env rl_dense"
ENTROPY_COEFF="--entropy-coeff 0.001"
ACTION_WEIGHTS_DIVIDER="--action-weights-divider 100.0 --action-weights-divider 1.0"
ACTION_CUSTOM_DIST="--action-dist default --action-dist torch_diag_clipped_gaussian"
NB_RANDOM_SEED="2"
N_ITERS="300"
EVALUATION_INTERVAL="15"
if [ -z "$GLOBALSCRATCH" ]
then
      ROOT_DIR=$HOME
else
      ROOT_DIR=$GLOBALSCRATCH
fi
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline --evaluation-interval $EVALUATION_INTERVAL \
--wandb-project rl_study_rec_7_stochastic $ENVIRONMENTS --Delta-M 4 $DELTAS_P --Delta-P-prime 0 --n-cpus 4 --n-cpus-extras 0 --mem-per-cpu 10500 $RL --rl-env-eval rl \
 $NE $BS $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $KL_TARGET $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 400 --root-dir $ROOT_DIR  --id-job 1_rec_7_stochastic_summer_end_1_rl --job-dir-name rec_7_stochastic_summer_end_rl_jobs --time-limits 0 10 00 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter --tar-gz-results --sha-folders &&
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline --evaluation-interval $EVALUATION_INTERVAL \
--wandb-project rl_study_rec_7_stochastic $ENVIRONMENTS --Delta-M 4 $DELTAS_P --Delta-P-prime 0 --n-cpus 4 --n-cpus-extras 0 --mem-per-cpu 10500 $RL_DENSE --rl-env-eval rl \
 $NE $BS $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $KL_TARGET $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 400 --root-dir $ROOT_DIR  --id-job 2_rec_7_stochastic_summer_end_1_rl_dense --job-dir-name rec_7_stochastic_summer_end_rl_jobs --time-limits 0 10 00 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter --tar-gz-results --sha-folders &&
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline --evaluation-interval $EVALUATION_INTERVAL \
--wandb-project rl_study_rec_7_stochastic $ENVIRONMENTS --Delta-M 4 $DELTAS_P --Delta-P-prime 0 --n-cpus 4 --n-cpus-extras 0 --mem-per-cpu 10500 $RL_RETAIL --rl-env-eval rl \
 $NE $BS $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $KL_TARGET $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 400 --root-dir $ROOT_DIR  --id-job 3_rec_7_stochastic_summer_end_1_rl_retail --job-dir-name rec_7_stochastic_summer_end_rl_jobs --time-limits 0 10 00 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter --tar-gz-results --sha-folders &&
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline --evaluation-interval $EVALUATION_INTERVAL \
--wandb-project rl_study_rec_7_stochastic $ENVIRONMENTS --Delta-M 4 $DELTAS_P --Delta-P-prime 0 --n-cpus 4 --n-cpus-extras 0 --mem-per-cpu 10500 $RL_RETAIL_DENSE --rl-env-eval rl \
 $NE $BS $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $KL_TARGET $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 400 --root-dir $ROOT_DIR  --id-job 4_rec_7_summer_end_1_rl_retail_dense --job-dir-name rec_7_stochastic_summer_end_rl_jobs --time-limits 0 10 00 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter --tar-gz-results --sha-folders