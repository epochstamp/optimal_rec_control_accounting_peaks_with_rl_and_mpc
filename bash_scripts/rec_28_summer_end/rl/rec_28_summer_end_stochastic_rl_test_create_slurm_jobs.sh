ENVIRONMENTS="--env rec_28_summer_end_red_stochastic_25 --env rec_28_summer_end_red_stochastic_50 --env rec_28_summer_end_red_stochastic_75"
SPACE_CONVERTERS="--space-converter squeeze_and_scale_battery#net_meters#binary_masks#flatten_and_boxify"
MEAN_STD_FILTER_MODES="--mean-std-filter-mode obs_and_rew_multi_optim"
DELTAS_P="--Delta-P 120 --Delta-P 360"
N_EVAL="--ne-eval 10"
RL="--rl-env rl --rl-env rl_commodity_peak_force"
RL_METERING_DENSE="--rl-env rl_metering_dense --rl-env rl_commodity_peak_force_metering_dense"
RL_DENSE="--rl-env rl_dense --rl-env rl_commodity_peak_force_dense"
LAMBDA_GAES="--lambda-gae 1.0"
GCS="--gc 8.0"
LRS="--lr 0.000005 --lr 0.0000005"
NE="--ne 32"
BS="--bs 2048"
N_SGDS="--n-sgds 10"
KL_COEFF="--kl-coeff 0.2"
CLIP_PARAM="--clip-param 0.3"
VF_COEFF="--vf-coeff 1.0"
GAMMAS_POLICY="--gamma-policy 0.99#0.999992#175"
GAMMAS="--gamma 0.999992"
MODEL_CONFIGS="--model-config rec_28_default_separated_vf"
#RL_ENV="--rl-env rl --rl-env rl_dense"
ENTROPY_COEFF="--entropy-coeff 0.01"
ACTION_WEIGHTS_DIVIDER="--action-weights-divider 1.0"
ACTION_CUSTOM_DIST="--action-dist default"
NB_RANDOM_SEED="10"
N_ITERS="350"
if [ -z "$GLOBALSCRATCH" ]
then
      ROOT_DIR=$HOME
else
      ROOT_DIR=$GLOBALSCRATCH
fi
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline \
--wandb-project rl_study_rec_28_stochastic $ENVIRONMENTS --Delta-M 4 $DELTAS_P --Delta-P-prime 0 --n-cpus 1 --n-cpus-extras 0 --mem-per-cpu 2500 $RL --rl-env-eval rl \
$NE $BS $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 499 --root-dir $ROOT_DIR  --id-job 1_rec_28_stochastic_summer_end_1_rl --job-dir-name rec_28_stochastic_summer_end_rl_jobs --time-limits 0 48 00 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter &&
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline \
--wandb-project rl_study_rec_28_stochastic $ENVIRONMENTS --Delta-M 4 $DELTAS_P --Delta-P-prime 0 --n-cpus 1 --n-cpus-extras 0 --mem-per-cpu 2500 $RL_METERING_DENSE --rl-env-eval rl \
$NE $BS $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 499 --root-dir $ROOT_DIR  --id-job 2_rec_28_summer_end_1_rl_metering_dense --job-dir-name rec_28_stochastic_summer_end_rl_jobs --time-limits 0 48 00 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline \
--wandb-project rl_study_rec_28_stochastic $ENVIRONMENTS --Delta-M 4 $DELTAS_P --Delta-P-prime 0 --n-cpus 1 --n-cpus-extras 0 --mem-per-cpu 2500 $RL_DENSE --rl-env-eval rl \
$NE $BS $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1436 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 499 --root-dir $ROOT_DIR  --id-job 2_rec_28_summer_end_1_rl_dense --job-dir-name rec_28_stochastic_summer_end_rl_jobs --time-limits 0 48 00 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter