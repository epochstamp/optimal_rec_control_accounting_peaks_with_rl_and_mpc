SPACE_CONVERTERS="--space-converter squeeze_and_scale_battery#net_meters#binary_masks#flatten_and_boxify"
MEAN_STD_FILTER_MODES="--mean-std-filter-mode obs_and_rew_optim --mean-std-filter-mode obs_optim --mean-std-filter-mode rew_optim --mean-std-filter-mode no_filter"
MEAN_STD_FILTER_MODES_2="--mean-std-filter-mode obs_and_rew_multi_optim --mean-std-filter-mode obs_multi_optim --mean-std-filter-mode rew_multi_optim --mean-std-filter-mode no_filter"
N_EVAL="--ne-eval 1"
N_EVAL_2="--ne-eval 10"
LAMBDA_GAES="--lambda-gae 0.99"
RL="--rl-env rl --rl-env rl_commodity_peak_force"
RL_METERING_DENSE="--rl-env rl_metering_dense --rl-env rl_commodity_peak_force_metering_dense"
RL_DENSE="--rl-env rl_dense --rl-env rl_commodity_peak_force_dense"
GCS="--gc 8.0"
LRS="--lr 0.00005"
N_SGDS="--n-sgds 10"
KL_COEFF="--kl-coeff 0.2"
CLIP_PARAM="--clip-param 0.3"
VF_COEFF="--vf-coeff 1.0"
GAMMAS_POLICY="--gamma-policy 0.99#0.9995#125"
GAMMAS="--gamma 0.9995"
MODEL_CONFIGS="--model-config rec_2_default_separated_vf"
#RL_ENV="--rl-env rl --rl-env rl_dense"
ENTROPY_COEFF="--entropy-coeff 0.01"
ACTION_WEIGHTS_DIVIDER="--action-weights-divider 1.0"
ACTION_CUSTOM_DIST="--action-dist default"
NB_RANDOM_SEED="16"
N_ITERS="150"
if [ -z "$GLOBALSCRATCH" ]
then
      ROOT_DIR=$HOME
else
      ROOT_DIR=$GLOBALSCRATCH
fi
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline \
--wandb-project rl_study_rec_2 --env rec_2 --Delta-M 4 --Delta-P 5 --Delta-P-prime 0 --n-cpus 1 --n-cpus-extras 0 --mem-per-cpu 2500 $RL --rl-env-eval rl \
--ne 64 --bs 64 $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1346 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 499 --root-dir $ROOT_DIR  --id-job 1_rec_2_1_rl --job-dir-name rec_2_rl_jobs --time-limits 0 5 30 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter &&
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline \
--wandb-project rl_study_rec_2 --env rec_2 --Delta-M 4 --Delta-P 5 --Delta-P-prime 0 --n-cpus 1 --n-cpus-extras 0 --mem-per-cpu 2500 $RL_METERING_DENSE --rl-env-eval rl \
--ne 64 --bs 64 $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1346 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 499 --root-dir $ROOT_DIR  --id-job 1_rec_2_1_rl_metering_dense --job-dir-name rec_2_rl_jobs --time-limits 0 7 30 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter &&
python -m slurm_utils.rec_param_rl_sweeper --partitions batch --wandb-offline \
--wandb-project rl_study_rec_2 --env rec_2 --Delta-M 4 --Delta-P 5 --Delta-P-prime 0 --n-cpus 1 --n-cpus-extras 0 --mem-per-cpu 2500 $RL_DENSE --rl-env-eval rl \
--ne 64 --bs 64 $GCS $N_SGDS $N_EVAL $VF_COEFF $KL_COEFF $LAMBDA_GAES $SPACE_CONVERTERS $MEAN_STD_FILTER_MODES $MODEL_CONFIGS $LRS $GAMMAS_POLICY $GAMMAS $CLIP_PARAM $ENTROPY_COEFF $ACTION_WEIGHTS_DIVIDER $ACTION_CUSTOM_DIST --n-iters $N_ITERS --random-seed 1346 --nb-random-seed $NB_RANDOM_SEED \
--max-n-jobs 499 --root-dir $ROOT_DIR  --id-job 2_rec_2_1_rl_dense --job-dir-name rec_2_rl_jobs --time-limits 0 9 00 --multiprice-flags no-multiprice --gymnasium-wrap --time-iter