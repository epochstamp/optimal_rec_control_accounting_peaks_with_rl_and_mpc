parallel --bar python -m experiment_scripts.mpc.mpc_experiment --env {3} --exogenous-data-provider {4} --Delta-M 4 --Delta-P 5 --Delta-P-prime 0 --mpc-policy {1} --Delta-P-prime 0 --n-cpus 1 --remove-historical-peak-costs --use-wandb --wandb-project mpc_study_rec_2_stochastic --K {2} --gamma 0.9995 --rescaled-gamma-mode rescale_terminal --random-seed {5} --n-samples 128 ::: perfect_foresight_mpc ::: {1..101} ::: rec_2_red_stochastic_50_3 ::: pseudo_forecast_rec_2_100_input pseudo_forecast_rec_2_085_input pseudo_forecast_rec_2_050_input pseudo_forecast_rec_2_001_input pseudo_forecast_rec_2_100_sample pseudo_forecast_rec_2_085_sample pseudo_forecast_rec_2_050_sample pseudo_forecast_rec_2_001_sample ::: 422951 126169 923561 927046 557584 481225 260469 854937 488258 911459 49912 5859 602666 640472 549738 753201