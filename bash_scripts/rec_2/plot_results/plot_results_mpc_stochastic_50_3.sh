python -m experiment_scripts.generic.plot_results --folder-results $HOME/OneDrive/rec_data_results/rec_2_red_stochastic_50_3_final/MPC_2/ --where-equals env rec_2_red_stochastic_50_3 --group-by mpc_policy --group-by exogenous_data_provider --min-number-of-points 100   --plot-title "" --where-equals exogenous_data_provider pseudo_forecast_rec_2_100_sample --where-equals exogenous_data_provider pseudo_forecast_rec_2_085_sample --where-equals exogenous_data_provider pseudo_forecast_rec_2_050_sample --legend-x-shift 0.92 --graph-width 750 --graph-height 550 --where-equals exogenous_data_provider pseudo_forecast_rec_2_100_sample --where-equals exogenous_data_provider pseudo_forecast_rec_2_085_sample --where-equals exogenous_data_provider pseudo_forecast_rec_2_050_sample --min-y 6.6 --max-y 7.6 --max-x-axis 100 --min-x-axis 1 --margin-right 109 --flat-values "OPT policy" 6.645 1E90FF --flat-values "RL policy" 6.83 E28D00 --flat-values "RL dense" 6.93642554964338 E28D00 --flat-values "RL retail" 7.153474758693154#-0.011 E28D00 --flat-values "RL retail dense" 7.057881063733782 E28D00 --flat-values "OPT retail" 6.879 1E90FF --flat-values "REC policy" 7.22#0.011 800020 --flat-values "SELF policy" 7.46 800020 --output-format pdf --output-file ~/OneDrive/rec_plots/rec_2_red_stochastic_50_3/results_rec2_global_only_mean --show-only-mean