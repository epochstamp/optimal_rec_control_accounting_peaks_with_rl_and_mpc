python -m experiment_scripts.generic.plot_results --folder-results ../rec_experiments/MPC --group-by env --where-equals env rec_2_red_stochastic_95_1 --group-by mpc_policy --group-by exogenous_data_provider --vertical-dash-line-every 20 --round-precision 3 --flat-values-with-placement "Optimal Policy" "#5BC5DB" "bottom left" 4.06 --flat-values "Optimal Commodity Policy" "#5387DD" 4.3 --flat-values "REC Consumption" "#CCCC00" 4.635336004446144 --flat-values "SELF consumption" "#FF0000" 4.917942361803348 --output-file ~/OneDrive/rec_plots/rec_2_red_stochastic_95_1/mpc --output-format html --max-y 6 --min-y 4 --plot-title "Expected Effective Bill with MPC on REC2 (stochastic)" #--where-equals exogenous_data_provider pseudo_forecast_rec_2_099 --where-equals exogenous_data_provider pseudo_forecast_rec_2_050 --where-equals exogenous_data_provider pseudo_forecast_rec_2_025