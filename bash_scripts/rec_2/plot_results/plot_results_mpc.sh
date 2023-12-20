python -m experiment_scripts.generic.plot_results --folder-results ../rec_experiments/MPC --group-by mpc_policy --group-by solution_chained_optimisation --group-by env_data_provider --vertical-dash-line-every 20 --where-equals env rec_2 --where-equals mpc_policy perfect_foresight_mpc --where-equals mpc_policy perfect_foresight_mpc_commodity_peak_force --where-equals rescaled_gamma rescale_terminal --round-precision 3 --flat-values "Optimal Policy" "#5BC5DB" 4.289135956324137 --flat-values "Optimal Commodity Policy" "#5387DD" 4.530833947928269 --flat-values "REC Consumption" "#CCCC00" 4.561236431896849 --flat-values "SELF consumption" "#FF0000" 5.024034206930218 --output-file ~/OneDrive/rec_plots/rec_2/mpc