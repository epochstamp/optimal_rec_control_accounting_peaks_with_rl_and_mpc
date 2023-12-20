python -m experiment_scripts.generic.plot_results --folder-results ../rec_experiments/MPC/ --group-by mpc_policy --group-by rescaled_gamma --vertical-dash-line-every 480 --where-equals rescaled_gamma rescale_terminal --where-equals env rec_28_summer_end --round-precision 2 --flat-values "Optimal Policy" "#5BC5DB" 286935.286311707 --flat-values "Optimal Commodity Policy" "#5387DD" 287674.45486468205 --output-file ~/OneDrive/rec_plots/rec_28_summer_end/mpc  --where-equals Delta_P 120 --flat-values "REC Consumption" "#CCCC00" 287939.5149536645 --flat-values-with-placement "SELF consumption" "#FF0000" "top right" 287676.56573432713 --where-equals Delta_P 120