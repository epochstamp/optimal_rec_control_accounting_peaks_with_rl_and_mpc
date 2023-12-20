from envs import create_env
import os
import click
import numpy as np
from envs import create_env_fcts
from utils.utils import unique_consecutives_values
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import time
import plotly.io as pio
pio.kaleido.scope.mathjax = None
FONT_FAMILY="NewComputerModern Roman, NewComputerModern08"



def aggregate_over_n_timesteps_f(time_serie, n=1):
    if n == 1:
        return np.asarray(time_serie)
    return np.sum(np.asarray(time_serie[:-1]).reshape(-1, n), axis=1)

@click.command()
@click.option('--env', "env", type=click.Choice(list(create_env_fcts.keys())), help='Environment to launch', required=True)
@click.option('--multiprice/--no-multiprice', "multiprice", is_flag=True, help='Whether (buying) are changing per metering period.')
@click.option('--aggregate-over-n-timesteps', "aggregate_over_n_timesteps", type=int, default=1, help='Number of time steps to aggregate for each x-axis tick.')
@click.option('--root-folder', "root_folder", type=str, help='Folder to register plots', default=os.path.expanduser("~")+"/rec_profiles")
@click.option('--output-file', "output_file", type=str, help='Folder to output files', default="consprod.pdf")
@click.option('--abbreviate/--no-abbreviate', "abbreviate", is_flag=True, help='Whether member names are abbreviated.')
@click.option('--merge-profiles/--no-merge-profiles', "merge_profiles", is_flag=True, help='Whether member profiles are merged by type of profile.')
@click.option('--legend-x-shift', "legend_x_shift", type=float, help='legend x shift', default=0.0)
@click.option('--n-samples', "n_samples", type=int, default=1, help='Number of samples to show for each time serie.')
@click.option('--font-size-increase', "font_size_increase", type=float, help='Font size increase', default=0.0)
@click.option('--font-size-increase-axes-label-ratio', "font_size_increase_axes_label_ratio", type=float, help='(10+font_size_increase)*font_size_increase_axes_label_ratio', default=1.0)
@click.option("--graph-width", "graph_width", help="Graph width", type=int, default=2000)
@click.option("--graph-height", "graph_height", help="Graph height", type=int, default=800)
def run_plot(env, multiprice, aggregate_over_n_timesteps, root_folder, output_file, legend_x_shift, merge_profiles, abbreviate, n_samples,font_size_increase, font_size_increase_axes_label_ratio, graph_width, graph_height):
    complete_path = root_folder+"/"+env+"/"+f"aggregated={aggregate_over_n_timesteps}"
    os.makedirs(complete_path, exist_ok=True)
    rec_env, _ = create_env(
        env,
        Delta_P_prime=0
    )
    
    consumption_profiles = dict()
    production_profiles = dict()
    labels = []
    pd.options.plotting.backend = "plotly"
    for i in range(1, n_samples+1):
        rec_env.reset()
        exogenous_members = rec_env.observe_all_members_exogenous_variables()
        k = "" if i == 1 else f" {i}"
        if merge_profiles:
            consumption_profiles_sample = {
                f"Aggregated consumption{k}": np.sum([aggregate_over_n_timesteps_f(exogenous_members[(member, "consumption")], n=aggregate_over_n_timesteps) for member in rec_env.members if (member, "consumption") in exogenous_members], axis=0) 
            }
            production_profiles_sample = {
                f"Aggregated production{k}": np.sum([aggregate_over_n_timesteps_f(exogenous_members[(member, "production")], n=aggregate_over_n_timesteps) for member in rec_env.members if (member, "production") in exogenous_members], axis=0) 
            }
            labels = [f"Aggregated consumption{k}", f"Aggregated production{k}"]
        elif abbreviate:
            consumption_profiles_sample = {
                f"M{mid+1} consumption{k}": aggregate_over_n_timesteps_f(exogenous_members[(member, "consumption")], n=aggregate_over_n_timesteps) for mid, member in enumerate(rec_env.members) if (member, "consumption") in exogenous_members
            }
            production_profiles_sample = {
                f"M{mid+1} production{k}": aggregate_over_n_timesteps_f(exogenous_members[(member, "production")], n=aggregate_over_n_timesteps) for mid, member in enumerate(rec_env.members) if (member, "production") in exogenous_members
            }
            labels_tmp = list(consumption_profiles_sample.keys()) + list(production_profiles_sample.keys())
            labels_tmp = sorted(labels_tmp, key=lambda k: int(k.split(" ")[0][1:]))
            labels = labels_tmp
            
        else:
            consumption_profiles_sample = {
                member+f" consumption{k}": aggregate_over_n_timesteps_f(exogenous_members[(member, "consumption")], n=aggregate_over_n_timesteps) for member in rec_env.members if (member, "consumption") in exogenous_members
            }
            production_profiles_sample = {
                member+f" production{k}": aggregate_over_n_timesteps_f(exogenous_members[(member, "production")], n=aggregate_over_n_timesteps) for member in rec_env.members if (member, "production") in exogenous_members
            }
            labels = list(consumption_profiles_sample.keys()) + list(production_profiles_sample.keys())
    
        consumption_profiles_sample = {
            k:v for k,v in consumption_profiles_sample.items() if sum(v) >= 1.0
        }
        production_profiles_sample = {
            k:v for k,v in production_profiles_sample.items() if sum(v) >= 1.0
        }
        
        df = pd.DataFrame.from_dict({**production_profiles_sample, **consumption_profiles_sample})
        #print(df.columns)
        df = df[labels]
        
        fig = df.plot()
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Energy (kWh)",
            font=dict(
                family=FONT_FAMILY,
                size=10
            ),
            legend=dict(
                title="",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=legend_x_shift,
                traceorder="normal"
            )
        )
        fig.update_layout(
            font_family=FONT_FAMILY,
            title_font_family=FONT_FAMILY,
            plot_bgcolor="white",
            font_size=10+font_size_increase
        )
        fig.update_traces(line={'width': 3})
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )

        fig.update_yaxes(tickfont=dict(size=(10+font_size_increase)*font_size_increase_axes_label_ratio), title_standoff = 5)
        fig.update_xaxes(tickfont=dict(size=(10+font_size_increase)*font_size_increase_axes_label_ratio), title_standoff = 5)
        complete_filename = complete_path + (
            f"/{output_file}_sample_{i}.pdf"
            if n_samples > 1
            else f"/{output_file}.pdf"
        )
        fig.write_image(complete_filename)

if __name__ == "__main__":
    run_plot()