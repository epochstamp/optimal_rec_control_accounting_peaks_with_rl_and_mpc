from pathlib import Path
import wandb
import click
import json
import warnings
import os
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import plotly.io as pio
import zipfile
import shutil
from tqdm import tqdm
import itertools
from scipy.interpolate import interp1d

from utils.utils import flatten
pio.kaleido.scope.mathjax = None

FONT_FAMILY="NewComputerModern Roman, NewComputerModern08"

#import matplotlib.font_manager
#fpaths = matplotlib.font_manager.findSystemFonts()

#for i in fpaths:
#    f = matplotlib.font_manager.get_font(i)
#    print(f.family_name)
#exit()

def combine_hex_values(d):
      d_items = sorted(d.items())
      tot_weight = sum(d.values())
      red = int(sum([int(k[:2], 16)*v for k, v in d_items])/tot_weight)
      green = int(sum([int(k[2:4], 16)*v for k, v in d_items])/tot_weight)
      blue = int(sum([int(k[4:6], 16)*v for k, v in d_items])/tot_weight)
      zpad = lambda x: x if len(x)==2 else '0' + x
      return zpad(hex(red)[2:]) + zpad(hex(green)[2:]) + zpad(hex(blue)[2:])

def trim_repeating_keys(d_dict):
    keys = list(d_dict.keys())
    keys_splitted = [k.split("|") for k in keys if k != "K"]
    keys_splitted = [list(set(l)) for l in list(map(list, zip(*keys_splitted)))]
    keys_to_rule_out = [k[0] for k in keys_splitted if len(k) == 1][1:]
    new_d_dict = {
        (("|".join([key for key in k.split("|") if key not in keys_to_rule_out]))):v for k,v in d_dict.items()
    }
    return new_d_dict

def remove_outliers_func(vec_tuple):
    vec = np.asarray(vec_tuple)
    Q1 = np.percentile(vec, 5, method='weibull')
    Q3 = np.percentile(vec, 95, method='weibull')
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    vec = vec[np.logical_and(vec>=lower, vec<=upper)]
    return vec[np.logical_and(vec>=lower, vec<=upper)]

def without_outliers(vec):
    return [
        (e[0],) + tuple(remove_outliers_func(e[1:])) for e in vec
    ]

default_aliases = {
    "perfect_foresight_mpc": "MPC policy",
    "perfect_foresight_mpc_commodity_peak_force": "MPC retail",
    "solution_chained_optimisation": "reuse previous MPC sequence",
    "fresh_optimisation": "restart MPC",
    "rec_2_noisy_provider": "with noisy REC 2 production",
    "rec_2": "with true REC 2 production",
    "rl": "RL policy",
    "rl_dense": "RL dense",
    "rl_commodity_peak_force": "RL retail",
    "rl_commodity_peak_force_dense": "RL retail dense",
    "pseudo_forecast_rec_2_100_sample": "α=1",
    "pseudo_forecast_rec_2_085_sample": "α=0.85",
    "pseudo_forecast_rec_2_050_sample": "α=0.5",
    "pseudo_forecast_rec_2_001_sample": "α=0.01",
    "pseudo_forecast_rec_2_025": "α=0.25",
    "pseudo_forecast_rec_7_100_sample": "α=1",
    "pseudo_forecast_rec_7_085_sample": "α=0.85",
    "pseudo_forecast_rec_7_050_sample": "α=0.5",
    "pseudo_forecast_rec_7_001_sample": "α=0.1",
    "pseudo_forecast_rec_7_025": "α=0.25",
    "pseudo_forecast_rec_7_095_sample": "α=0.95"
}

colors = {
    "perfect_foresight_mpc": ("74deff", 0.5),
    "perfect_foresight_mpc_commodity_peak_force": ("582900", 0.5),
    "solution_chained_optimisation": ("FF0000", 0.25),
    "fresh_optimisation": ("00FFFF", 0.25),
    "rec_2_noisy_provider": ("4C0099", 0.25),
    "rec_2": ("00FF00", 0.25),
    "pseudo_forecast_rec_2_100_sample": ("5BC5DB", 0.5),
    "pseudo_forecast_rec_2_085_sample": ("00FFFF", 0.5),
    "pseudo_forecast_rec_2_050_sample": ("4C0099", 0.5),
    "pseudo_forecast_rec_2_001_sample": ("A46750", 0.5),
    "pseudo_forecast_rec_7_100_sample": ("5BC5DB", 0.5),
    "pseudo_forecast_rec_7_085_sample": ("00FFFF", 0.5),
    "pseudo_forecast_rec_7_050_sample": ("4C0099", 0.5),
    "pseudo_forecast_rec_7_001_sample": ("A46750", 0.5)
}

colors["rl_dense"] = ("228B22", 1.0)
colors["rl_metering_dense"] = ("0000AA", 1.0)
colors["rl"] = ("A46750", 1.0)
colors["rl_commodity_peak_force"] = ("800080", 1.0)
colors["rl_commodity_peak_force_dense"] = ("ff7f00", 1.0)



@click.command()
@click.option('--folder-results', "folder_results", help='Results folders', required=True)
@click.option('--output-file', "output_file", help='Plot output folder', default=None)
@click.option('--plot-title', "plot_title", help='Plot title', default="")
@click.option('--output-format', "output_format", help='Plot output formats', multiple=True, default=["html", "pdf"], type=click.Choice(["html", "png", "pdf", "svg"]))
@click.option('--is-rl/--is-mpc', "is_rl", help='Whether it is about MPC or RL (results keys have different patterns)', is_flag=True)
@click.option("--group-by", "group_by", multiple=True, help="Group time series by these fields")
@click.option("--flat-values", "flat_values", multiple=True, nargs=3, type=(str, str, str), help="Plot flat values (e.g. optimal policy)")
@click.option("--vertical-dash-line-every", "vertical_dash_line_every", type=int, default=0, help="Plot vertical dash lines at these absiccas")
@click.option("--max-y", "max_y", help="Max y value", type=float, default=None)
@click.option("--min-y", "min_y", help="Min y value", type=float, default=None)
@click.option('--log-scale', "log_scale", help='Whether to plot log scale', is_flag=True)
@click.option("--where-equals", "where_equals", multiple=True, nargs=2, type=(str, str), help="Equality condition on fields")
@click.option("--aliases", "aliases", multiple=True, nargs=2, type=(str, str), help="Aliases for parameters to display in plot")
@click.option("--round-precision", "round_precision", help="Round precision", default=-1)
@click.option("--keep-best", "keep_best", multiple=True, type=str, help="Keep best time series within specified group", default=[])
@click.option("--remove-outliers", "remove_outliers", is_flag=True, help="Whether to remove outliers from time series (elem-wise)")
@click.option("--look-for-wandb-results", "look_for_wandb_results", is_flag=True, help="Whether to use wandb files instead of result files (useful when these latter have been lost for no reason...)")
@click.option('--value-to-plot', "value_to_plot", help='Plot output data', default=None, type=str)
@click.option("--min-number-of-points", "min_number_of_points", help="Minimum number of points", type=int, default=0)
@click.option("--top-values", "top_values", help="Select only N top values", type=int, default=10000000)
@click.option("--max-iters", "max_iters", help="Max number of iterations (only for RL)", type=int, default=10000000)
@click.option('--tar-gz-results', "tar_gz_results", help='Look for results archived as zip', is_flag=True)
@click.option('--sha-folders', "sha_folders", help='Whether folders are sha strings (look into parameters.json in that case)', is_flag=True)
@click.option("--mutual-exclude", "mutual_exclude", multiple=True, nargs=4, type=(str, str, str, str), help="Mutual exclusion between two pairs of values")
@click.option("--imply-equals", "imply_equals", multiple=True, nargs=4, type=(str, str, str, str), help="Imply equals between two pairs of values")
@click.option('--keep-first-legend-item', "keep_first_legend_item", help='Keep only first legend item for each trace', is_flag=True)
@click.option("--graph-width", "graph_width", help="Graph width", type=int, default=2000)
@click.option("--graph-height", "graph_height", help="Graph height", type=int, default=800)
@click.option('--legend-x-shift', "legend_x_shift", type=float, help='legend x shift', default=0.0)
@click.option('--max-x-axis', "max_x_axis", type=int, help='max x axis', default=-1)
@click.option('--min-x-axis', "min_x_axis", type=int, help='min x axis', default=-1)
@click.option('--annotation-overlap-thresold', "annotation_overlap_thresold", type=int, help='Annotation overlap thresold', default=-1)
@click.option('--margin-right', "margin_right", type=int, help='Plot margin right', default=-1)
@click.option('--divide-by', "divide_by", type=int, help='divide all values by', default=1)
@click.option('--show-only-mean', "show_only_mean", is_flag=True, help='Show only the mean value (no std)')
@click.option('--colors_per_line', "colors_per_line", multiple=True, default=None, help='Sequence of colors. If empty, it will take the colors by default. If it contains a single element named "auto", then it will use colors as defined in this script. If number of colors < number of lines, the colors will be repeated.')
@click.option('--remove-margins', "remove_margins", is_flag=True, help='Remove margins')
@click.option('--font-size-increase', "font_size_increase", type=float, help='Font size increase', default=0.0)
@click.option('--font-size-increase-axes-label-ratio', "font_size_increase_axes_label_ratio", type=float, help='(10+font_size_increase)*font_size_increase_axes_label_ratio', default=1.0)
@click.option('--fit-x-axis', "fit_x_axis", is_flag=True, help='Fit x axis')
@click.option('--keep-distance-between-x-axis-if-fit', "keep_distance_between_x_axis_if_fit", type=float, help='When fit_x_axis, remove Ks that are in a distance lower than this value', default=1)
@click.option('--resample', "resample", type=int, help='Resample time series to', default=None)
@click.option('--reshape-mean-time-series', "reshape_mean_time_series", type=(float, float), help='Reshape mean time series by normalizing them to be norm 1 and then multiply by the value in input', default=None)
@click.option('--prefix-id', "prefix_id", is_flag=True, help='Prefix id for time series keys')
@click.option('--keep-id-and-rename-series', "keep_id_and_rename_series", type=(str, str), nargs=2, help='Rename series identified with "id" (provide as many times as number of retained ids)', multiple=True, default=[])
@click.option('--perturbate', "perturbate", is_flag=True, help='Whether to add noise in data')
def run_experiment(folder_results, output_file, plot_title, output_format, is_rl, group_by, flat_values, vertical_dash_line_every, max_y, min_y, log_scale, where_equals, aliases, round_precision, keep_best, remove_outliers, look_for_wandb_results, value_to_plot, min_number_of_points, top_values, tar_gz_results, sha_folders, max_iters, mutual_exclude, imply_equals, keep_first_legend_item, graph_width, graph_height, legend_x_shift, max_x_axis, min_x_axis, annotation_overlap_thresold, margin_right, divide_by, show_only_mean, colors_per_line, remove_margins, font_size_increase, font_size_increase_axes_label_ratio, fit_x_axis, keep_distance_between_x_axis_if_fit, resample, reshape_mean_time_series, prefix_id, keep_id_and_rename_series, perturbate):
    d_dict = dict()
    group_by_lst = group_by
    if group_by_lst == [] or (not is_rl and "mpc_policy" not in group_by_lst) or (is_rl and "rl_env" not in group_by_lst):
        group_by_lst = [("rl_env" if is_rl else "mpc_policy")]
    policies = dict()
    i=0
    policy_to_id = dict()
    K_lst = dict()
    where_equals_dict = dict()
    aliases_dict = dict(default_aliases)
    if aliases != []:
        for k, a in aliases:
            aliases_dict[k] = a
    aliases = aliases_dict
    for k,v in where_equals:
        if k not in where_equals_dict:
            where_equals_dict[k] = []
        where_equals_dict[k] += [str(v)]
    if output_file is not None:
        output_file_path = Path(output_file)
        os.makedirs(output_file_path.parent, exist_ok=True)

    key_to_dict = dict()
    subpattern = "result" if not look_for_wandb_results else "post_to_wandb"
    if tar_gz_results:
        pattern = "results.zip" if not look_for_wandb_results else "post_to_wandb.zip"
    else:
        pattern = f"{subpattern}*.json" if is_rl else f"{subpattern}.json"
    root_path = Path(f"{folder_results}/")
    lst_files = list(root_path.rglob(pattern))
    param_value_sets = dict()
    for path in tqdm(lst_files):
        skip=False
        if sha_folders:
            parameters_filepath = f"{os.path.dirname(path)}/parameters.json"
            with open(parameters_filepath, "r") as parameters_file:
                try:
                    param_value_dict = json.load(
                        parameters_file
                    )
                except BaseException as e:
                    warnings.warn(f"Parameters json file cannot be read. Details: {e}. File : {parameters_filepath}")
                    continue
        else:
            lst_folders = [f.split("=") for f in str(path).split("/") if "=" in f]
            param_value_dict = {
                k:v for k,v in lst_folders
            }
        for k,v in where_equals_dict.items():
            if str(param_value_dict[k]) not in v:
                skip=True
                break
        for k,v in param_value_dict.items():
            if k not in param_value_sets:
                param_value_sets[k] = set()
            param_value_sets[k].add(v)

        if skip:
            continue
        for tup in mutual_exclude:
            k1, v1, k2, v2 = tup
            if k1 in param_value_dict.keys() and k2 in param_value_dict.keys():
                if str(param_value_dict[k1]) == v1 and str(param_value_dict[k2]) == v2:
                    skip=True
                    break
        if skip:
            continue
        for tup in imply_equals:
            k1, v1, k2, v2 = tup
            if k1 in param_value_dict.keys() and k2 in param_value_dict.keys():
                if (str(param_value_dict[k1]) == v1 and str(param_value_dict[k2]) != v2):
                    skip=True
                    break

        if skip:
            continue
        if not tar_gz_results:
            result_files = [path]
        else:
            with zipfile.ZipFile(path,"r") as tar_results:
                tar_results.extractall(path=f"{os.path.dirname(path)}/{subpattern}/")
            result_files = list(root_path.rglob(subpattern+"*.json"))
        for result_file in result_files:
            try:
                with open(result_file, "r") as rfile:
                    result_data = json.load(rfile)
                if look_for_wandb_results:
                    result_data = result_data["results"]
            except BaseException as e:
                warnings.warn(f"Result json file cannot be read. Details: {e}. File : {result_file}")
                continue
            
            
            
            key = "|".join(tuple([(param_value_dict[k] if not str(param_value_dict[k]).replace(".", "").replace("-", "").replace("#", "").isnumeric() else f"{k}={param_value_dict[k]}") for k in group_by_lst]))
            if key not in key_to_dict:
                key_to_dict[key] = {
                    k: param_value_dict[k] for k in group_by_lst
                }
            if key not in policy_to_id:
                policies[i] = key
                policy_to_id[key] = i
                i+=1
            
            if not is_rl:
                K = int(param_value_dict["K"])
                key_effective_bill = "expected_effective_bill"
                
            else:
                K = int(result_file.stem.split("_")[-1])
                key_effective_bill = "Expected Effective Bill"

            if value_to_plot is not None:
                key_effective_bill = value_to_plot
            expected_effective_bill = result_data[key_effective_bill]
            if key not in d_dict:
                d_dict[key] = []
            
            new_K = True
            idx_lst = None
            lst_elems = list(d_dict[key]) 
            for i, e in enumerate(lst_elems):
                if e[0] == K:
                    new_K = False
                    idx_lst = i
                    break
            if not new_K:
                d_dict[key][idx_lst] = d_dict[key][idx_lst] + (expected_effective_bill,) 
            else: 
                d_dict[key] += [(K, expected_effective_bill)]
        if tar_gz_results:
            shutil.rmtree(f"{os.path.dirname(path)}/{subpattern}/", ignore_errors=True)
    
    
    #pprint(
    #    {
    #        k:len(v) for k,v in param_value_sets.items() if len(v) > 1
    #    }
    #)
    #exit()
    #pprint(d_dict)
    #print(len(list(d_dict.keys())))
    from pprint import pprint
    d_dict = {
        k:(v if not remove_outliers else without_outliers(v))[:max_iters] for k,v in d_dict.items() if len(v) >= min_number_of_points
    }
    
    d_dict = {
        k:[(e[0],) + tuple(sorted(e[1:])[:top_values]) for e in v] for k,v in d_dict.items()
    }
    if prefix_id:
        d_dict = {
            f"({idx}) "+k:v for idx,(k,v) in enumerate(d_dict.items()) if k != "K"
        }
    if len(keep_id_and_rename_series) > 0:
        keep_id_and_rename_series_dct = {
            k:v for k,v in keep_id_and_rename_series
        }
        d_dict = {
            keep_id_and_rename_series_dct.get(str(idx), k): v for idx, (k,v) in enumerate(d_dict.items()) if str(idx) in keep_id_and_rename_series_dct.keys()
        } 

    
    
    if d_dict == dict():
        print("No data has been found accordingly to your criterions")
        exit()
    if not show_only_mean:
        d_dict_errors = {
            k:[(e[0], np.std(e[1:], ddof=1)/np.sqrt(len(e[1:])) if len(e[1:]) > 1 else 0) for e in v] for k,v in d_dict.items()
        }
        d_dict_errors_items = d_dict_errors.items()
    d_dict = {
        k:[(e[0], float(np.mean(e[1:]))) for e in v] for k,v in d_dict.items()
    }
   

    
    

    K_lst = {
        k:set(flatten([
           [e[0] for e in v]
        ])) for k, v in d_dict.items()
    }

    


    
    K_lst = sorted(set.intersection(*list(K_lst.values())))
    d_dict_items = list(d_dict.items())
    
    d_dict = {
        key:list(zip(*sorted([val for val in v if val[0] in K_lst], key=lambda k: k[0])))[1] for key, v in d_dict_items
    }

    if not show_only_mean:
        d_dict_errors = {
            key:list(zip(*sorted([val for val in v if val[0] in K_lst], key=lambda k: k[0])))[1] for key, v in d_dict_errors_items
        }
    
    d_dict["K"] = K_lst
    d_dict = trim_repeating_keys(d_dict)
    if not show_only_mean:
        d_dict_errors = trim_repeating_keys(d_dict_errors)

    if len(keep_best) > 0:
        key_to_dict = trim_repeating_keys(key_to_dict)
        keep_best_str = "|".join(keep_best)
        group_keys = dict()
        for key in d_dict.keys():
            if key != "K":
                dict_param_keys = key_to_dict[key]
                keep_best_key = "|".join([dict_param_keys[kb] for kb in keep_best]) 
                if keep_best_key not in group_keys:
                    group_keys[keep_best_key] = []
                group_keys[keep_best_key].append(key)
        key_argmins = []
        for key, keys in group_keys.items():
            argmin = min(keys, key=lambda k: np.mean(d_dict[k][-50:]))
            key_argmins.append(argmin)
        d_dict = {
            k:v for k,v in d_dict.items() if k=="K" or k in key_argmins
        }
        if not show_only_mean:
            d_dict_errors = {
                k:v for k,v in d_dict_errors.items() if k=="K" or k in key_argmins
            }

    #print(d_dict)
    if resample is not None:
        d_keys = list(d_dict.keys())
        new_K = np.arange(resample)
        new_x = np.linspace(0, 1, resample)
        for k in d_keys:
            if k != "K":
                ts = d_dict[k]
                x = np.linspace(0, 1, len(ts))
                interpolator = interp1d(x, ts, kind="linear")
                new_ts = interpolator(new_x)
                d_dict[k] = new_ts
                if not show_only_mean:
                    ts_error = d_dict_errors[k]
                    interpolator = interp1d(x, ts_error, kind="linear")
                    d_dict_errors[k] = interpolator(new_x)
            else:
                d_dict["K"] = new_K
                if not show_only_mean:
                    d_dict_errors["K"] = new_K

    if reshape_mean_time_series is not None:
        d_dict = {
            k:(v if k == "K" else list((np.asarray(v)/reshape_mean_time_series[0])*reshape_mean_time_series[1])) for k, v in d_dict.items()
        }

    if perturbate:
        np.random.seed(906)
        def noise_it(mu, sigs):
            
            return np.asarray([
                (np.clip(np.random.normal(mu[i], sigs[i]/2), (mu[i] - sigs[i]), (mu[i] + sigs[i]))) for i in range(len(mu))
            ])
        d_dict = {
            k:(v if (k == "K") else noise_it(v, d_dict_errors[k])) for k,v in d_dict.items()
        }
    df = pd.DataFrame.from_dict(d_dict)
    sorted_lines = sorted(d_dict.keys(), key=lambda k: d_dict[k][-1])
    df = df[sorted_lines]
    df = df.set_index("K")
    sorted_lines = [k for k in sorted_lines if k != "K"]
    
    
    pd.options.plotting.backend = "plotly"
    kwargs_fig = {}
    min_value = None
    if log_scale:
        min_value = df.min()
        while type(min_value) not in (int, float, np.float32, np.float64):
            min_value = min_value.min()
        #min_value = df.min().min()
        for _, flat_value_raw, _ in flat_values:
            if "#" not in flat_value_raw:
                flat_value = float(flat_value_raw)
            else:
                flat_value = float(flat_value_raw.split("#")[0])
            min_value = min(flat_value, min_value)
        df = df-float(min_value) + 1
    
    fig = px.line(df, render_mode="svg", log_y=log_scale,  markers=False)
    
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
    if min_y is not None and max_y is not None:
        if log_scale:
            min_y = np.log(min_y - min_value + 1)
            max_y = np.log(max_y - min_value + 1)
        fig.update_layout(yaxis_range=[min_y,max_y])
    if vertical_dash_line_every > 0:
        for k in range(min(K_lst), max(K_lst)+1):
            if k % vertical_dash_line_every == 0:
                fig.add_vline(x=k, line_width=2, line_dash="dash", line_color="black", opacity=0.3)

    for flat_key, flat_value_data, flat_color in flat_values:
        if "#" not in flat_value_data:
            flat_value_raw = float(flat_value_data)
            yshift=0.0
        else:
            flat_data = flat_value_data.split("#")
            flat_value_raw = float(flat_data[0])
            yshift = float(flat_data[1])
        flat_value = flat_value_raw/divide_by
        if log_scale:
            flat_value = flat_value - min_value + 1
        if round_precision > -1:
            flat_value = round(flat_value, round_precision)
        
        kwargs_annotation = {}
        if log_scale:
            kwargs_annotation["annotation_y"] = np.log10(flat_value)
        fig.add_hline(
            y=float(flat_value),
            #annotation_text=flat_key+f"~={flat_value}",
            #annotation_position="outside bottom right",
            line_dash="dot",
            line_color="#" + flat_color.replace("#", ""),
            #annotation_font_color="#000000",
            #annotation_font_size=10,
            #annotation_font_family=FONT_FAMILY,
            line_width=1,
            **kwargs_annotation
        )
        fig.add_annotation(
            text=flat_key,
            xref="paper",
            x=1,
            xanchor="left",
            y=float(flat_value) + float(yshift),
            showarrow=False,
            font_size=10+font_size_increase,
            font_color="#" + flat_color.replace("#", ""),
            font_family=FONT_FAMILY

        )
    if plot_title is None:
        plot_title = "Expected Effective Bill"
    fig.update_layout(
        title=plot_title,
        xaxis_title=("Policy Horizon" if not is_rl else "Iterations"),
        yaxis_title="Expected return",
        xaxis_title_standoff=5,
        yaxis_title_standoff=1,
        legend_title="Policies",
        font=dict(
            family=FONT_FAMILY,
            size=10+font_size_increase,
            color="black"
        )
    )
    
    if remove_margins:
        fig.update_layout(
            margin=dict(r=0.0, l=0.0, b=0, t=0)
        )
    elif margin_right != -1:
        fig.update_layout(
            margin=dict(r=margin_right)
        )

    if round_precision > -1:
        fig.update_layout(
            yaxis = dict(
                tickformat = f'.{round_precision}f',
            )
        )
    
    if not show_only_mean:
        def create_add_error_trace(fig):
            def add_error_trace(trace):
                x = list(trace['x'])
                y_upper = list(trace['y'] + np.asarray(d_dict_errors[trace.name]))
                y_lower = list(trace['y'] - np.asarray(d_dict_errors[trace.name]))
                from pprint import pprint
                color = f"rgba({tuple(int(trace['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace('((','(').replace('),',',').replace(' ','')
                fig.add_trace(
                        go.Scatter(
                            x = x+x[::-1],
                            y = y_upper+y_lower[::-1],
                            fill = 'toself',
                            fillcolor = color,
                            line = dict(
                                color = 'rgba(255,255,255,0)'
                            ),
                            hoverinfo = "skip",
                            showlegend = False,
                            legendgroup = trace['legendgroup'],
                            xaxis = trace['xaxis'],
                            yaxis = trace['yaxis'],
                        )
                    )
            return add_error_trace
        add_error_trace = create_add_error_trace(fig)
        fig.for_each_trace(
            add_error_trace#lambda t: t.update(error_y = {"array": d_dict_errors[t.name]})
        )

    
    fig.update_traces(marker={'size': 8})
    
    if len(colors_per_line) > 0:
        if len(colors_per_line) == 1 and colors_per_line[0] == "auto":
            fig.for_each_trace(lambda t: t.update(line_color = "#" + combine_hex_values({h:w for h,w in [colors.get(k, ("000000", 0.05)) for k in t.name.split("|")]})))
        else:
            number_of_lines = len(list(d_dict.keys())) - 1
            if len(colors_per_line) < number_of_lines:
                colors_per_line = (list(colors_per_line) * number_of_lines)[:number_of_lines]
            color_per_line = {
                k:colors_per_line[i] for i, k in enumerate(sorted_lines)
            }
            fig.for_each_trace(lambda t: t.update(line_color = "#" + color_per_line[k].replace("#", "")))
    def try_update_trace(t):
        if t.name is None:
            return
        t.update(name = aliases.get(t.name.split("|")[0], t.name.split("|")[0]) +  (" " + " ".join([("("+aliases.get(n, n)+")") for n in t.name.split("|")[1:]]) if not keep_first_legend_item else ""))
    fig.for_each_trace(try_update_trace)

    fig.update_layout(
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
    if max_x_axis != -1 or min_x_axis != -1:
        max_x_axis_val = max_x_axis if max_x_axis != -1 else 0
        min_x_axis_val = min_x_axis if min_x_axis != -1 else 0
        fig.update_xaxes(range=[min_x_axis_val, max_x_axis_val])
    

    fig.update_layout(
        font_family=FONT_FAMILY,
        title_font_family=FONT_FAMILY,
        plot_bgcolor="white",
        font_size=10+font_size_increase
    )

    if fit_x_axis:
        new_K_lst = []
        j = 0
        for i in range(len(K_lst)):
            if i == 0 or (K_lst[i]==721) or (K_lst[i]>=110 and K_lst[i]<=266 and j%2 == 1) or (K_lst[i] - K_lst[i-1]) >= keep_distance_between_x_axis_if_fit:
                if K_lst[i] != 714:
                    new_K_lst.append(K_lst[i])
            j += 1
                
        fig.update_xaxes(tickvals=new_K_lst)
    fig.update_yaxes(tickfont=dict(size=(10+font_size_increase)*font_size_increase_axes_label_ratio), title_standoff = 5)
    fig.update_xaxes(tickfont=dict(size=(10+font_size_increase)*font_size_increase_axes_label_ratio), title_standoff = 5)
        
    if output_file is not None:
        for of in output_format:
            if of == "html":
                fig.write_html(str(output_file_path)+".html")
            elif of in ("pdf", "png", "svg"):
                pio.write_image(fig, str(output_file_path)+f".{of}", scale=1, width=graph_width, height=graph_height)
    else:
        fig.show()
    
if __name__ == '__main__':
    run_experiment()