
import datetime
import json
import os
import warnings
import click
import sys
import numpy as np
from experiment_scripts.mpc.mpc_policies_zoo import mpc_policies
from slurm_utils.break_sweep_dict import break_sweep_dict
from envs import create_env_fcts
import itertools
from experiment_scripts.mpc import solver_params_dict, solvers
from experiment_scripts.mpc.exogenous_data_provider_zoo import exogenous_data_provider_zoo
'''
A script for generating SLURM submission scripts which sweep parameters

author: Tyson Jones
        tyson.jones@materials.ox.ac.uk
date:   7 Jan 2018
'''


# SLURM fields assumed if the particular field isn't passed to get_script
# can contain unused fields

DEFAULT_SLURM_FIELDS = {
    'memory': 8,
    'memory_unit': 'GB',
    'num_nodes': 1,
    'num_cpus': 3,
    'time_d': 0, 'time_h': 4, 'time_m': 0, 'time_s': 0,
    'reserve': 'nqit',
    'job_name': 'mpc_experiment',
    'output': 'mpc_outputs/output_%A_%a.txt',
    "ntasks": 1,
    "mem_per_cpu": 4096,
    "partition": "batch,debug"
}



# a template for the entire submit script
# (bash braces must be escaped by doubling: $var = ${{var}})
# num_jobs, param_arr_init, param_val_assign and param_list are special fields

TEMPLATE = '''

#!/bin/env bash

#SBATCH --array=0-{num_jobs}
#SBATCH --job-name={job_name}
#SBATCH --output={output}
#SBATCH --time={time_d}-{time_h}:{time_m}:{time_s}
#SBATCH --ntasks={ntasks}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --partition={partition}

{param_arr_init}

trial=${{SLURM_ARRAY_TASK_ID}}
{param_val_assign}

'''.strip()



# functions for making bash expressions
# bash braces are escaped by doubling

def _mth(exp):
    return '$(( %s ))' % exp
def _len(arr):
    return '${{#%s[@]}}' % arr
def _get(arr, elem):
    return '${{%s[%s]}}' % (arr, elem)
def _eq(var, val):
    return '%s=%s' % (var, val)
def _op(a, op, b):
    return _mth('%s %s %s' % (a, op, b))
def _arr(arr):
    return '( %s )' % ' '.join(map(str, arr))
def _seq(a, b, step):
    return '($( seq %d %d %d ))' % (a, step, b)
def _var(var):
    return '${%s}' % var



# templates for param array construction and element access

PARAM_ARR = '{param}_values'
PARAM_EXPRS = {
    'param_arr_init':
        _eq(PARAM_ARR, '{values}'),
    'param_val_assign': {
        'assign':
            _eq('{param}', _get(PARAM_ARR, _op('trial','%',_len(PARAM_ARR)))),
        'increment':
            _eq('trial', _op('trial', '/', _len(PARAM_ARR)))
    }
}



def _to_bash(obj):
    if isinstance(obj, range):
        return _seq(obj.start, obj.stop - 1, obj.step)
    if isinstance(obj, list) or isinstance(obj, tuple):
        return _arr(obj)
    raise ValueError('Unknown object type %s' % type(obj).__name__)



def _get_params_bash(params, values):
    # builds bash code to perform the equivalent of
    '''
    def get_inds(params, ind):
        inds = []
        for length in map(len, params):
            inds.append(ind % length)
            ind //= length
        return inds[::-1]
    '''

    # get lines of bash code for creating/accessing param arrays
    init_lines = []
    assign_lines = []
    init_temp = PARAM_EXPRS['param_arr_init']
    assign_temps = PARAM_EXPRS['param_val_assign']

    for param, vals in zip(params, values):
        init_lines.append(
            init_temp.format(param=param, values=_to_bash(vals)))
        assign_lines.append(
            assign_temps['assign'].format(param=param))
        assign_lines.append(
            assign_temps['increment'].format(param=param))

    # remove superfluous final trial reassign
    assign_lines.pop()

    return init_lines, assign_lines



def get_script(fields, params, param_order=None):
    '''
    returns a string of a SLURM submission script using the passed fields
    and which creates an array of jobs which sweep the given params

    fields:      dict of SLURM field names to their values. type is ignored
    params:      a dict of (param names, param value list) pairs.
                 The param name is the name of the bash variable created in
                 the submission script which will contain the param's current
                 value (for that SLURM job instance). param value list is
                 a list (or range instance) of the values the param should take,
                 to be run once against every other possible configuration of all params.
    param_order: a list containing all param names which indicates the ordering
                 of the params in the sweep. The last param changes every
                 job number. If not supplied, uses an arbitrary order
    '''

    # check arguments have correct type
    assert isinstance(fields, dict)
    assert isinstance(params, dict)
    assert (isinstance(param_order, list) or
            isinstance(param_order, tuple) or
            param_order==None)
    if param_order == None:
        param_order = list(params.keys())

    # check each field appears in the template
    for field in fields:
        if ('{%s}' % field) not in TEMPLATE:
            raise ValueError('passed field %s unused in template' % field)

    # calculate total number of jobs (minus 1; SLURM is inclusive)
    num_jobs = 1
    for vals in params.values():
        num_jobs *= len(vals)
    num_jobs -= 1

    # get bash code for param sweeping
    init_lines, assign_lines = _get_params_bash(
        param_order, [params[key] for key in param_order])

    # build template substitutions (overriding defaults)
    subs = {
        'param_arr_init': '\n'.join(init_lines),
        'param_val_assign': '\n'.join(assign_lines),
        'param_list': ', '.join(map(_var, param_order)),
        'num_jobs': num_jobs
    }
    for key, val in DEFAULT_SLURM_FIELDS.items():
        subs[key] = val
    for key, val in fields.items():
        subs[key] = val
    return TEMPLATE.format(**subs), num_jobs


def save_script(filename, fields, params, param_order=None):
    '''
    creates and writes to file a SLURM submission script using the passed
    fields and which creates an array of jobs which sweep the given params

    fields:      dict of SLURM field names to their values. type is ignored
    params:      a dict of (param names, param value list) pairs.
                 The param name is the name of the bash variable created in
                 the submission script which will contain the param's current
                 value (for that SLURM job instance). param value list is
                 a list (or range instance) of the values the param should take,
                 to be run once against every other possible configuration of all params.
    param_order: a list containing all param names which indicates the ordering
                 of the params in the sweep. The last param changes every
                 job number. If not supplied, uses an arbitrary order
    '''
    
    script_str = get_script(fields, params, param_order)
    if ('/' in filename) or ('\\' in filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        file.write(script_str)

from utils.utils import merge_dicts
@click.command()
@click.option('--env', "env", type=click.Choice(list(create_env_fcts.keys())), help='Environment to launch', required=True, multiple=True)
@click.option('--exogenous-data-provider', "exogenous_data_provider", type=click.Choice(list(exogenous_data_provider_zoo.keys())), help='Exogenous data provider', default=["none"], multiple=True)
@click.option('--Delta-M', "Delta_M", type=int, default=[2], help='Delta_M.', multiple=True)
@click.option('--Delta-P', "Delta_P", type=int, default=[1], help='Delta_P.', multiple=True)
@click.option('--Delta-P-prime', "Delta_P_prime", type=int, default=[0], help='Delta_P_prime.', multiple=True)
@click.option('--min-K', "minK", type=int, default=[1], help='min K.', multiple=True)
@click.option('--max-K', "maxK", type=int, default=[10], help='max K.', multiple=True)
@click.option('--step-K', "stepK", type=int, default=[1], help='step K.', multiple=True)
@click.option('--T', "T", type=int, default=None, help='T.')
@click.option('--gamma', "gamma", type=float, default=[0.99], help='Discount factor (env side).', multiple=True)
@click.option('--gamma-policy', "gamma_policy", type=float, default=[0.0], help='Discount factor (policy side).', multiple=True)
@click.option('--small-pen-ctrl-actions', "small_pen_ctrl_actions", type=float, default=[0.0], help='Small penalty applied to small pen actions.', multiple=True)
@click.option('--rescaled-gamma-mode', "rescaled_gamma_mode", type=click.Choice(["no_rescale", "rescale_terminal", "rescale_delayed_terminal"]), help='Gamma rescale mode', default=["no_rescale"], multiple=True)
@click.option("--solver-config", "solver_config", type=click.Choice(solver_params_dict.keys()), help="Id of Solvers pre-registered params", default=["none"], multiple=True)
@click.option("--solver", "solver", type=click.Choice(solvers), help="Id of Solvers pre-registered", default=["cplex"], multiple=True)
@click.option('--remove-current-peak-costs-flags', "remove_current_peak_costs_flags", type=click.Choice(["remove-current-peak-costs", "no-remove-current-peak-costs"]), default=["no-remove-current-peak-costs"], help='Whether current peak costs are removed.', multiple=True)
@click.option('--remove-historical-peak-costs-flags', "remove_historical_peak_costs_flags", type=click.Choice(["remove-historical-peak-costs", "no-remove-historical-peak-costs"]), default=["remove-historical-peak-costs"], help='Whether historical peak costs are removed.', multiple=True)
@click.option('--n-samples-policy', "n_samples_policy", type=int, default=[1], help='Number of times to repeat exogenous sequences for policy.', multiple=True)
@click.option('--n-samples', "n_samples", type=int, default=[1], help='Number of times to simulate policy in env.', multiple=True)
@click.option('--multiprice-flags', "multiprice_flags", type=click.Choice(["multiprice", "no-multiprice"]), default=["multiprice", "no-multiprice"], help='Whether (buying) are changing per metering period.', multiple=True)
@click.option('--random-seed', "random_seed", type=int, default=0, help='Random seed.')
@click.option('--nb-random-seed', "nb_random_seeds", type=int, default=0, help='Random seed (0 for not modifying random seeds).')
@click.option('--mpc-policy', "mpc_policy", type=click.Choice(mpc_policies.keys()), help='Policy to execute', default=list(mpc_policies.keys()), multiple=True)
@click.option('--net-cons-prod-mutex-before', "net_cons_prod_mutex_before", type=int, help="From which timestep net cons prod mutex are disabled", default=[10000000], multiple=True)
@click.option("--root-dir", "root_dir", default=os.path.expanduser('~'), help="Root directory")
@click.option("--lock-dir", "lock_dir", default=os.path.expanduser('~'), help="Lock (usually shared) directory")
@click.option('--id-job', "id_job", type=str, help='ID suffix of the job', default="")
@click.option('--job-dir-name', "job_dir_name", type=str, help='Jobs directory', default="jobs")
@click.option('--partitions', "partitions", type=str, help='Partitions (separated by commas)', default="batch,debug")
@click.option('--n-cpus', "n_cpus", type=int, help='Number of cpus', default=None)
@click.option('--mem-per-cpu', "mem_per_cpu", type=int, help='Memory per CPU', default=4096)
@click.option('--max-n-jobs', "max_n_jobs", type=int, help='Maximum number of jobs per script', default=sys.maxsize)
@click.option('--n-tasks', "n_tasks", type=int, help='Number of tasks to launch for that script', default=1)
@click.option('--output', "output", type=str, help='Slurm output pattern', default="mpc_outputs/output_%A_%a.txt")
@click.option('--time-limits', "time_limits", type=int, nargs=4, help='Time limit (day hour minuts seconds)', default=[0, 2, 0, 0])
@click.option("--time-table-file", "time_table_file", type=str, help="Time table for fine tuning time limit", default=None)
@click.option('--margin-time-table', "margin_time_table", type=float, help='Margin on time table', default=1.5)
@click.option("--wandb-project", "wandb_project", default="mpcstudy", help="Wandb project name")
@click.option("--wandb-offline", "wandb_offline", is_flag=True, help="Whether wandb is turned offline")
@click.option('--time-policy/--no-time-policy', "time_policy", is_flag=True, help='Whether to activate time-policy option.')
@click.option('--solver-verbose', "solver_verbose", is_flag=True, help='Whether to activate solver verbose.')
@click.option("--solution-chained-optimisation-flags", type=click.Choice(["solution-chained-optimisation", "fresh-optimisation"]), multiple=True, default=["fresh_optimisation"], help="Chained optimisation flags (EXPERIMENTAL, USE CAREFULLY)")
@click.option("--disable-env-ctrl-assets-constraints-flags", type=click.Choice(["disable-env-ctrl-assets-constraints", "enable-env-ctrl-assets-constraints"]), multiple=True, default=["disable-env-ctrl-assets-constraints"], help="Disable ctrl assets action flags (EXPERIMENTAL, USE CAREFULLY)")
def run_experiment(**kwargs):
    
    """
    script = get_script({}, {'a':range(10), 'b':range(10), 'c':range(10)},
                        param_order=['c','a','b'])
    print(script)
    """
    if kwargs["nb_random_seeds"] > 0:
        random_seed = kwargs["random_seed"]
        if random_seed == 0:
            random_seed = int(np.random.randint(low=1, high=1000000))
        np.random.seed(random_seed)
        random_seeds = np.random.randint(low=0, high=1000000, size=kwargs["nb_random_seeds"])
        
        kwargs["random_seed"] = list(random_seeds)
    else:
        kwargs.pop("random_seed", None)
    kwargs.pop("nb_random_seeds", None)
    time_policy = kwargs.pop("time_policy", False)
    solver_verbose = kwargs.pop("solver_verbose", False)
    #kwargs["minK"] = max(kwargs["minK"], 1)
    K = []
    for k in range(len(kwargs["minK"])):
        minK = max(kwargs["minK"][k], 1)
        minK = min(kwargs["minK"][k], kwargs["maxK"][k])
        maxK = max(kwargs["minK"][k], kwargs["maxK"][k])
        stepK = max(kwargs["stepK"][k], 1)
        K += list(np.arange(minK, maxK + 1, stepK))
    K = tuple(K)
    kwargs.pop("minK")
    kwargs.pop("maxK")
    kwargs.pop("stepK")
    T = kwargs["T"]
    ntasks = kwargs.pop("n_tasks")
    kwargs.pop("T")
    time_limits = kwargs["time_limits"]
    kwargs.pop("time_limits")
    kwargs["K"] = K
    root_dir = kwargs["root_dir"]
    kwargs.pop("root_dir")
    output = kwargs.pop("output")
    lock_dir = kwargs["lock_dir"]
    kwargs.pop("lock_dir")
    job_dir_name = kwargs["job_dir_name"]
    kwargs.pop("job_dir_name")
    n_cpus = kwargs["n_cpus"]
    kwargs.pop("n_cpus")
    memory_per_cpu = kwargs.pop("mem_per_cpu", 4000)
    max_n_jobs = kwargs["max_n_jobs"]
    kwargs.pop("max_n_jobs")
    id_job = kwargs["id_job"]
    kwargs.pop("id_job")
    time_table_file = kwargs["time_table_file"]
    kwargs.pop("time_table_file")
    margin_time_table = kwargs["margin_time_table"]
    kwargs.pop("margin_time_table")
    wandb_project = kwargs.pop("wandb_project", None)
    wandb_offline = kwargs.pop("wandb_offline", False)
    partitions = kwargs.pop("partitions", "batch,debug")
    
    # First attempt of creating script to check number of jobs < max n jobs
    d_time = None
    if time_table_file is not None:
        if os.path.isfile(time_table_file):
            with open(time_table_file, "r") as time_table_file_stream:
                try:
                    d_time = json.load(time_table_file_stream)
                except BaseException as e:
                    warnings.warn(f"Time table could not be opened despite existing file, details: {e}")
        else:
            warnings.warn(f"Time table does not exist, default time limit will be applied")
    def write_job(kwarg, i, dtime=None):
        time_limits_tuple = time_limits
        if d_time is None:
            time_limits_tuple = list(time_limits)
        else:
            keys, values = zip(*kwarg.items())
            permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
            max_time_limit = 0
            arg_max_time_limit = None
            for kwarg_item in permutations_dicts:
                key_time = "#".join(str(k) for k in (
                    kwarg_item["env"],
                    kwarg_item["Delta_M"],
                    kwarg_item["Delta_P"],
                    kwarg_item["Delta_P_prime"],
                    kwarg_item["policy"], 
                    kwarg_item["K"],
                    T, 
                    kwarg_item["repartition_keys_optimiser"],
                    kwarg_item["remove_peak_costs_policy_flags"] == "remove-peak-costs-policy",
                    kwarg_item["disable_net_consumption_production_mutex_flags"] == "disable-net-cons-prod-mutex",
                    kwarg_item["optimal_action_population_size"], 
                    n_cpus,
                    kwarg_item["remove_peak_costs_flags"] == "remove-peak-costs",
                    kwarg_item["n_samples_policy"]))
                if key_time not in d_time.keys():
                    warnings.warn(f"Time limits for the configuration {key_time} are not available in the dict. Taking default values")
                    arg_max_time_limit = None
                    break
                max_time_limit = max(d_time[key_time][-1], max_time_limit)
                if max_time_limit == d_time[key_time][-1]:
                    arg_max_time_limit = d_time[key_time]
            if arg_max_time_limit is not None:
                t_averaged_with_margin = datetime.timedelta(seconds=arg_max_time_limit[-1]*margin_time_table)
                arg_max_time_limit = (t_averaged_with_margin.days, t_averaged_with_margin.seconds // 3600, max((t_averaged_with_margin.seconds//60)%60, 0), max(t_averaged_with_margin.seconds, 59), t_averaged_with_margin)
            time_limits_tuple = arg_max_time_limit if arg_max_time_limit is not None else time_limits
        script, _ = get_script({"partition": partitions, "output":output, "ntasks": ntasks, "mem_per_cpu": memory_per_cpu, "num_cpus": n_cpus, 'time_d': time_limits_tuple[0], 'time_h': time_limits_tuple[1], 'time_m': max(time_limits_tuple[2], 1), "time_s": time_limits_tuple[3]}, kwarg)
        args = [
            (f"--{k.replace('_', '-')} ${'{' + k + '}'}" if "flags" not in k else f"--${'{' + k + '}'}") for k, v in kwarg.items()
        ]
        command = f"python -m experiment_scripts.mpc.mpc_experiment {' '.join(args)} --use-wandb --root-dir {root_dir} --lock-dir {lock_dir} --n-cpus {n_cpus} --wandb-project {wandb_project}{' --time-policy' if time_policy else ''}{' --wandb-offline' if wandb_offline else ''}{' --solver-verbose' if solver_verbose else ''}"
        if T is not None:
            command += f" --T {T}"
        script += f"\nexport WANDB_API_KEY=\"582a8a23121b546065d9a736970b13c4fab394e1\""
        script += f"\nexport OMP_NUM_THREADS={n_cpus}"
        script += f"\nexport MKL_NUM_THREADS={n_cpus}"
        script += f"\necho LAUNCHING PYTHON COMMAND {command}"
        script += f"\nmkdir -p {root_dir}/{job_dir_name}/old_jobs && mkdir -p {root_dir}/{job_dir_name}/running_jobs && {command}"
        os.makedirs(f"{root_dir}/{job_dir_name}", exist_ok=True)
        with open(f"{root_dir}/{job_dir_name}/rec_submit_mpc_{id_job}_{i}.sh", "w") as job_file:
            job_file.write(script)
        return f"{root_dir}/{job_dir_name}/rec_submit_mpc_{id_job}_{i}.sh"

    n_jobs = np.product([len(v) for v in kwargs.values()])
    if n_jobs <= max_n_jobs:
        write_job(kwargs, 1, dtime=d_time)
    else:
        lst_kwargs = break_sweep_dict(kwargs, max_size=max_n_jobs)
        i = 0
        for kwarg in sorted(lst_kwargs, key=lambda kwarg: (max(kwarg["K"]), max(kwarg["net_cons_prod_mutex_before"]))):
            write_job(kwarg, i+1, dtime=d_time)
            i += 1

if __name__ == '__main__':
    run_experiment()

