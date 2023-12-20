
import os
import click
import numpy as np
from envs import create_env_fcts
from experiment_scripts.rl.action_distributions import action_distribution_zoo
from slurm_utils.break_sweep_dict import break_sweep_dict
import sys
from experiment_scripts.rl.rl_envs_zoo import rl_envs
from experiment_scripts.rl.models_zoo import models_zoo
from experiment_scripts.rl.env_wrappers_zoo import env_wrapper_sequences
from experiment_scripts.rl.rl_experiment import validate_space_converter

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
    'num_cpus': 4,
    'num_gpus': 0,
    'time_d': 0, 'time_h': 10, 'time_m': 0, 'time_s': 0,
    'reserve': 'nqit',
    'job_name': 'rl_experiment',
    'output': 'rl_outputs/output_%A_%a.txt',
    "ntasks": 1,
    "mem_per_cpu": 4000,
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
#SBATCH --gres="gpu:{num_gpus}"

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

    return TEMPLATE.format(**subs)


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

def validate_space_converter_lst(c, p, v):
    for val in v:
        validate_space_converter(c, p, val)
    return v

@click.command()
@click.option('--env', "env", type=click.Choice(list(create_env_fcts.keys())), help='Environment to launch for training', required=True)
@click.option('--env-wrappers', "env_wrappers", type=click.Choice(list(env_wrapper_sequences.keys())), help='Reduce training environment time horizon (useful for sampling different exogenous variables starts). Multiple wrappers possible, sep by # character', default=None, multiple=True)
@click.option('--env-eval', "env_eval", type=click.Choice(list(create_env_fcts.keys())), help='Environment to launch for testing (default : same as training)', default=None, callback=lambda c, p, v: v if v is not None else c.params['env'])
@click.option('--env-valid', "env_valid", type=click.Choice(list(create_env_fcts.keys())), help='Environment to launch for validaation (default : same as training)', default=None, callback=lambda c, p, v: v if v is not None else c.params['env'])
@click.option('--rl-env', "rl_env", type=click.Choice(list(rl_envs.keys())), help='RL env configuration for training', default=["rl"], multiple=True)
@click.option('--rl-env-eval', "rl_env_eval", type=click.Choice(list(rl_envs.keys())), help='RL env configuration for eval (default : same as training)', default="rl")
@click.option('--Delta-M', "Delta_M", type=int, default=[2], help='Delta_M.', multiple=True)
@click.option('--Delta-P', "Delta_P", type=int, default=[1], help='Delta_P.', multiple=True)
@click.option('--Delta-P-prime', "Delta_P_prime", type=int, default=[0], help='Delta_P_prime.', multiple=True)
@click.option('--remove-current-peak-costs-flags', "remove_current_peak_costs_flags", type=click.Choice(["remove-current-peak-costs", "no-remove-current-peak-costs"]), default=["no-remove-current-peak-costs"], help='Whether current peak costs are removed.', multiple=True)
@click.option('--remove-historical-peak-costs-flags', "remove_historical_peak_costs_flags", type=click.Choice(["remove-historical-peak-costs", "no-remove-historical-peak-costs"]), default=["remove-historical-peak-costs"], help='Whether historical peak costs are removed.', multiple=True)
@click.option('--multiprice-flags', "multiprice_flags", type=click.Choice(["multiprice", "no-multiprice"]), default=["multiprice", "no-multiprice"], help='Whether (buying) are changing per metering period.', multiple=True)
@click.option('--random-seed', "random_seed", type=int, default=1, help='Random seed.')
@click.option('--nb-random-seed', "nb_random_seeds", type=int, default=15, help='Random seed.')
@click.option("--root-dir", "root_dir", default=os.path.expanduser('~'), help="Root directory")
@click.option('--n-gpus', "n_gpus", type=int, help='Number of gpus', default=0)
@click.option('--n-cpus', "n_cpus", type=int, help='Number of cpus', default=1)
@click.option('--n-cpus-extras', "n_cpus_extras", type=int, help='Number of cpus', default=0)
@click.option('--mem-per-cpu', "mem_per_cpu", type=int, help='Memory per CPU', default=4096)
@click.option('--partitions', "partitions", type=str, help='Partitions (separated by commas)', default="batch,debug")
@click.option('--max-n-jobs', "max_n_jobs", type=int, help='Maximum number of jobs per script', default=sys.maxsize)
@click.option('--id-job', "id_job", type=str, help='ID suffix of the job', default="")
@click.option('--job-dir-name', "job_dir_name", type=str, help='Jobs directory', default="jobs")
@click.option('--time-limits', "time_limits", type=int, nargs=3, help='Time limit (day hour minuts)', default=[0, 5, 0])
@click.option("--wandb-project", "wandb_project", default="rlstudy", help="Wandb project name")
@click.option("--wandb-offline", "wandb_offline", is_flag=True, help="Whether wandb is put offline")
@click.option("--gymnasium-wrap", "gymnasium_wrap", is_flag=True, help="Whether Gym environment is wrapped on Gymnasium")
@click.option("--time-iter", "time_iter", is_flag=True, help="Whether to display time per iteration")
@click.option('--n-iters', "n_iters", type=int, help='Number of RL iterations', default=150)
@click.option('--evaluation-interval', "evaluation_interval", type=int, help='RL eval periodicity', default=1)
@click.option('--space-converter', "space_converter", type=str, help='Space converter (can use several with # separator)', default=["no_converter"], callback=validate_space_converter_lst, multiple=True)
#hyperparameters
@click.option("--mean-std-filter-mode", "mean_std_filter_mode", type=click.Choice(["no_filter", "only_obs", "obs_and_rew", "obs_optim", "rew_optim", "obs_and_rew_optim", "obs_multi_optim", "rew_multi_optim", "obs_and_rew_multi_optim"]), help="Choose whether observation and/or is zscored by running mean/std of current/optimal trajectory", default=["no_filter", "only_obs", "obs_and_rew", "obs_optim", "rew_optim", "obs_and_rew_optim", "obs_multi_optim", "rew_multi_optim", "obs_and_rew_multi_optim"], multiple=True)
@click.option('--model-config', "model_config", type=click.Choice(models_zoo.keys()), default=list(models_zoo.keys()), help="Model config available from models zoo (see experiment_scripts/rl/models_zoo.py)", multiple=True)
@click.option('--gamma', "gamma", type=float, help='Discount factor gamma', default=[0.99], multiple=True)
@click.option('--gamma-policy', "gamma_policy", type=str, help='Discount factor gamma for RL (either single value or three values separated by #)', default=["0.0"], multiple=True)
@click.option('--lambda-gae', "lambda_gae", type=float, help='GAE Lambda value', default=[1.0], multiple=True)
@click.option('--vf-coeff', "vf_coeff", type=float, help='Value function coeff', default=[1.0], multiple=True)
@click.option('--entropy-coeff', "entropy_coeff", type=float, help='Entropy coeff', default=[0.0], multiple=True)
@click.option('--kl-coeff', "kl_coeff", type=float, help='KL coeff', default=[0.2], multiple=True)
@click.option('--kl-target', "kl_target", type=float, help='KL coeff target', default=[0.01], multiple=True)
@click.option('--clip-param', "clip_param", type=str, help='PPO clip param (single value or three values separated by # for schedule)', default=["0.3"], multiple=True)
@click.option('--vf-clip-param', "vf_clip_param", type=float, help='PPO clip param for VF', default=[10.0], multiple=True)
@click.option('--lr', "lr", type=str, help='Learning rate', default=["5e-05"], multiple=True)
@click.option('--bs', "bs", type=int, help='Batch size', multiple=True, default=[64])
@click.option("--gc", "gc", type=float, help="Gradient clipping value (0 for default clipping per algo)", multiple=True, default=[10])
@click.option('--ne', "ne", type=int, help='Number of episodes per training iter', multiple=True, default=[64])
@click.option('--ne-eval', "ne_eval", type=int, help='Number of episodes per evaluation iter', multiple=True, default=[1])
@click.option('--n-sgds', "n_sgds", type=int, help='Number of SGD passes', multiple=True, default=[10])
@click.option('--action-weights-divider', "action_weights_divider", type=float, help='Divider of the weights of the output action layer', default=[1.0], multiple=True)
@click.option("--action-dist", "action_dist", type=click.Choice(list(action_distribution_zoo.keys())), default=click.Choice(list(action_distribution_zoo.keys())), help="Choice of action distribution for policy", multiple=True)
@click.option('--sha-folders', "sha_folders", is_flag=True, help='Whether to rename results folders on sha256 (parameters are registered in a separate json).')
@click.option('--tar-gz-results', "tar_gz_results", is_flag=True, help='Whether to compress results files on a single archive ((except parameters files)).')
def run_experiment(**kwargs):
    
    """
    script = get_script({}, {'a':range(10), 'b':range(10), 'c':range(10)},
                        param_order=['c','a','b'])
    print(script)
    """
    np.random.seed(kwargs["random_seed"])
    random_seeds = np.random.randint(low=0, high=1000000, size=kwargs["nb_random_seeds"])
    kwargs.pop("random_seed")
    kwargs.pop("nb_random_seeds")
    root_dir = kwargs["root_dir"]
    kwargs.pop("root_dir")
    kwargs["random_seed"] = list(random_seeds)
    n_cpus = kwargs["n_cpus"]
    n_cpus_extras = kwargs.pop("n_cpus_extras", 0)
    n_gpus = kwargs["n_gpus"]
    kwargs.pop("n_cpus")
    kwargs.pop("n_gpus")
    n_iterations = kwargs["n_iters"]
    kwargs.pop("n_iters")
    kwargs["env"] = [kwargs["env"]]
    kwargs["env_eval"] = [kwargs["env_eval"]]
    kwargs["env_valid"] = [kwargs["env_valid"]]
    kwargs["rl_env_eval"] = [kwargs["rl_env_eval"]]
    wandb_project = kwargs.pop("wandb_project")
    gymnasium_wrap = kwargs.pop("gymnasium_wrap", False)
    wandb_offline = kwargs.pop("wandb_offline", False)
    memory_per_cpu = kwargs.pop("mem_per_cpu", 4096)
    time_iter = kwargs.pop("time_iter", False)
    partitions = kwargs.pop("partitions", "batch,debug")
    if kwargs["env_wrappers"] is None or len(kwargs["env_wrappers"]) == 0:
        kwargs.pop("env_wrappers")
    sha_folders = kwargs.pop("sha_folders")
    tar_gz_results = kwargs.pop("tar_gz_results")

    job_dir = kwargs["job_dir_name"]
    kwargs.pop("job_dir_name")
    max_n_jobs = kwargs["max_n_jobs"]
    kwargs.pop("max_n_jobs")
    id_job = kwargs["id_job"]
    kwargs.pop("id_job")
    time_limits = kwargs["time_limits"]
    kwargs.pop("time_limits")
    evaluation_interval = kwargs.pop("evaluation_interval")
    def write_job(kwarg, i):
        script = get_script({"partition": partitions, "mem_per_cpu": memory_per_cpu, "num_cpus": n_cpus+n_cpus_extras, 'time_d': time_limits[0], 'time_h': time_limits[1], 'time_m': time_limits[2]}, kwarg)
        args = [
            (f"--{k.replace('_', '-')} ${'{' + k + '}'}" if "flags" not in k else f"--${'{' + k + '}'}") for k, v in kwarg.items()
        ]
        command = f"python -m experiment_scripts.rl.rl_experiment {' '.join(args)} --evaluation-interval {evaluation_interval} --use-wandb --wandb-project {wandb_project} --root-dir {root_dir} --n-cpus {n_cpus} --n-iters {n_iterations}{' --gymnasium-wrap' if gymnasium_wrap else ''}{' --wandb-offline' if wandb_offline else ''}{' --time-iter' if time_iter else ''}{' --sha-folders' if sha_folders else ''}{' --tar-gz-results' if tar_gz_results else ''}"
        script += f"\nexport WANDB_API_KEY=\"582a8a23121b546065d9a736970b13c4fab394e1\""
        script += f"\nexport OMP_NUM_THREADS={n_cpus}"
        script += f"\nexport MKL_NUM_THREADS={n_cpus}"
        script += f"\nexport OPENBLAS_NUM_THREADS={n_cpus}"
        script += f"\nexport VECLIB_MAXIMUM_THREADS={n_cpus}"
        script += f"\nexport NUMEXPR_NUM_THREADS={n_cpus}"
        script += f"\necho LAUNCHING PYTHON COMMAND {command}"
        script += f"\n{command} && mkdir -p {root_dir}/{job_dir}/running_jobs"
        os.makedirs(f"{root_dir}/{job_dir}", exist_ok=True)
        with open(f"{root_dir}/{job_dir}/rec_submit_rl_{id_job}_{i}.sh", "w") as job_file:
            job_file.write(script)
        return f"{root_dir}/{job_dir}/rec_submit_rl_{id_job}_{i}.sh"

    n_jobs = np.product([len(v) for v in kwargs.values()])
    if n_jobs <= max_n_jobs:
        write_job(kwargs, 1)
    else:
        lst_kwargs = break_sweep_dict(kwargs, max_size=max_n_jobs)
        i = 0
        for kwarg in sorted(lst_kwargs, key=lambda kwarg: (sum(kwarg["ne"]), sum(kwargs["n_sgds"]), sum(kwargs["bs"]))):
            write_job(kwarg, i+1)
            i += 1

    

if __name__ == '__main__':
    run_experiment()

