import hashlib
from pathlib import Path
import wandb
import click
import json
import warnings
import os
import zipfile
import shutil
from tqdm import tqdm

@click.command()
@click.option('--folder-results', "folder_results", help='Results folders', required=True)
@click.option('--dry-run', "dry_run", help='Whether to dry run wandb', is_flag=True)
@click.option('--is-rl/--is-mpc', "is_rl", help='Whether it is about MPC or RL (results keys have different patterns)', is_flag=True)
@click.option('--tar-gz-results', "tar_gz_results", help='Whether results are archived as zip', is_flag=True)
@click.option('--where-equals', "where_equals", help='Conditional fields to select hyperparameters to send to wandb', multiple=True, nargs=2)
def run_experiment(folder_results, dry_run, is_rl, tar_gz_results, where_equals):
    if tar_gz_results:
        rglob_file = "post_to_wandb"
        file_extension = "zip"
    else:
        rglob_file = "post_to_wandb" if not is_rl else "post_to_wandb_0"
        file_extension = "json"
    where_equals_dict = dict()
    for k, v in where_equals:
        if k not in where_equals_dict:
            where_equals_dict[k] = []
        where_equals_dict[k].append(float(v) if str(v).isnumeric() else str(v))
    lst_files = list(Path(f"{folder_results}/").rglob(rglob_file+f".{file_extension}"))
    for name in tqdm(lst_files):
        if not is_rl and os.path.isfile(os.path.dirname(name)+"/post_to_wandb.lockjson"):
            #print(os.path.dirname(name)+"/post_to_wandb.lockjson exists, skip")
            continue
        #Get all wandb file
        wandb_mode = "disabled" if dry_run else "online"
        if is_rl:
            if tar_gz_results:
                with zipfile.ZipFile(name,"a") as tar_results:
                    tar_results.extractall(f"{os.path.dirname(name)}/results/")

            lst_file = list(Path(f"{os.path.dirname(name)}/").rglob("post_to_wandb_*.json"))
            lst_file = sorted(lst_file, key=lambda k: int(Path(k).stem.split("_")[-1]))
            for i, subname in enumerate(lst_file):
                if os.path.isfile(str(subname) + ".lockjson"):
                    continue
                wandb_data_file = open(subname, "r")
                try:
                    wandb_data = json.load(wandb_data_file)
                except BaseException as e:
                    warnings.warn(f"Json file cannot be read. Details: {e}")
                    continue
                wandb_data_file.close()
                
                if i == 0:
                    config = wandb_data["config"]
                    skip = False
                    for k,v in config.items():
                        if k in where_equals_dict.keys() and v not in where_equals_dict[k]:
                            skip=True
                            break
                    if skip:
                        break
                    wandb_project = wandb_data["wandb_project"]
                    
                    if not dry_run:
                        wandb.init(config=config, project=wandb_project, entity="samait", mode=wandb_mode)
                results = wandb_data["results"]
                if not dry_run:
                    wandb.log(results)
            if not dry_run:
                with open(os.path.dirname(name)+"/post_to_wandb.lockjson", "w+") as _:
                    pass
            else:
                pass
            wandb.finish()
            if tar_gz_results:
                shutil.rmtree(f"{os.path.dirname(name)}/results", ignore_errors=True)
        else:

            wandb_data_file = open(name, "r")
            try:
                wandb_data = json.load(wandb_data_file)
            except BaseException as e:
                warnings.warn(f"Json file cannot be read. Details: {e}")
                continue
            wandb_data_file.close()
            #id_wandb = wandb_data["id_wandb"]
            wandb_project = wandb_data["wandb_project"]
            config = wandb_data["config"]
            #group_wandb = wandb_data["group_wandb"]
            config_sorted_keys = sorted(list(config.keys()))
            id_wandb = "/".join([f"{str(k)}={str(config[k])}" for k in config_sorted_keys if k != "K"])
            id_wandb = hashlib.sha256(id_wandb.encode("utf-8")).hexdigest()
            group_wandb = "/".join([f"{str(k)}={str(config[k])}" for k in config_sorted_keys if k not in ("K", "random_seed")])
            group_wandb = hashlib.sha256(id_wandb.encode("utf-8")).hexdigest()
            
            if "step_metric" not in wandb_data:
                splitted_path = str(name).split("/")
                for folder in splitted_path:
                    if "K=" in folder:
                        K = int(folder.split("=")[1])
                        break
            else:
                K = int(wandb_data["step_metric"])
            runs = wandb.Api().runs(path=f'samait/{wandb_project}')
            runs = [run for run in runs if id_wandb in run.id]
            if len(runs) > 0:
                iwandb_unique_run_id = runs[0].id
            else:    
                iwandb_unique_run_id = id_wandb + "_" + str(wandb.util.generate_id())
            
            wandb.init(group=group_wandb, id=iwandb_unique_run_id, config=config, project=wandb_project, entity="samait", resume=("must" if len(runs) > 0 else False), mode=wandb_mode)
            wandb.define_metric("K")
            results = {
                "K": K
            }
            for wandb_result_key_orig, wandb_result_value in wandb_data["results"].items():
                wandb_result_key = " ".join([r.capitalize() for r in wandb_result_key_orig.split("_")])
                wandb.define_metric(wandb_result_key, step_metric="K")
                results[wandb_result_key] = wandb_result_value
                
            wandb.log(results)
            wandb.finish()
            if not dry_run:
                with open(os.path.dirname(name)+"/post_to_wandb.lockjson", "w+") as _:
                    pass
    

                

if __name__ == '__main__':
    run_experiment()