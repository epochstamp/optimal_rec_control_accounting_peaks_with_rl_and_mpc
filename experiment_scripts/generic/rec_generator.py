import base64
import dataclasses
import hashlib
import json
import lzma
import pathlib
import shutil
import struct
import tarfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from types import NoneType
from typing import Union
import click

import numpy as np
import numpy.typing as npt
import os

def my_hash(hashobj, entry):
    if isinstance(entry, np.ndarray):
        hashobj.update(entry.data.tobytes())
    elif isinstance(entry, RECMember):
        entry.hash(hashobj)
    elif isinstance(entry, tuple):
        my_hash(hashobj, len(entry))
        for e in entry:
            my_hash(hashobj, e)
    elif isinstance(entry, float):
        hashobj.update(struct.pack('!d', entry))
    elif isinstance(entry, int):
        hashobj.update(entry.to_bytes(length=(max(entry.bit_length(), 1) + 7) // 8, byteorder='little'))
    elif isinstance(entry, NoneType):
        hashobj.update(b'NoneType')
    else:
        raise Exception("Unknown type", type(entry))

def summarize(dataclass_obj, sub_path_prefix):
    out = {}
    fields = dataclasses.fields(dataclass_obj)
    for f in fields:
        entry = dataclass_obj.__getattribute__(f.name)
        if f.name == "members":
            out["members"] = [
                summarize(member, sub_path_prefix / "members" / f"{idx}")
                for idx, member in enumerate(entry)
            ]
        elif isinstance(entry, np.ndarray):
            h = hashlib.new('sha256')
            h.update(entry.data.tobytes())
            out[f.name] = {"path": str(sub_path_prefix / f"{f.name}.csv"), "hash": str(h.hexdigest())}
            #np.savetxt(path_prefix / sub_path_prefix / f"{f}.csv", entry, delimiter=",", fmt="%1.8f")
        else:
            out[f.name] = entry
    return out


def register_files(dataclass_obj, sub_path_prefix):
    fields = dataclasses.fields(dataclass_obj)
    for f in fields:
        entry = dataclass_obj.__getattribute__(f.name)
        if f.name == "members":
            (sub_path_prefix / "members").mkdir()
            for idx, member in enumerate(entry):
                (sub_path_prefix / "members" / f"{idx}").mkdir()
                register_files(member, sub_path_prefix / "members" / f"{idx}")
        elif isinstance(entry, np.ndarray):
            np.savetxt(sub_path_prefix / f"{f.name}.csv", entry, delimiter=",", fmt="%1.8f")


@dataclass(frozen=True)
class RECMember:
    name: str
    pv_maximum_capacity: int  # kW
    pv_minimum_capacity: int  # kW
    battery_maximum_capacity: int  # kWh
    battery_minimum_capacity: int  # kWh
    battery_cost_y1: int   # €/kWh
    battery_cost_y15: int  # €/kWh
    demand: npt.NDArray[np.float64] # kWh each hour
    base_injection: npt.NDArray[np.float64] # kWh each hour
    grid_import_price: npt.NDArray[np.float64]
    grid_export_price: npt.NDArray[np.float64]
    rec_energy_surcharge: npt.NDArray[np.float64] #€/kWh
    peak_month_price: float
    peak_month_price_deg: float
    peak_hist_price: float
    peak_hist_price_deg: float
    peak_demand_1st_to_11th_coef: float = 1.0
    


class OptimizationType(str, Enum):
    RAW_COSTS = 'RAW_COSTS'
    CO2_COSTS = 'CO2_COSTS'
    RAW_AND_CO2_COSTS = 'RAW_AND_CO2_COSTS'
    ENVIRONMENTAL_IMPACT = 'ENVIRONMENTAL_IMPACT'


@dataclass(frozen=True)
class REC:
    members: tuple[RECMember]
    lifetime: int = 30             # years
    interest_rate: float = 0.06    # ratio
    discount_rate: float = 0.06    # ratio
    CO2_price: float = 52.         # €/ton
    pv_CO2_emission: float = 89.6  # grCO2eq/kWh
    pv_CO2_price: float = CO2_price  # EUR/kWh
    pv_price_multiply: float = 1.0
    battery_CO2_emission: float = 85702  # grCO2eq/kWh installed
    battery_CO2_price: float = CO2_price  # EUR/kWh
    grid_environmental_impact: npt.NDArray[np.float64] = field(default_factory=lambda: np.loadtxt("data/impact_grid_2019.csv"))
    allow_rec_exchanges: bool = True
    allow_grey_storage: bool = True
    environmental_improvement_required: Union[float, None] = None  # percentage (None: no constraint, 100: no improvement, 90: 10% improvement, 110: 10% worsening)
    max_co2_emissions: float = None
    max_costs: float = None
    time_horizon: int = 8760
    opti_type: OptimizationType = OptimizationType.RAW_COSTS
    model_ver: int = 8


def fix_stupid_file_format(file):
    return np.array([np.float64(x) for x in open(file).read().replace(";", "\n").replace(",", "\n").split("\n") if x != ""])


def gen_instance(instance_folder: Path, rec: REC, expename=None, **kwargs):
    summary = summarize(rec, Path("data"))
    if expename is not None:
        summary["expename"] = expename
    for a,b in kwargs.items():
        summary[a] = b

    summary_json = json.dumps(summary, indent=4)
    h = hashlib.new('sha256')
    h.update(summary_json.encode())
    name = h.hexdigest()

    if (instance_folder / f"{name}.instance.tar.xz").exists():
        return name

    if (instance_folder / name).exists():
        shutil.rmtree(instance_folder/name)

    (instance_folder / name).mkdir()
    (instance_folder / name / "data").mkdir()
    (instance_folder / name / "model").mkdir()
    


    register_files(rec, instance_folder / name / "data")

    thispath = Path(kwargs.get("base_path", str(pathlib.Path.cwd())))
    base_files = [
        "model/member.gboml",
        "model/meter.gboml",
        "model/pv_panel.gboml",
        "model/battery.gboml",
        "model/degressive_peak.gboml",
        "data/battery_caracteristics_MegaPack.csv",
        "data/cost_pv_technology.csv",
        "data/pv_cap_lim.csv",
        "data/rate_power_generation.csv",
        "data/pv_caracteristics_single_si.csv"
    ]
    for n in base_files:
        shutil.copy(thispath / n, instance_folder / name / n)
    with open(instance_folder / name / "main.gboml", "w") as main_gboml:
        main_gboml.write(f"""
        #TIMEHORIZON
            T = {rec.time_horizon};
        #GLOBAL
            lifetime = {rec.lifetime};
            interest_rate = {rec.interest_rate};
            discount_rate = {rec.discount_rate};
            
            wacc = interest_rate / (1 - (1 + interest_rate)**(-lifetime));
            discount_factor = 1; //(1 - (1 + discount_rate)**(-lifetime - 1)) / (1 - (1 + discount_rate)**(-1));
            
            CO2_price = {rec.CO2_price} * (1/1000000);
            pv_CO2_emission = {rec.pv_CO2_emission};
            pv_CO2_price = {rec.pv_CO2_price} * (1/1000000);
            pv_price_multiply = {rec.pv_price_multiply};
            battery_CO2_emission = {rec.battery_CO2_emission};
            battery_CO2_price = {rec.battery_CO2_price} * (1/1000000);
            
            grid_environmental_impact = import "data/grid_environmental_impact.csv";
            
            allow_grey_storage = {1 if rec.allow_grey_storage else 0};
        """)

        for idx, member in enumerate(rec.members):
            main_gboml.write(f"""
                #NODE MEMBER_{idx} = import MEMBER from "model/member.gboml" with
                    pv_minimum_capacity = {member.pv_minimum_capacity};
                    pv_maximum_capacity = {member.pv_maximum_capacity};
                    battery_minimum_capacity = {member.battery_minimum_capacity};
                    battery_maximum_capacity = {member.battery_maximum_capacity};
                    battery_cost_y1 = {member.battery_cost_y1};
                    battery_cost_y15 = {member.battery_cost_y15};
                    
                    rec_energy_surcharge = import "data/members/{idx}/rec_energy_surcharge.csv";
                    peak_month_price = {member.peak_month_price};
                    peak_month_price_deg = {member.peak_month_price_deg};
                    peak_hist_price = {member.peak_hist_price};
                    peak_hist_price_deg = {member.peak_hist_price_deg};
                    peak_demand_1st_to_11th_coef = {member.peak_demand_1st_to_11th_coef};
                    
                    
                    demand = import "data/members/{idx}/demand.csv";
                    base_injection = import "data/members/{idx}/base_injection.csv";
                    grid_import_price = import "data/members/{idx}/grid_import_price.csv";
                    grid_export_price = import "data/members/{idx}/grid_export_price.csv";
            """)

        if rec.allow_rec_exchanges:
            main_gboml.write(f"""
            #HYPEREDGE REC
                #CONSTRAINTS
                    {" + ".join(f"MEMBER_{idx}.rec_import[t]"  for idx in range(len(rec.members)))} == {" + ".join(f"MEMBER_{idx}.rec_export[t]"  for idx in range(len(rec.members)))};
            """)
        else:
            main_gboml.write(f"""
            #HYPEREDGE REC
                #CONSTRAINTS
                    {"; ".join(f"MEMBER_{idx}.rec_import[t] == 0" for idx in range(len(rec.members)))};
                    {"; ".join(f"MEMBER_{idx}.rec_export[t] == 0" for idx in range(len(rec.members)))};
            """)

        if rec.environmental_improvement_required is not None:
            main_gboml.write(f"""
            #HYPEREDGE ENVIRONMENTAL_CONSTRAINTS
                #PARAMETERS
                    constraint = {rec.environmental_improvement_required/100.}; // [unitless, ratio]
                #CONSTRAINTS
                    {" + ".join(f"MEMBER_{idx}.environmental_impact" for idx in range(len(rec.members)))} <= constraint * ({" + ".join(f"MEMBER_{idx}.environmental_impact_demand"  for idx in range(len(rec.members)))});
            """)

        if rec.max_co2_emissions is not None:
            main_gboml.write(f"""
            #HYPEREDGE ENVIRONMENTAL_CONSTRAINTS_MAX
                #PARAMETERS
                    constraint = {rec.max_co2_emissions};
                #CONSTRAINTS
                    {" + ".join(f"MEMBER_{idx}.environmental_impact" for idx in range(len(rec.members)))} <= constraint;
            """)

        if rec.max_costs is not None:
            main_gboml.write(f"""
            #HYPEREDGE COSTS_MAX
                #PARAMETERS
                    constraint = {rec.max_costs};
                #CONSTRAINTS
                    {" + ".join(f"MEMBER_{idx}.total_raw_costs" for idx in range(len(rec.members)))} <= constraint;
            """)

    with open(instance_folder / name / "instance.json", 'w') as f:
        f.write(summary_json)

    with open(instance_folder / name / "parameters.json", 'w') as f:
        args = []
        match rec.opti_type:
            case OptimizationType.RAW_COSTS:
                args.append("--disable-obj")
                args.append("*co2*")
            case OptimizationType.CO2_COSTS:
                args.append("--disable-obj")
                args.append("*raw_costs*")
                args.append("--disable-obj")
                args.append("*co2_raw*")
            case OptimizationType.RAW_AND_CO2_COSTS:
                args.append("--disable-obj")
                args.append("*co2_raw*")
            case OptimizationType.ENVIRONMENTAL_IMPACT:
                args.append("--disable-obj")
                args.append("*costs*")
        json.dump(args, f)

    xz_file = lzma.LZMAFile(instance_folder / f"{name}.instance.tar.xz", mode='w')

    with tarfile.open(mode='w', fileobj=xz_file) as tar_xz_file:
        tar_xz_file.add(instance_folder / name / "", arcname=name)

    xz_file.close()
    shutil.rmtree(instance_folder / name)
    return name

@click.command()
@click.option('--base-path', "base_path", type=click.Choice(["simple", "simple_long", "first", "complete_first"]), help='Base path to look for data', required=True)
def gen_rec():
    pass