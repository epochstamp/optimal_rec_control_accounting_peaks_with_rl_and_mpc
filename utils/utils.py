import numpy as np
import collections
from math import floor, log10
from itertools import tee, islice, chain
from functools import reduce

def merge_dicts(ds):
    return {k:v for d in ds for k,v in d.items()}

def get_values(d):
    lv = []
    for v in d.values():
        if isinstance(v, dict):
            lv.extend(get_values(v))
        else:
            lv.append(v)
    return lv

def net_value(v1: float, v2: float) -> float:
    return float(max(v1 - v2, 0))

def epsilonify(v:float, epsilon:float = 1e-4):
    return v if abs(v) >= epsilon else 0.0

def roundify(v:float, decimals:int = 4):
    return float(np.round(v, decimals=decimals))

def flatten(xs):
    for x in xs:
        if (isinstance(x, list) or isinstance(x, tuple)):
            yield from flatten(x)
        else:
            yield x

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def normalize_0_1(x):
    if float(epsilonify(np.max(x)-np.min(x))) == 0.0:
        return x*0
    return (x-np.min(x))/(np.max(x)-np.min(x))

def to_0_1_range(x, min_x, max_x):
    return (x-min_x)/(max_x-min_x+1e-6)

def from_0_1_range(x, min_x, max_x):
    return x * (max_x - min_x) + min_x



def normalize_bounds(x, a, b, min_x, max_x, convert_to_float=True):
    if epsilonify(max_x - min_x) == 0:
        return x*0
    r = ((b-a) * ((x - min_x) / (max_x - min_x))) + a
    if convert_to_float:
        r = float(r)
    return r

def normalize_1_1(x, min_x_raw, max_x_raw):
    min_x = min_x_raw
    max_x = max_x_raw
    if min_x < 0:
        x = float(x + abs(min_x))
        max_x = float(max_x + abs(min_x))
        min_x = float(min_x + abs(min_x))
    return 2 * ((x-min_x)/(max_x-min_x)) - 1

def unnormalize_1_1(x, min_x_raw, max_x_raw):
    min_x = min_x_raw
    max_x = max_x_raw
    if min_x < 0:
        max_x = float(max_x + abs(min_x))
        min_x = float(min_x + abs(min_x))
    v = (max_x - min_x)*((x+1)/2)
    if min_x_raw < 0:
        v -= abs(min_x_raw)
    return v

"""
    def create_soc_charge_dynamics_constraint(Delta_C = 1, charge_efficiency=1):
        def soc_charge_dynamics(s, _, u):
            value = s[("PVB", "soc")] + Delta_C * charge_efficiency * u[("PVB", "charge")]
            if type(value) == float:
                value = round(value, 4)
            return value, 1 + 1e-4, IneqType.LOWER_OR_EQUALS
        return soc_charge_dynamics

    def create_soc_discharge_dynamics_constraint(Delta_C = 1, discharge_efficiency=1):
        def soc_discharge_dynamics(s, _, u):
            value = s[("PVB", "soc")] - Delta_C * (u[("PVB", "discharge")] / discharge_efficiency)
            if type(value) == float:
                value = round(value, 4)
            return value, -1e-4, IneqType.GREATER_OR_EQUALS
        return soc_discharge_dynamics

"""

def create_clip_battery_action(batteries_by_members: dict, Delta_C=1.0):
    def clip_battery_action(state, _, action):
        new_action = dict(action)
        for member, batteries in batteries_by_members.items():
            for battery in batteries:
                battery_offset = action[(member, battery["charge_as"])] - action[(member, battery["discharge_as"])]
                soc = state[(member, battery["soc_as"])]
                min_soc = battery["minsoc"]
                max_soc = battery["maxsoc"]
                absolute_battery_offset = abs(battery_offset)
                if battery_offset < 0:
                    new_action[(member, battery["charge_as"])] = 0
                    discharge_efficiency = battery.get("discharge_efficiency", 1)
                    new_action[(member, battery["discharge_as"])] = round(float(np.clip(absolute_battery_offset, 0, discharge_efficiency*((soc - min_soc)/Delta_C))), 4)
                else:
                    new_action[(member, battery["discharge_as"])] = 0
                    charge_efficiency = battery.get("charge_efficiency", 1)
                    new_action[(member, battery["charge_as"])] = round(float(np.clip(absolute_battery_offset, 0, (max_soc-soc) / (Delta_C*charge_efficiency))), 4)
        return new_action
    return clip_battery_action


def num_zeros(decimal):
    return 1 if decimal == 0 else -floor(log10(abs(decimal)))


def list_with_previous(lst):
    prevs, items = tee(lst, 2)
    prevs = chain([lst[0]], prevs)
    return list(zip(prevs, items))

def locate_elem_in_list_of_chunks(lst_chunks, t, size_chunk_func = lambda chunk: len(chunk)):
    j = 0
    new_t = t
    while new_t - size_chunk_func(lst_chunks[j]) >= 0:
        current_chunk = lst_chunks[j]
        j += 1
        new_t -= size_chunk_func(current_chunk)
    return lst_chunks[j][new_t]

def sliding_window(elements, min_window_size, max_window_size, step_size):
    if len(elements) <= min_window_size:
       return elements 
    for i in np.arange(0, len(elements)- min_window_size + 1, step_size):
        yield elements[i:i+max_window_size]

def split_list_by_number_np(input_list, specific_number, check_end=False, return_indices=False, shift_indices=True):
    # Convertit la liste en un tableau NumPy
    input_array = np.array(input_list)

    # Trouve les indices où le nombre spécifique se trouve dans le tableau
    specific_number_indices = np.where(input_array == specific_number)[0]

    # Ajoute un indice de départ et un indice de fin pour faciliter le découpage
    start_indices = np.concatenate(([0], specific_number_indices + 1))
    end_indices = np.concatenate((specific_number_indices, [len(input_list)]))

    # Trouve les tailles des sous-listes
    subarray_lengths = end_indices - start_indices + 1
    
    # Découpe le tableau d'entrée en sous-tableaux en fonction des indices de début et des tailles
    if return_indices:
        indices = np.arange(0, input_array.shape[0], 1) - (1 if shift_indices else 0)
        subarrays = np.split(indices, np.cumsum(subarray_lengths)[:-1])
    else:
        subarrays = np.split(input_array, np.cumsum(subarray_lengths)[:-1])

    # Convertit les sous-tableaux en listes et les place dans une liste
    result = [subarray.tolist() for subarray in subarrays]
    if result[-1] == []:
        result = result[:-1]
    if (not check_end and input_array[-1] != specific_number):
        return result[:-1]
    else:
        return result
    
def unique_values(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def unique_consecutives_values(seq):
    new_seq = np.asarray(seq)
    seq1 = np.append(seq[1:], 0)
    return new_seq[new_seq != seq1]

def split_list(elements, lengths):
    return reduce(lambda acc, x: acc + [list(islice(elements, sum(lengths[:x]), sum(lengths[:x+1])))], range(len(lengths)), [])

def rindex(lst, value):
    if value not in lst:
        return None
    return len(lst) - lst[::-1].index(value) - 1

def rec_gamma_sequence(gamma, Delta_M=1, Delta_P=1, T=2):
    nb_time_steps_in_peak_period = Delta_M * Delta_P
    nb_peak_periods = (T-1)//nb_time_steps_in_peak_period
    gammas = [(gamma**nb_time_steps_in_peak_period)] * (nb_time_steps_in_peak_period+1)
    if nb_peak_periods > 1:
        for _ in range(nb_peak_periods-1):
            gammas.extend([gammas[-1]*(gamma**nb_time_steps_in_peak_period)]*(nb_time_steps_in_peak_period))
    gammas = np.asarray(gammas, dtype=np.float32)
    return gammas

def find_indices(vector, element, return_last):
    indices = np.where(vector == element)[0]
    if return_last and (indices.size == 0 or vector[-1] != element):
        last_index = np.where(vector == vector[-1])[0]
        if last_index.size:
            indices = np.concatenate([indices, [last_index[-1]]])
    return indices