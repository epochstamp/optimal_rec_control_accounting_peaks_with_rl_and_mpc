import numpy as np
import warnings

def break_sweep_dict(d, max_size=np.inf, warn=False):
    nb_combinations = float(np.prod([len(v) for v in d.values()]))
    if nb_combinations > max_size:
        sorted_d_items = list(sorted([ditem for ditem in d.items() if len(ditem[1]) > 1], key=lambda e: -len(e)))
        if sorted_d_items == []:
            if warn:
                warnings.warn("Can't break up this sweep dict anymore")
            return d
        d_key, d_value = sorted_d_items[0]
        d_value_1 = d_value[len(d_value)//2:]
        d_value_2 = d_value[:len(d_value)//2]
        new_d = dict(d)
        new_d2 = dict(d)
        new_d[d_key] = d_value_2
        new_d2[d_key] = d_value_1
        return break_sweep_dict(new_d, max_size=max_size, warn=warn) + break_sweep_dict(new_d2, max_size=max_size, warn=warn)
    else:
        return [d]