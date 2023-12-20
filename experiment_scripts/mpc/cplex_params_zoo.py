def none(model):
    return model

def create_mip_gap(mipgap: float):
    def mipgap_fct(model):
        model.parameters.mip.tolerances.mipgap = mipgap
        return model
    return mipgap_fct

def create_abs_mip_gap(absmipgap: float):
    def mipgap_fct(model):
        model.parameters.mip.tolerances.absmipgap = absmipgap
        return model
    return mipgap_fct

def switch_off_presolve(model):
    model.parameters.preprocessing.presolve = 0
    return model

def strong_branching(model):
    model.parameters.mip.strategy.variableselect = 3
    return model

def very_aggressive_probe(model):
    model.parameters.mip.strategy.probe = 3
    return model

cplex_params_dict = {
    "mip_gap_01": create_mip_gap(1e-1),
    "mip_gap_001": create_mip_gap(1e-2),
    "mip_gap_0001": create_mip_gap(1e-3),
    "mip_gap_00001": create_mip_gap(1e-4),
    "mip_gap_000001": create_mip_gap(1e-5),
    "mip_gap_1e_10": create_mip_gap(1e-10),
    "mip_gap_00005": create_mip_gap(5e-4),
    "mip_gap_00006": create_mip_gap(6e-4),
    "abs_mip_gap_5": create_abs_mip_gap(5.0),
    "no_presolve": switch_off_presolve,
    "strong_branching": strong_branching,
    "very_aggressive_probe": very_aggressive_probe
}