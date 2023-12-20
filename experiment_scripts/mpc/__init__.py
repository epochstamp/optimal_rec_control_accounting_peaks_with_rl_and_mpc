from .cplex_params_zoo import cplex_params_dict
from .gurobi_params_zoo import gurobi_params_dict

def generic_none(model):
    pass

solvers = [
    "cplex",
    "gurobi",
    "mosek"
]



solver_params_dict = {
    **{
        "cplex_"+key: v for key, v in cplex_params_dict.items()
    },
    **{
        "gurobi_"+key: v for key, v in gurobi_params_dict.items()
    }
}

cplex_params_dict["none"] = generic_none
gurobi_params_dict["none"] = generic_none
solver_params_dict["none"] = generic_none
