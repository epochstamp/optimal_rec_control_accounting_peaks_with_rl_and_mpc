
def create_mip_gap(mipgap: float):
    def mipgap_fct(model):
        model.Params.mip.tolerances.mipgap = mipgap
        return model
    return mipgap_fct

gurobi_params_dict = {
    
}