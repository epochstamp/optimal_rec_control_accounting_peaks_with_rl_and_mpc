def create_default_metering_trigger(Delta_M, Delta_P):
    def metering_trigger(tau_m, tau_p):
        return (tau_p is None and tau_m == Delta_M) or tau_p == Delta_P
    return metering_trigger
    
def create_default_peak_trigger(Delta_M, Delta_P):
    def peak_trigger(tau_m, tau_p):
        return tau_p == Delta_P
    return peak_trigger

def create_metering_period_metering_trigger(Delta_M, Delta_P):
    def metering_trigger(tau_m, tau_p):
        return tau_m == Delta_M
    return metering_trigger
    
def create_peak_period_peak_trigger(Delta_M, Delta_P):
    def peak_trigger(tau_m, tau_p):
        return tau_p == Delta_P
    return peak_trigger

def create_everytime_trigger(Delta_M, Delta_P):
    def trigger(tau_m, tau_p):
        return True
    return trigger

def create_peak_period_trigger_th(d = 1.0):
    def create_peak_period_trigger(Delta_M, Delta_P):
        def peak_trigger(tau_m, tau_p):
            return tau_m*tau_p % ((Delta_M*Delta_P)/d) == 0
        return peak_trigger
    return create_peak_period_trigger


metering_period_trigger_global_bill_functions = {
    "default": create_default_metering_trigger,
    "metering_period_trigger": create_metering_period_metering_trigger,
    "everytime_trigger": create_everytime_trigger
}

peak_period_trigger_global_bill_functions = {
    "default": create_default_peak_trigger,
    "peak_period_trigger": create_peak_period_peak_trigger,
    "metering_period_trigger": create_metering_period_metering_trigger,
    "everytime_trigger": create_everytime_trigger
}

period_trigger_global_bill_functions = {
    "peak_period_2nd_trigger": create_peak_period_trigger_th(d=2.0),
    "peak_period_4th_trigger": create_peak_period_trigger_th(d=4.0),
    "peak_period_8th_trigger": create_peak_period_trigger_th(d=8.0),
    "peak_period_16th_trigger": create_peak_period_trigger_th(d=16.0),
    "peak_period_32th_trigger": create_peak_period_trigger_th(d=32.0)

}

def create_triggers_global_bill_function(rec_env, id_metering_period_trigger=None, id_peak_period_trigger=None):
    metering_period_trigger = None
    peak_period_trigger = None
    if id_metering_period_trigger is not None:
        metering_period_trigger = metering_period_trigger_global_bill_functions[id_metering_period_trigger](rec_env.Delta_M, rec_env.Delta_P)
    if id_peak_period_trigger is not None:
        peak_period_trigger = peak_period_trigger_global_bill_functions[id_peak_period_trigger](rec_env.Delta_M, rec_env.Delta_P)
    return metering_period_trigger, peak_period_trigger
