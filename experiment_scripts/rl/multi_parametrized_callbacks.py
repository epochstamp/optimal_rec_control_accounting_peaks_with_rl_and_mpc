from ray.rllib.algorithms.callbacks import MultiCallbacks
from ray.rllib.utils.annotations import override, OverrideToImplementCustomLogic, OverrideToImplementCustomLogic_CallToSuperRecommended

def transform_callback_elem(cb):
    if type(cb) not in (tuple, list):
        return (cb, [], dict())
    elif len(cb) == 1:
        return (cb[0], [], dict())
    elif len(cb) == 2:
        return (cb[0], [(cb[1] if type(cb[1]) in (tuple, list) else [])], (cb[1] if type(cb[1]) == dict else dict()))
    else:
        return cb

def create_callback_creator(cb):    
    def create_callback_from_tuple():
        return cb[0](*cb[1], **cb[2])
    return create_callback_from_tuple
    

class MultiParametrizedCallbacks(MultiCallbacks):
    

    def __init__(self, callback_class_list):
        
        super().__init__(callback_class_list)
        self._callback_class_list = [
            transform_callback_elem(cb) for cb in self._callback_class_list
        ]
        self._callback_class_list = [
            create_callback_creator(cb) for cb in self._callback_class_list
        ]