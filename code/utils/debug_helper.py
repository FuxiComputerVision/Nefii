import torch

_debug = False

_global_dict = None
_global_gradiant_dict = None


def set_debug(flag):
    global _debug
    _debug = flag


def get_debug():
    return _debug


def set_value(key, value):
    if not _debug:
        return

    global _global_dict
    if _global_dict is None:
        _global_dict = {}

    _global_dict[key] = value


def get_all():
    try:
        return _global_dict
    except NameError:
        return {}


def create_gradiant_hook(key):
    def extract(g):

        global _global_gradiant_dict
        if _global_gradiant_dict is None:
            _global_gradiant_dict = {}

        _global_gradiant_dict[key] = g

    return extract


def watch_gradiant(variable: torch.Tensor, key: str):
    if not _debug:
        return

    global _global_gradiant_dict
    if _global_gradiant_dict is None:
        _global_gradiant_dict = {}

    if variable.requires_grad:
        variable.register_hook(create_gradiant_hook(key))
    else:
        # remove out date value
        if _global_gradiant_dict.get(key, None) is not None:
            del _global_gradiant_dict[key]


def get_gradiant_all():
    try:
        return _global_gradiant_dict
    except NameError:
        return {}


def watch_value(variable: torch.Tensor, key: str):
    if not _debug:
        return

    value = variable.detach().clone()
    set_value(key, value)
