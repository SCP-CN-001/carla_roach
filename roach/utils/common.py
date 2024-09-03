import os
from importlib import import_module

WORKSPACE = os.path.join(os.path.dirname(__file__), "../..")


def load_entry_point(name):
    mod_name, attr_name = name.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def get_absolute_path(path_):
    if not os.path.isabs(path_):
        return os.path.join(WORKSPACE, path_)
    else:
        return path_
