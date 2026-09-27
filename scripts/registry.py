import importlib
from collections import OrderedDict

_MODULES = [
    "methods.padim",
    "methods.patchcore",
    "methods.reverse_distillation",
    "methods.efficientad",
    "methods.ganomaly",
]

METHODS = OrderedDict()
MODULE_FILES = OrderedDict()

for _module_name in _MODULES:
    _module = importlib.import_module(_module_name)
    METHODS[_module.SPEC.name] = _module.SPEC
    MODULE_FILES[_module.SPEC.name] = _module.__file__

METHOD_NAMES = list(METHODS)


def select(names):
    wanted = set(names)
    return [spec for name, spec in METHODS.items() if name in wanted]
