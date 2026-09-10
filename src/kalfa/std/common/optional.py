import importlib

from kalfa.std.common.log import logger_for


logger = logger_for("after.plots")


def installed(name):
    return importlib.util.find_spec(name) is not None


def load(name, what=None):
    try:
        return importlib.import_module(name)
    except ImportError:
        logger.warning(f"{what or name}: {name} is not installed, so the step is skipped; pip install {name} for it")
        return None
