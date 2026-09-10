from kalfa.std.common.log import logger_for


logger = logger_for("after.plots")


def seaborn_module(what):
    try:
        import seaborn
    except ImportError:
        logger.warning(f"{what}: seaborn is not installed, so the plot is skipped; pip install seaborn for it")
        return None
    return seaborn


def sampled(table, sample, seed=0):
    if sample and len(table) > int(sample):
        return table.sample(int(sample), random_state=int(seed))
    return table
