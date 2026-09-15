from kalfa.std.common.rng import derived_seed


def derived(seed, name, index):
    return None if seed is None else derived_seed(seed, name)


def indexed(seed, name, index):
    return None if seed is None else derived_seed(seed, index)


def global_stream(seed, name, index):
    return None
