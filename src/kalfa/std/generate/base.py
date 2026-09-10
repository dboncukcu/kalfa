def pick_model(models, name):
    if name not in models:
        raise KeyError(f"generate: {name!r} is no model; the models are {sorted(models)}")
    return models[name]
