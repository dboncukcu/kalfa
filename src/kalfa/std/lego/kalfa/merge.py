from kalfa.registration import lego


@lego("/lego/kalfa/merge", description="Merge the per set metrics under the prefixes of the sets (train/, val/, test/)")
def merge(parts, prefixes):
    merged = {}
    order = list(prefixes)
    keys = sorted(parts or {}, key=lambda key: (order.index(key.removesuffix("_metrics"))
                                                 if key.removesuffix("_metrics") in order else len(order), key))
    for key in keys:
        set_name = key.removesuffix("_metrics")
        prefix = prefixes.get(set_name, set_name)
        for name, value in (parts[key] or {}).items():
            merged[f"{prefix}/{name}"] = value
    return merged
