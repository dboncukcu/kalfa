import math


def monitored(metrics, monitor):
    value = (metrics or {}).get(monitor)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)
