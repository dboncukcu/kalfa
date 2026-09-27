import numpy

from .std.common.history import is_number


def classes_of(labels):
    present = labels.dropna().unique().tolist()
    try:
        return sorted(present)
    except TypeError:
        return sorted(present, key=str)


def same_label(item, text):
    if text is None:
        return False
    if isinstance(item, (bool, numpy.bool_)):
        return str(item).lower() == str(text).lower()
    if is_number(item) or isinstance(item, (numpy.integer, numpy.floating)):
        try:
            return float(text) == float(item)
        except ValueError:
            return False
    return str(item) == str(text)
