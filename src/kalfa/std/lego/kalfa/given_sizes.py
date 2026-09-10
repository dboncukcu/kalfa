from kalfa.registration import lego


def rows_of(path, header):
    if path is None:
        return 0
    if header is None:
        return None
    try:
        return header(path)["rows"]
    except Exception:
        return None


@lego("/lego/kalfa/given_sizes",
      description="The set sizes of a given split: the source rows for train, the header of every given file "
                  "for the other sets (header reads a path like the source)")
def given_sizes(rows, valid=None, test=None, header=None):
    return {"train": rows, "valid": rows_of(valid, header), "test": rows_of(test, header)}
