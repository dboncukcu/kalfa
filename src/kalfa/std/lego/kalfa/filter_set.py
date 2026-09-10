from kalfa.registration import lego


@lego("/lego/kalfa/filter_set",
      description="Apply the {query, sets} filters that name this set; the frame passes untouched otherwise")
def filter_set(df, set, filters):
    out = df
    for entry in filters or []:
        if set in (entry.get("sets") or []):
            out = out.query(entry["query"])
    return out
