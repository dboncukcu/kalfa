from kalfa.registration import lego


@lego("/lego/kalfa/apply_frames",
      description="Apply the fitted frame transforms to one set, in the order they were fitted")
def apply_frames(df, frames):
    table = df
    for transform in frames or []:
        table = transform.apply(table)
    return table
