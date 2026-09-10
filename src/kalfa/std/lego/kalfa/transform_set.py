from kalfa.registration import lego


@lego("/lego/kalfa/transform_set",
      description="Apply the transforms that name this set, in order; the frame passes untouched without any")
def transform_set(df, set, transforms):
    out = df
    for transform in transforms or []:
        out = transform(out)
    return out
