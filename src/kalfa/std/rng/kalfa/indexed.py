from kalfa.registration import lego
from kalfa.std.common.rng import derived_seed


@lego("/rng/kalfa/indexed", alias="indexed", partial=True,
      description="A substream per model by position: the build runs under sha256(seed:index), the rule of the "
                  "runs made before the rng key, which reproduce with it; None without a seed")
def indexed(seed, name, index):
    return None if seed is None else derived_seed(seed, index)
