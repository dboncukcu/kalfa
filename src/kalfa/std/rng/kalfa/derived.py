from kalfa.registration import lego
from kalfa.std.common.rng import derived_seed


@lego("/rng/kalfa/derived", alias="derived", partial=True,
      description="A substream per model by name: the build runs under sha256(seed:name), so inserting or "
                  "renaming another model changes no weights; None without a seed")
def derived(seed, name, index):
    return None if seed is None else derived_seed(seed, name)
