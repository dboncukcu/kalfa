from kalfa.registration import lego
from kalfa.std.common.rng import derived_seed


@lego("/rng/kalfa/derived", alias="derived", partial=True,
      description="A substream per model by name: the build runs under sha256(seed:name), so inserting or "
                  "renaming another model changes no weights; None without a seed")
def derived(seed, name, index):
    return None if seed is None else derived_seed(seed, name)


@lego("/rng/kalfa/indexed", alias="indexed", partial=True,
      description="A substream per model by position: the build runs under sha256(seed:index), the rule of the "
                  "runs made before the rng key, which reproduce with it; None without a seed")
def indexed(seed, name, index):
    return None if seed is None else derived_seed(seed, index)


@lego("/rng/kalfa/global", alias="global", partial=True,
      description="One global stream: no model touches the RNG, every build draws from the stream the seed "
                  "started in build order, the way a plain script does")
def global_stream(seed, name, index):
    return None
