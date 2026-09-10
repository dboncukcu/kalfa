from kalfa.registration import lego


@lego("/rng/kalfa/global", alias="global", partial=True,
      description="One global stream: no model touches the RNG, every build draws from the stream the seed "
                  "started in build order, the way a plain script does")
def global_stream(seed, name, index):
    return None
