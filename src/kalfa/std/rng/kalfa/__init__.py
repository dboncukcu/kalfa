from kalfa.registration import pack


lego = pack(__name__)


lego("/rng/kalfa/derived", "seeds:derived", alias="derived", partial=True,
     description="A substream per model by name: the build runs under sha256(seed:name), so inserting or renaming "
                 "another model changes no weights; None without a seed")
lego("/rng/kalfa/indexed", "seeds:indexed", alias="indexed", partial=True,
     description="A substream per model by position: the build runs under sha256(seed:index), the rule of the runs "
                 "made before the rng key, which reproduce with it; None without a seed")
lego("/rng/kalfa/global", "seeds:global_stream", alias="global", partial=True,
     description="One global stream: no model touches the RNG, every build draws from the stream the seed started in "
                 "build order, the way a plain script does")
