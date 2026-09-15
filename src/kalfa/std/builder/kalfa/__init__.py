from kalfa.registration import pack


lego = pack(__name__)


lego("/builder/kalfa/module", "module:module", bus=["prep", "train_loader"], roles=["weights", "bias", "scale"],
     description="Build a model graph into an nn.Module in the stream the rng lego derives from the seed, the name "
                 "and the index, apply init roles, trainable and weights; reference nodes take the models dict; layer "
                 "params that are kind data components are built from prep and the train loader")
