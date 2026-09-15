from kalfa.registration import pack


lego = pack(__name__)


lego("/optimizer/torch/sgd", "optimizers:Sgd", state=True, refs={"loss": "loss", "schedule": "schedule"},
     aliases="models", alias="sgd", description="torch SGD over the union of its models")
lego("/optimizer/torch/adam", "optimizers:Adam", state=True, refs={"loss": "loss", "schedule": "schedule"},
     aliases="models", alias="adam",
     description="torch Adam over the union of its models; params are Adam's keyword arguments")
lego("/optimizer/torch/adamw", "optimizers:AdamW", state=True, refs={"loss": "loss", "schedule": "schedule"},
     aliases="models", alias="adamw", description="torch AdamW over the union of its models")
