from kalfa.registration import pack

lego = pack(__name__)


lego("/data/kalfa/class_weights", "components:class_weights", alias="class_weights", counts=True,
     description="Inverse frequency class weights of the train set's target field, mean one; built once the train "
                 "loader exists")
lego("/data/kalfa/vocab_size", "components:vocab_size", alias="vocab_size",
     description="The vocabulary size of the fitted tokenizer among the preprocessors; built once prep exists")
lego("/data/kalfa/target_weights", "components:target_weights", alias="target_weights",
     description="A weight per target column, from names and globs resolved against the columns of the target field "
                 "the dataset carries (every target field in order without target), default for the rest; built once "
                 "the train loader exists")
lego("/data/kalfa/feature_width", "components:feature_width", alias="feature_width",
     description="The width of the feature tensor x, from the fitted plan the train loader carries; for a layer whose "
                 "shape follows it (layer_norm)")
