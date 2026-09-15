from kalfa.registration import pack

lego = pack(__name__)


lego("/layer/kalfa/linear", "blocks:linear", alias="linear",
     description="Linear layer; without in_features the input width is taken from the first batch")
lego("/layer/kalfa/linear_relu", "blocks:linear_relu", alias="linear_relu",
     description="Linear layer followed by ReLU; lazy without in_features")
lego("/layer/kalfa/mlp", "blocks:mlp", alias="mlp",
     description="A multilayer perceptron in one node: a linear layer, the activation and dropout for every width, "
                 "then a plain linear layer of out_features when it is written; lazy without in_features")

lego("/layer/kalfa/polynomial", "features:Polynomial", alias="polynomial",
     description="Polynomial expansion of the feature vector: the features and every product of degree of them "
                 "(interaction_only drops the squares, bias adds a constant column, keep: false returns the products "
                 "alone); the place for feature interactions, computed per batch")
lego("/layer/kalfa/l2_normalize", "features:L2Normalize", alias="l2_normalize",
     description="Divide every sample by the L2 norm of its own feature vector (sklearn's Normalizer as a layer: it "
                 "reads the whole vector, so it belongs to the model, not to a column chain)")

lego("/layer/kalfa/l1_distance", "wires:L1Distance", alias="l1_distance",
     description="Mean absolute difference of two wires per sample")
lego("/layer/kalfa/reparam", "wires:Reparam", alias="reparam",
     description="Sample z from mu and logvar in train mode, return mu in eval mode")
lego("/layer/kalfa/unflatten", "wires:unflatten", alias="unflatten",
     description="Reshape the features of every sample to shape")
