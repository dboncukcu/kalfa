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

lego("/layer/kalfa/select", "features:select", alias="select",
     description="The positions of the feature axis the index names, in that order; index is a list of positions or a "
                 "run time component such as feature_index, and dim picks the axis (1, the features, by default). The "
                 "piece a long range skip connection needs: it takes the columns the output is a correction of out of "
                 "the input wire")

lego("/layer/kalfa/add", "wires:Add", alias="add",
     description="The elementwise sum of its input wires, the residual connection of a model graph; shapes broadcast")
lego("/layer/kalfa/subtract", "wires:Subtract", alias="subtract",
     description="The first input wire minus the second; shapes broadcast")
lego("/layer/kalfa/multiply", "wires:Multiply", alias="multiply",
     description="The elementwise product of its input wires, the gate of a model graph; shapes broadcast")
lego("/layer/kalfa/divide", "wires:Divide", alias="divide",
     description="The first input wire divided by the second; shapes broadcast and a zero divisor gives an infinity, "
                 "as torch does")
lego("/layer/kalfa/negate", "wires:Negate", alias="negate",
     description="Minus the input wire")

lego("/layer/kalfa/multipliers", "multipliers:Multipliers", alias="multipliers",
     description="The Lagrange multipliers of an mdmm loss as a model of their own: one learnable lambda per name, "
                 "in the order written, started at the lmbda_init of its entry (init without one); names takes the "
                 "constraints mapping of the loss (names: $constraints$) so the two stay in step, and forward ignores "
                 "its input and returns the vector. A model, so an optimizer updates them (a group matching <name>.* "
                 "with a negative lr, the gradient ascent of a saddle point) and the checkpoint carries them")

lego("/layer/kalfa/l1_distance", "wires:L1Distance", alias="l1_distance",
     description="Mean absolute difference of two wires per sample")
lego("/layer/kalfa/reparam", "wires:Reparam", alias="reparam",
     description="Sample z from mu and logvar in train mode, return mu in eval mode")
lego("/layer/kalfa/unflatten", "wires:unflatten", alias="unflatten",
     description="Reshape the features of every sample to shape")
