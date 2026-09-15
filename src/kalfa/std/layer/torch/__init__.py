from kalfa.registration import pack

lego = pack(__name__)


lego("/layer/torch/relu", "activations:relu", alias="relu", description="torch.nn.ReLU")
lego("/layer/torch/relu6", "activations:relu6", alias="relu6", description="torch.nn.ReLU6, ReLU clipped at 6")
lego("/layer/torch/leaky_relu", "activations:leaky_relu", alias="leaky_relu",
     description="torch.nn.LeakyReLU with negative_slope")
lego("/layer/torch/prelu", "activations:prelu", alias="prelu",
     description="torch.nn.PReLU, a learned slope per channel (num_parameters) starting at init")
lego("/layer/torch/rrelu", "activations:rrelu", alias="rrelu",
     description="torch.nn.RReLU, a random slope in [lower, upper] in train mode")
lego("/layer/torch/elu", "activations:elu", alias="elu", description="torch.nn.ELU with alpha")
lego("/layer/torch/celu", "activations:celu", alias="celu", description="torch.nn.CELU with alpha")
lego("/layer/torch/selu", "activations:selu", alias="selu",
     description="torch.nn.SELU, self normalizing; pairs with alpha_dropout")
lego("/layer/torch/gelu", "activations:gelu", alias="gelu", description="torch.nn.GELU; approximate none or tanh")
lego("/layer/torch/silu", "activations:silu", alias="silu", description="torch.nn.SiLU, x times sigmoid(x)")
lego("/layer/torch/mish", "activations:mish", alias="mish", description="torch.nn.Mish")
lego("/layer/torch/hardswish", "activations:hardswish", alias="hardswish", description="torch.nn.Hardswish")
lego("/layer/torch/hardsigmoid", "activations:hardsigmoid", alias="hardsigmoid", description="torch.nn.Hardsigmoid")
lego("/layer/torch/hardtanh", "activations:hardtanh", alias="hardtanh",
     description="torch.nn.Hardtanh clipped to [min_val, max_val]")
lego("/layer/torch/hardshrink", "activations:hardshrink", alias="hardshrink",
     description="torch.nn.Hardshrink, zero inside [-lambd, lambd]")
lego("/layer/torch/softshrink", "activations:softshrink", alias="softshrink",
     description="torch.nn.Softshrink, shrunk toward zero by lambd")
lego("/layer/torch/tanhshrink", "activations:tanhshrink", alias="tanhshrink",
     description="torch.nn.Tanhshrink, x minus tanh(x)")
lego("/layer/torch/softsign", "activations:softsign", alias="softsign",
     description="torch.nn.Softsign, x over 1 plus |x|")
lego("/layer/torch/softplus", "activations:softplus", alias="softplus",
     description="torch.nn.Softplus with beta and the linear threshold")
lego("/layer/torch/sigmoid", "activations:sigmoid", alias="sigmoid", description="torch.nn.Sigmoid")
lego("/layer/torch/log_sigmoid", "activations:log_sigmoid", alias="log_sigmoid", description="torch.nn.LogSigmoid")
lego("/layer/torch/tanh", "activations:tanh", alias="tanh_layer",
     description="torch.nn.Tanh; the alias is tanh_layer because tanh names the preprocessor")
lego("/layer/torch/threshold", "activations:threshold", alias="threshold_layer",
     description="torch.nn.Threshold: value where x is at or below threshold; the alias is threshold_layer because "
                 "threshold names the calibration")
lego("/layer/torch/glu", "activations:glu", alias="glu", description="torch.nn.GLU, the gated linear unit over dim")
lego("/layer/torch/softmax", "activations:softmax", alias="softmax", description="torch.nn.Softmax over dim")
lego("/layer/torch/softmin", "activations:softmin", alias="softmin", description="torch.nn.Softmin over dim")
lego("/layer/torch/log_softmax", "activations:log_softmax", alias="log_softmax",
     description="torch.nn.LogSoftmax over dim")
lego("/layer/torch/softmax2d", "activations:softmax2d", alias="softmax2d",
     description="torch.nn.Softmax2d, the softmax over the channels of an image")

lego("/layer/torch/multihead_attention", "attention:multihead_attention", alias="multihead_attention",
     description="torch.nn.MultiheadAttention over (batch, steps, embed_dim); inputs query, key and value, outputs "
                 "the attended values and the weights")
lego("/layer/torch/transformer_encoder_layer", "attention:transformer_encoder_layer", alias="transformer_encoder_layer",
     description="torch.nn.TransformerEncoderLayer over (batch, steps, d_model)")
lego("/layer/torch/transformer_encoder", "attention:transformer_encoder", alias="transformer_encoder",
     description="torch.nn.TransformerEncoder of layers encoder layers over (batch, steps, d_model)")
lego("/layer/torch/transformer_decoder_layer", "attention:transformer_decoder_layer", alias="transformer_decoder_layer",
     description="torch.nn.TransformerDecoderLayer; inputs the target sequence and the memory")
lego("/layer/torch/transformer_decoder", "attention:transformer_decoder", alias="transformer_decoder",
     description="torch.nn.TransformerDecoder of layers decoder layers; inputs the target sequence and the memory")
lego("/layer/torch/transformer", "attention:transformer", alias="transformer",
     description="torch.nn.Transformer, the encoder and the decoder; inputs the source and the target sequences")

lego("/layer/torch/conv1d", "convolution:conv1d", alias="conv1d",
     description="torch.nn.Conv1d; without in_channels the input channels are taken from the first batch")
lego("/layer/torch/conv2d", "convolution:conv2d", alias="conv2d",
     description="2d convolution; without in_channels the input channels are taken from the first batch")
lego("/layer/torch/conv3d", "convolution:conv3d", alias="conv3d",
     description="torch.nn.Conv3d; without in_channels the input channels are taken from the first batch")
lego("/layer/torch/conv_transpose1d", "convolution:conv_transpose1d", alias="conv_transpose1d",
     description="torch.nn.ConvTranspose1d; without in_channels the input channels are taken from the first batch")
lego("/layer/torch/conv_transpose2d", "convolution:conv_transpose2d", alias="conv_transpose2d",
     description="torch.nn.ConvTranspose2d; without in_channels the input channels are taken from the first batch")
lego("/layer/torch/conv_transpose3d", "convolution:conv_transpose3d", alias="conv_transpose3d",
     description="torch.nn.ConvTranspose3d; without in_channels the input channels are taken from the first batch")

lego("/layer/torch/cosine_similarity", "distance:cosine_similarity", alias="cosine_similarity",
     description="torch.nn.CosineSimilarity of two inputs along dim")
lego("/layer/torch/pairwise_distance", "distance:pairwise_distance", alias="pairwise_distance",
     description="torch.nn.PairwiseDistance of two inputs, the p norm of their difference")

lego("/layer/torch/dropout", "dropout:dropout", alias="dropout", description="torch.nn.Dropout")
lego("/layer/torch/dropout1d", "dropout:dropout1d", alias="dropout1d",
     description="torch.nn.Dropout1d, whole channels of a sequence")
lego("/layer/torch/dropout2d", "dropout:dropout2d", alias="dropout2d",
     description="torch.nn.Dropout2d, whole channels of an image")
lego("/layer/torch/dropout3d", "dropout:dropout3d", alias="dropout3d",
     description="torch.nn.Dropout3d, whole channels of a volume")
lego("/layer/torch/alpha_dropout", "dropout:alpha_dropout", alias="alpha_dropout",
     description="torch.nn.AlphaDropout, the dropout of selu")
lego("/layer/torch/feature_alpha_dropout", "dropout:feature_alpha_dropout", alias="feature_alpha_dropout",
     description="torch.nn.FeatureAlphaDropout, alpha dropout of whole channels")

lego("/layer/torch/embedding", "embedding:embedding", alias="embedding",
     description="torch.nn.Embedding(num, dim); num may be a kind data component such as vocab_size")
lego("/layer/torch/embedding_bag", "embedding:embedding_bag", alias="embedding_bag",
     description="torch.nn.EmbeddingBag(num, dim), the mean, sum or max of a bag of ids; num may be vocab_size")

lego("/layer/torch/linear", "linear:torch_linear", description="torch.nn.Linear with in_features written out")
lego("/layer/torch/bilinear", "linear:bilinear", alias="bilinear",
     description="torch.nn.Bilinear over two inputs of in1_features and in2_features")
lego("/layer/torch/identity", "linear:identity", alias="identity", description="torch.nn.Identity, the input as it is")

lego("/layer/torch/batch_norm", "normalization:batch_norm", alias="batch_norm",
     description="torch.nn.BatchNorm over the feature axis, lazy in the number of features: dims 1 for (batch, "
                 "features) and sequences, 2 for images, 3 for volumes")
lego("/layer/torch/instance_norm", "normalization:instance_norm", alias="instance_norm",
     description="torch.nn.InstanceNorm, lazy in the number of features: dims 1 for sequences, 2 for images, 3 for "
                 "volumes")
lego("/layer/torch/layer_norm", "normalization:layer_norm", alias="layer_norm",
     description="torch.nn.LayerNorm over the last axis; normalized_shape is its width, a number or {uri: "
                 "feature_width} for the width of the feature tensor")
lego("/layer/torch/rms_norm", "normalization:rms_norm", alias="rms_norm",
     description="torch.nn.RMSNorm over the last axis; normalized_shape is its width, a number or {uri: feature_width}")
lego("/layer/torch/group_norm", "normalization:group_norm", alias="group_norm",
     description="torch.nn.GroupNorm: num_groups groups over num_channels channels")
lego("/layer/torch/local_response_norm", "normalization:local_response_norm", alias="local_response_norm",
     description="torch.nn.LocalResponseNorm over size neighbouring channels")

lego("/layer/torch/zero_pad", "padding:zero_pad", alias="zero_pad",
     description="torch.nn.ZeroPad2d; padding is a number or [left, right, top, bottom]")
lego("/layer/torch/zero_pad1d", "padding:zero_pad1d", alias="zero_pad1d", description="torch.nn.ZeroPad1d")
lego("/layer/torch/zero_pad3d", "padding:zero_pad3d", alias="zero_pad3d", description="torch.nn.ZeroPad3d")
lego("/layer/torch/constant_pad", "padding:constant_pad", alias="constant_pad",
     description="torch.nn.ConstantPad2d with value")
lego("/layer/torch/constant_pad1d", "padding:constant_pad1d", alias="constant_pad1d",
     description="torch.nn.ConstantPad1d")
lego("/layer/torch/constant_pad3d", "padding:constant_pad3d", alias="constant_pad3d",
     description="torch.nn.ConstantPad3d")
lego("/layer/torch/reflection_pad", "padding:reflection_pad", alias="reflection_pad",
     description="torch.nn.ReflectionPad2d")
lego("/layer/torch/reflection_pad1d", "padding:reflection_pad1d", alias="reflection_pad1d",
     description="torch.nn.ReflectionPad1d")
lego("/layer/torch/reflection_pad3d", "padding:reflection_pad3d", alias="reflection_pad3d",
     description="torch.nn.ReflectionPad3d")
lego("/layer/torch/replication_pad", "padding:replication_pad", alias="replication_pad",
     description="torch.nn.ReplicationPad2d")
lego("/layer/torch/replication_pad1d", "padding:replication_pad1d", alias="replication_pad1d",
     description="torch.nn.ReplicationPad1d")
lego("/layer/torch/replication_pad3d", "padding:replication_pad3d", alias="replication_pad3d",
     description="torch.nn.ReplicationPad3d")
lego("/layer/torch/circular_pad", "padding:circular_pad", alias="circular_pad", description="torch.nn.CircularPad2d")
lego("/layer/torch/circular_pad1d", "padding:circular_pad1d", alias="circular_pad1d",
     description="torch.nn.CircularPad1d")
lego("/layer/torch/circular_pad3d", "padding:circular_pad3d", alias="circular_pad3d",
     description="torch.nn.CircularPad3d")

lego("/layer/torch/maxpool", "pooling:maxpool", alias="maxpool", description="torch.nn.MaxPool2d")
lego("/layer/torch/maxpool1d", "pooling:maxpool1d", alias="maxpool1d", description="torch.nn.MaxPool1d")
lego("/layer/torch/maxpool3d", "pooling:maxpool3d", alias="maxpool3d", description="torch.nn.MaxPool3d")
lego("/layer/torch/avgpool", "pooling:avgpool", alias="avgpool", description="torch.nn.AvgPool2d")
lego("/layer/torch/avgpool1d", "pooling:avgpool1d", alias="avgpool1d", description="torch.nn.AvgPool1d")
lego("/layer/torch/avgpool3d", "pooling:avgpool3d", alias="avgpool3d", description="torch.nn.AvgPool3d")
lego("/layer/torch/adaptive_avgpool", "pooling:adaptive_avgpool", alias="adaptive_avgpool",
     description="torch.nn.AdaptiveAvgPool2d to output_size, a number or [height, width]")
lego("/layer/torch/adaptive_avgpool1d", "pooling:adaptive_avgpool1d", alias="adaptive_avgpool1d",
     description="torch.nn.AdaptiveAvgPool1d")
lego("/layer/torch/adaptive_avgpool3d", "pooling:adaptive_avgpool3d", alias="adaptive_avgpool3d",
     description="torch.nn.AdaptiveAvgPool3d")
lego("/layer/torch/adaptive_maxpool", "pooling:adaptive_maxpool", alias="adaptive_maxpool",
     description="torch.nn.AdaptiveMaxPool2d to output_size, a number or [height, width]")
lego("/layer/torch/adaptive_maxpool1d", "pooling:adaptive_maxpool1d", alias="adaptive_maxpool1d",
     description="torch.nn.AdaptiveMaxPool1d")
lego("/layer/torch/adaptive_maxpool3d", "pooling:adaptive_maxpool3d", alias="adaptive_maxpool3d",
     description="torch.nn.AdaptiveMaxPool3d")
lego("/layer/torch/lppool", "pooling:lppool", alias="lppool",
     description="torch.nn.LPPool2d, the power average pool of norm_type")
lego("/layer/torch/lppool1d", "pooling:lppool1d", alias="lppool1d", description="torch.nn.LPPool1d")
lego("/layer/torch/lppool3d", "pooling:lppool3d", alias="lppool3d", description="torch.nn.LPPool3d")
lego("/layer/torch/fractional_maxpool", "pooling:fractional_maxpool", alias="fractional_maxpool",
     description="torch.nn.FractionalMaxPool2d to output_size or output_ratio")
lego("/layer/torch/fractional_maxpool3d", "pooling:fractional_maxpool3d", alias="fractional_maxpool3d",
     description="torch.nn.FractionalMaxPool3d")
lego("/layer/torch/max_unpool", "pooling:max_unpool", alias="max_unpool",
     description="torch.nn.MaxUnpool2d, the inverse of a max pool that kept its indices; two inputs")
lego("/layer/torch/max_unpool1d", "pooling:max_unpool1d", alias="max_unpool1d",
     description="torch.nn.MaxUnpool1d; two inputs")
lego("/layer/torch/max_unpool3d", "pooling:max_unpool3d", alias="max_unpool3d",
     description="torch.nn.MaxUnpool3d; two inputs")

lego("/layer/torch/gru", "recurrent:Gru", alias="gru",
     description="GRU over (batch, steps, features) returning every step; the input width comes from the first batch")
lego("/layer/torch/lstm", "recurrent:Lstm", alias="lstm",
     description="LSTM over (batch, steps, features) returning every step; the input width comes from the first batch")
lego("/layer/torch/rnn", "recurrent:Rnn", alias="rnn",
     description="Elman RNN over (batch, steps, features) returning every step, nonlinearity tanh or relu; the input "
                 "width comes from the first batch")
lego("/layer/torch/gru_cell", "recurrent:gru_cell", alias="gru_cell",
     description="torch.nn.GRUCell over one step; two inputs, x and h")
lego("/layer/torch/lstm_cell", "recurrent:lstm_cell", alias="lstm_cell",
     description="torch.nn.LSTMCell over one step; inputs x and (h, c), outputs (h, c)")
lego("/layer/torch/rnn_cell", "recurrent:rnn_cell", alias="rnn_cell",
     description="torch.nn.RNNCell over one step; two inputs, x and h")

lego("/layer/torch/concat", "shape:Concat", alias="concat", description="Concatenate wires along a dimension")
lego("/layer/torch/last_step", "shape:LastStep", alias="last_step", description="The last step of a sequence")
lego("/layer/torch/flatten", "shape:flatten", alias="flatten", description="torch.nn.Flatten from start_dim to end_dim")
lego("/layer/torch/unflatten", "shape:unflatten",
     description="torch.nn.Unflatten of dim into size; the alias unflatten names kalfa's per sample reshape")
lego("/layer/torch/fold", "shape:fold", alias="fold",
     description="torch.nn.Fold, sliding blocks back into an image of output_size")
lego("/layer/torch/unfold", "shape:unfold", alias="unfold", description="torch.nn.Unfold, an image into sliding blocks")
lego("/layer/torch/pixel_shuffle", "shape:pixel_shuffle", alias="pixel_shuffle",
     description="torch.nn.PixelShuffle by factor")
lego("/layer/torch/pixel_unshuffle", "shape:pixel_unshuffle", alias="pixel_unshuffle",
     description="torch.nn.PixelUnshuffle by factor")
lego("/layer/torch/channel_shuffle", "shape:channel_shuffle", alias="channel_shuffle",
     description="torch.nn.ChannelShuffle over groups")
lego("/layer/torch/upsample", "shape:upsample", alias="upsample",
     description="torch.nn.Upsample to size or by scale_factor; mode nearest, linear, bilinear, bicubic or trilinear")
