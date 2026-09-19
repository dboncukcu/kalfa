import math
from types import SimpleNamespace

import pytest
import torch
from cirak import Deferred
from cirak.registry import registry
from torch import nn

from helpers import build
from kalfa.std import STD_URIS
from kalfa.std.common.deferred import DeferredLayer, LazyLayer
from kalfa.std.layer.torch.recurrent import Recurrent


LAYER_URIS = sorted(uri for uri in STD_URIS if uri.startswith("/layer/"))
KALFA_URIS = ["/layer/kalfa/add", "/layer/kalfa/divide", "/layer/kalfa/l1_distance", "/layer/kalfa/l2_normalize",
              "/layer/kalfa/linear", "/layer/kalfa/linear_relu", "/layer/kalfa/mlp", "/layer/kalfa/multipliers",
              "/layer/kalfa/multiply", "/layer/kalfa/negate", "/layer/kalfa/polynomial", "/layer/kalfa/reparam",
              "/layer/kalfa/select", "/layer/kalfa/subtract", "/layer/kalfa/unflatten"]
SPECIAL_ALIASES = {"/layer/torch/tanh": "tanh_layer", "/layer/torch/threshold": "threshold_layer"}
UNALIASED = {"/layer/torch/linear", "/layer/torch/unflatten"}


def floats(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def ids(*shape, high=10, seed=0):
    return torch.randint(0, high, shape, generator=torch.Generator().manual_seed(seed))


def pooled(kind, *shape):
    return kind(2, return_indices=True)(floats(*shape))


def shapes_of(result):
    if isinstance(result, torch.Tensor):
        return tuple(result.shape)
    return tuple(shapes_of(item) for item in result)


def values(*numbers):
    return torch.tensor([list(numbers)])


ROW = (-2.0, -0.5, 0.0, 0.5, 2.0)

TORCH = [
    ("relu", "/layer/torch/relu", {}, (floats(2, 6),), (2, 6)),
    ("relu6", "/layer/torch/relu6", {}, (floats(2, 6),), (2, 6)),
    ("leaky_relu", "/layer/torch/leaky_relu", {"negative_slope": 0.2}, (floats(2, 6),), (2, 6)),
    ("prelu", "/layer/torch/prelu", {"num_parameters": 6, "init": 0.1}, (floats(2, 6),), (2, 6)),
    ("rrelu", "/layer/torch/rrelu", {"lower": 0.1, "upper": 0.3}, (floats(2, 6),), (2, 6)),
    ("elu", "/layer/torch/elu", {"alpha": 0.5}, (floats(2, 6),), (2, 6)),
    ("celu", "/layer/torch/celu", {"alpha": 2.0}, (floats(2, 6),), (2, 6)),
    ("selu", "/layer/torch/selu", {}, (floats(2, 6),), (2, 6)),
    ("gelu", "/layer/torch/gelu", {"approximate": "tanh"}, (floats(2, 6),), (2, 6)),
    ("silu", "/layer/torch/silu", {}, (floats(2, 6),), (2, 6)),
    ("mish", "/layer/torch/mish", {}, (floats(2, 6),), (2, 6)),
    ("hardswish", "/layer/torch/hardswish", {}, (floats(2, 6),), (2, 6)),
    ("hardsigmoid", "/layer/torch/hardsigmoid", {}, (floats(2, 6),), (2, 6)),
    ("hardtanh", "/layer/torch/hardtanh", {"min_val": -2.0, "max_val": 2.0}, (floats(2, 6),), (2, 6)),
    ("hardshrink", "/layer/torch/hardshrink", {"lambd": 0.3}, (floats(2, 6),), (2, 6)),
    ("softshrink", "/layer/torch/softshrink", {"lambd": 0.3}, (floats(2, 6),), (2, 6)),
    ("tanhshrink", "/layer/torch/tanhshrink", {}, (floats(2, 6),), (2, 6)),
    ("softsign", "/layer/torch/softsign", {}, (floats(2, 6),), (2, 6)),
    ("softplus", "/layer/torch/softplus", {"beta": 2.0, "threshold": 10.0}, (floats(2, 6),), (2, 6)),
    ("sigmoid", "/layer/torch/sigmoid", {}, (floats(2, 6),), (2, 6)),
    ("log_sigmoid", "/layer/torch/log_sigmoid", {}, (floats(2, 6),), (2, 6)),
    ("tanh", "/layer/torch/tanh", {}, (floats(2, 6),), (2, 6)),
    ("threshold", "/layer/torch/threshold", {"threshold": 0.1, "value": 20.0}, (floats(2, 6),), (2, 6)),
    ("glu", "/layer/torch/glu", {"dim": -1}, (floats(2, 6),), (2, 3)),
    ("softmax", "/layer/torch/softmax", {"dim": 1}, (floats(2, 6),), (2, 6)),
    ("softmin", "/layer/torch/softmin", {"dim": 1}, (floats(2, 6),), (2, 6)),
    ("log_softmax", "/layer/torch/log_softmax", {"dim": -1}, (floats(2, 6),), (2, 6)),
    ("softmax2d", "/layer/torch/softmax2d", {}, (floats(2, 3, 4, 4),), (2, 3, 4, 4)),
    ("multihead_attention", "/layer/torch/multihead_attention", {"embed_dim": 8, "heads": 2},
     (floats(2, 7, 8), floats(2, 7, 8, seed=1), floats(2, 7, 8, seed=2)), ((2, 7, 8), (2, 7, 7))),
    ("transformer_encoder_layer", "/layer/torch/transformer_encoder_layer",
     {"d_model": 8, "heads": 2, "dim_feedforward": 16, "norm_first": True}, (floats(2, 7, 8),), (2, 7, 8)),
    ("transformer_encoder", "/layer/torch/transformer_encoder",
     {"d_model": 8, "heads": 2, "layers": 2, "dim_feedforward": 16}, (floats(2, 7, 8),), (2, 7, 8)),
    ("transformer_decoder_layer", "/layer/torch/transformer_decoder_layer",
     {"d_model": 8, "heads": 2, "dim_feedforward": 16}, (floats(2, 7, 8), floats(2, 5, 8, seed=1)), (2, 7, 8)),
    ("transformer_decoder", "/layer/torch/transformer_decoder",
     {"d_model": 8, "heads": 2, "layers": 2, "dim_feedforward": 16}, (floats(2, 7, 8), floats(2, 5, 8, seed=1)),
     (2, 7, 8)),
    ("transformer", "/layer/torch/transformer",
     {"d_model": 8, "heads": 2, "encoder_layers": 1, "decoder_layers": 1, "dim_feedforward": 16},
     (floats(2, 5, 8), floats(2, 7, 8, seed=1)), (2, 7, 8)),
    ("conv1d", "/layer/torch/conv1d", {"out_channels": 4, "kernel": 3, "padding": 1}, (floats(2, 3, 8),), (2, 4, 8)),
    ("conv2d", "/layer/torch/conv2d", {"out_channels": 4, "kernel": 3}, (floats(2, 3, 8, 8),), (2, 4, 6, 6)),
    ("conv2d-in_channels", "/layer/torch/conv2d", {"out_channels": 4, "kernel": 3, "in_channels": 3},
     (floats(2, 3, 8, 8),), (2, 4, 6, 6)),
    ("conv3d", "/layer/torch/conv3d", {"out_channels": 4, "kernel": 3}, (floats(2, 3, 6, 6, 6),), (2, 4, 4, 4, 4)),
    ("conv_transpose1d", "/layer/torch/conv_transpose1d", {"out_channels": 4, "kernel": 2, "stride": 2},
     (floats(2, 3, 8),), (2, 4, 16)),
    ("conv_transpose2d", "/layer/torch/conv_transpose2d", {"out_channels": 4, "kernel": 2, "stride": 2},
     (floats(2, 3, 8, 8),), (2, 4, 16, 16)),
    ("conv_transpose2d-in_channels", "/layer/torch/conv_transpose2d",
     {"out_channels": 4, "kernel": 2, "stride": 2, "in_channels": 3}, (floats(2, 3, 8, 8),), (2, 4, 16, 16)),
    ("conv_transpose3d", "/layer/torch/conv_transpose3d", {"out_channels": 4, "kernel": 2, "stride": 2},
     (floats(2, 3, 4, 4, 4),), (2, 4, 8, 8, 8)),
    ("cosine_similarity", "/layer/torch/cosine_similarity", {}, (floats(2, 5), floats(2, 5, seed=1)), (2,)),
    ("pairwise_distance", "/layer/torch/pairwise_distance", {"p": 1}, (floats(2, 5), floats(2, 5, seed=1)), (2,)),
    ("pairwise_distance-keepdim", "/layer/torch/pairwise_distance", {"keepdim": True},
     (floats(2, 5), floats(2, 5, seed=1)), (2, 1)),
    ("dropout", "/layer/torch/dropout", {"p": 0.2}, (floats(2, 6),), (2, 6)),
    ("dropout1d", "/layer/torch/dropout1d", {}, (floats(2, 3, 7),), (2, 3, 7)),
    ("dropout2d", "/layer/torch/dropout2d", {}, (floats(2, 3, 5, 5),), (2, 3, 5, 5)),
    ("dropout3d", "/layer/torch/dropout3d", {}, (floats(2, 3, 4, 4, 4),), (2, 3, 4, 4, 4)),
    ("alpha_dropout", "/layer/torch/alpha_dropout", {"p": 0.2}, (floats(2, 6),), (2, 6)),
    ("feature_alpha_dropout", "/layer/torch/feature_alpha_dropout", {}, (floats(2, 3, 5, 5),), (2, 3, 5, 5)),
    ("embedding", "/layer/torch/embedding", {"num": 10, "dim": 4, "padding_idx": 0}, (ids(2, 3),), (2, 3, 4)),
    ("embedding_bag", "/layer/torch/embedding_bag", {"num": 10, "dim": 4, "mode": "sum"}, (ids(2, 3),), (2, 4)),
    ("linear", "/layer/torch/linear", {"in_features": 5, "out_features": 3, "bias": False}, (floats(2, 5),),
     (2, 3)),
    ("bilinear", "/layer/torch/bilinear", {"in1_features": 5, "in2_features": 4, "out_features": 3},
     (floats(2, 5), floats(2, 4, seed=1)), (2, 3)),
    ("identity", "/layer/torch/identity", {}, (floats(2, 6),), (2, 6)),
    ("batch_norm", "/layer/torch/batch_norm", {}, (floats(4, 6),), (4, 6)),
    ("batch_norm-dims2", "/layer/torch/batch_norm", {"dims": 2}, (floats(4, 3, 5, 5),), (4, 3, 5, 5)),
    ("batch_norm-dims3", "/layer/torch/batch_norm", {"dims": 3, "affine": False}, (floats(4, 3, 2, 2, 2),),
     (4, 3, 2, 2, 2)),
    ("instance_norm", "/layer/torch/instance_norm", {}, (floats(4, 3, 5, 5),), (4, 3, 5, 5)),
    ("instance_norm-dims1", "/layer/torch/instance_norm", {"dims": 1}, (floats(4, 3, 7),), (4, 3, 7)),
    ("instance_norm-dims3", "/layer/torch/instance_norm", {"dims": 3, "affine": True}, (floats(4, 3, 2, 2, 2),),
     (4, 3, 2, 2, 2)),
    ("layer_norm", "/layer/torch/layer_norm", {"normalized_shape": 6}, (floats(4, 6),), (4, 6)),
    ("layer_norm-list", "/layer/torch/layer_norm", {"normalized_shape": [5, 6], "elementwise_affine": False},
     (floats(4, 5, 6),), (4, 5, 6)),
    ("rms_norm", "/layer/torch/rms_norm", {"normalized_shape": 6, "eps": 1e-6}, (floats(4, 6),), (4, 6)),
    ("group_norm", "/layer/torch/group_norm", {"num_groups": 2, "num_channels": 6}, (floats(4, 6, 5, 5),),
     (4, 6, 5, 5)),
    ("local_response_norm", "/layer/torch/local_response_norm", {"size": 2}, (floats(4, 6, 5, 5),), (4, 6, 5, 5)),
    ("zero_pad", "/layer/torch/zero_pad", {"padding": 1}, (floats(2, 3, 4, 4),), (2, 3, 6, 6)),
    ("zero_pad1d", "/layer/torch/zero_pad1d", {"padding": [1, 2]}, (floats(2, 3, 4),), (2, 3, 7)),
    ("zero_pad3d", "/layer/torch/zero_pad3d", {"padding": 1}, (floats(2, 3, 4, 4, 4),), (2, 3, 6, 6, 6)),
    ("constant_pad", "/layer/torch/constant_pad", {"padding": [1, 0, 2, 0], "value": 7.0}, (floats(2, 3, 4, 4),),
     (2, 3, 6, 5)),
    ("constant_pad1d", "/layer/torch/constant_pad1d", {"padding": [1, 2]}, (floats(2, 3, 4),), (2, 3, 7)),
    ("constant_pad3d", "/layer/torch/constant_pad3d", {"padding": 1, "value": -1.0}, (floats(2, 3, 4, 4, 4),),
     (2, 3, 6, 6, 6)),
    ("reflection_pad", "/layer/torch/reflection_pad", {"padding": 1}, (floats(2, 3, 4, 4),), (2, 3, 6, 6)),
    ("reflection_pad1d", "/layer/torch/reflection_pad1d", {"padding": [1, 2]}, (floats(2, 3, 4),), (2, 3, 7)),
    ("reflection_pad3d", "/layer/torch/reflection_pad3d", {"padding": 1}, (floats(2, 3, 4, 4, 4),),
     (2, 3, 6, 6, 6)),
    ("replication_pad", "/layer/torch/replication_pad", {"padding": 1}, (floats(2, 3, 4, 4),), (2, 3, 6, 6)),
    ("replication_pad1d", "/layer/torch/replication_pad1d", {"padding": [1, 2]}, (floats(2, 3, 4),), (2, 3, 7)),
    ("replication_pad3d", "/layer/torch/replication_pad3d", {"padding": 1}, (floats(2, 3, 4, 4, 4),),
     (2, 3, 6, 6, 6)),
    ("circular_pad", "/layer/torch/circular_pad", {"padding": 1}, (floats(2, 3, 4, 4),), (2, 3, 6, 6)),
    ("circular_pad1d", "/layer/torch/circular_pad1d", {"padding": [1, 2]}, (floats(2, 3, 4),), (2, 3, 7)),
    ("circular_pad3d", "/layer/torch/circular_pad3d", {"padding": 1}, (floats(2, 3, 4, 4, 4),), (2, 3, 6, 6, 6)),
    ("maxpool", "/layer/torch/maxpool", {"kernel": 2}, (floats(2, 3, 8, 8),), (2, 3, 4, 4)),
    ("maxpool1d", "/layer/torch/maxpool1d", {"kernel": 2}, (floats(2, 3, 8),), (2, 3, 4)),
    ("maxpool3d", "/layer/torch/maxpool3d", {"kernel": 2}, (floats(2, 3, 8, 8, 8),), (2, 3, 4, 4, 4)),
    ("avgpool", "/layer/torch/avgpool", {"kernel": 2, "stride": 2}, (floats(2, 3, 8, 8),), (2, 3, 4, 4)),
    ("avgpool1d", "/layer/torch/avgpool1d", {"kernel": 3, "stride": 1, "padding": 1}, (floats(2, 3, 8),),
     (2, 3, 8)),
    ("avgpool3d", "/layer/torch/avgpool3d", {"kernel": 2}, (floats(2, 3, 8, 8, 8),), (2, 3, 4, 4, 4)),
    ("adaptive_avgpool", "/layer/torch/adaptive_avgpool", {}, (floats(2, 3, 8, 8),), (2, 3, 1, 1)),
    ("adaptive_avgpool1d", "/layer/torch/adaptive_avgpool1d", {"output_size": 2}, (floats(2, 3, 8),), (2, 3, 2)),
    ("adaptive_avgpool3d", "/layer/torch/adaptive_avgpool3d", {}, (floats(2, 3, 8, 8, 8),), (2, 3, 1, 1, 1)),
    ("adaptive_maxpool", "/layer/torch/adaptive_maxpool", {"output_size": [2, 3]}, (floats(2, 3, 8, 8),),
     (2, 3, 2, 3)),
    ("adaptive_maxpool1d", "/layer/torch/adaptive_maxpool1d", {}, (floats(2, 3, 8),), (2, 3, 1)),
    ("adaptive_maxpool3d", "/layer/torch/adaptive_maxpool3d", {}, (floats(2, 3, 8, 8, 8),), (2, 3, 1, 1, 1)),
    ("lppool", "/layer/torch/lppool", {"norm_type": 2, "kernel": 2}, (floats(2, 3, 8, 8),), (2, 3, 4, 4)),
    ("lppool1d", "/layer/torch/lppool1d", {"norm_type": 2, "kernel": 2}, (floats(2, 3, 8),), (2, 3, 4)),
    ("lppool3d", "/layer/torch/lppool3d", {"norm_type": 2, "kernel": 2}, (floats(2, 3, 8, 8, 8),), (2, 3, 4, 4, 4)),
    ("fractional_maxpool", "/layer/torch/fractional_maxpool", {"kernel": 2, "output_size": 5},
     (floats(2, 3, 8, 8),), (2, 3, 5, 5)),
    ("fractional_maxpool3d", "/layer/torch/fractional_maxpool3d", {"kernel": 2, "output_ratio": 0.5},
     (floats(2, 3, 8, 8, 8),), (2, 3, 4, 4, 4)),
    ("max_unpool", "/layer/torch/max_unpool", {"kernel": 2}, pooled(nn.MaxPool2d, 2, 3, 8, 8), (2, 3, 8, 8)),
    ("max_unpool1d", "/layer/torch/max_unpool1d", {"kernel": 2}, pooled(nn.MaxPool1d, 2, 3, 8), (2, 3, 8)),
    ("max_unpool3d", "/layer/torch/max_unpool3d", {"kernel": 2}, pooled(nn.MaxPool3d, 2, 3, 8, 8, 8),
     (2, 3, 8, 8, 8)),
    ("gru", "/layer/torch/gru", {"hidden": 6, "layers": 2, "dropout": 0.1}, (floats(2, 7, 5),), (2, 7, 6)),
    ("gru-bidirectional", "/layer/torch/gru", {"hidden": 6, "bidirectional": True}, (floats(2, 7, 5),), (2, 7, 12)),
    ("lstm", "/layer/torch/lstm", {"hidden": 6, "layers": 2, "dropout": 0.1}, (floats(2, 7, 5),), (2, 7, 6)),
    ("rnn", "/layer/torch/rnn", {"hidden": 6, "nonlinearity": "relu"}, (floats(2, 7, 5),), (2, 7, 6)),
    ("gru_cell", "/layer/torch/gru_cell", {"input_size": 5, "hidden": 6}, (floats(2, 5), torch.zeros(2, 6)), (2, 6)),
    ("lstm_cell", "/layer/torch/lstm_cell", {"input_size": 5, "hidden": 6}, (floats(2, 5),), ((2, 6), (2, 6))),
    ("rnn_cell", "/layer/torch/rnn_cell", {"input_size": 5, "hidden": 6, "nonlinearity": "relu"}, (floats(2, 5),),
     (2, 6)),
    ("concat", "/layer/torch/concat", {}, (floats(2, 3), floats(2, 4, seed=1)), (2, 7)),
    ("concat-dim0", "/layer/torch/concat", {"dim": 0}, (floats(2, 3), floats(5, 3, seed=1)), (7, 3)),
    ("last_step", "/layer/torch/last_step", {}, (floats(2, 7, 5),), (2, 5)),
    ("flatten", "/layer/torch/flatten", {}, (floats(2, 3, 4, 4),), (2, 48)),
    ("flatten-start_dim", "/layer/torch/flatten", {"start_dim": 2}, (floats(2, 3, 4, 4),), (2, 3, 16)),
    ("unflatten", "/layer/torch/unflatten", {"dim": 1, "size": [3, 4]}, (floats(2, 12),), (2, 3, 4)),
    ("fold", "/layer/torch/fold", {"output_size": [4, 4], "kernel": 2}, (floats(2, 12, 9),), (2, 3, 4, 4)),
    ("unfold", "/layer/torch/unfold", {"kernel": 2}, (floats(2, 3, 4, 4),), (2, 12, 9)),
    ("pixel_shuffle", "/layer/torch/pixel_shuffle", {"factor": 2}, (floats(2, 12, 4, 4),), (2, 3, 8, 8)),
    ("pixel_unshuffle", "/layer/torch/pixel_unshuffle", {"factor": 2}, (floats(2, 3, 8, 8),), (2, 12, 4, 4)),
    ("channel_shuffle", "/layer/torch/channel_shuffle", {"groups": 2}, (floats(2, 6, 4, 4),), (2, 6, 4, 4)),
    ("upsample", "/layer/torch/upsample", {"scale_factor": 2}, (floats(2, 3, 4, 4),), (2, 3, 8, 8)),
    ("upsample-bilinear", "/layer/torch/upsample", {"size": [6, 6], "mode": "bilinear", "align_corners": False},
     (floats(2, 3, 4, 4),), (2, 3, 6, 6)),
]

EXACT = [
    ("relu", "/layer/torch/relu", {}, ROW, [0.0, 0.0, 0.0, 0.5, 2.0]),
    ("relu6", "/layer/torch/relu6", {}, (-2.0, 0.5, 8.0), [0.0, 0.5, 6.0]),
    ("leaky_relu", "/layer/torch/leaky_relu", {"negative_slope": 0.2}, ROW, [-0.4, -0.1, 0.0, 0.5, 2.0]),
    ("prelu", "/layer/torch/prelu", {}, ROW, [-0.5, -0.125, 0.0, 0.5, 2.0]),
    ("rrelu", "/layer/torch/rrelu", {}, ROW, [-0.4583333, -0.1145833, 0.0, 0.5, 2.0]),
    ("elu", "/layer/torch/elu", {"alpha": 0.5}, ROW, [-0.4323324, -0.1967347, 0.0, 0.5, 2.0]),
    ("celu", "/layer/torch/celu", {"alpha": 2.0}, ROW, [-1.2642411, -0.4423984, 0.0, 0.5, 2.0]),
    ("selu", "/layer/torch/selu", {}, ROW, [-1.5201665, -0.6917582, 0.0, 0.5253505, 2.101402]),
    ("gelu", "/layer/torch/gelu", {}, ROW, [-0.0455003, -0.1542688, 0.0, 0.3457312, 1.9544997]),
    ("gelu-tanh", "/layer/torch/gelu", {"approximate": "tanh"}, ROW,
     [-0.0454023, -0.154286, 0.0, 0.345714, 1.9545977]),
    ("silu", "/layer/torch/silu", {}, ROW, [-0.2384058, -0.1887703, 0.0, 0.3112297, 1.7615942]),
    ("mish", "/layer/torch/mish", {}, ROW, [-0.2525015, -0.2207438, 0.0, 0.3752452, 1.943959]),
    ("hardswish", "/layer/torch/hardswish", {}, ROW, [-0.3333333, -0.2083333, 0.0, 0.2916667, 1.6666667]),
    ("hardsigmoid", "/layer/torch/hardsigmoid", {}, ROW, [0.1666667, 0.4166667, 0.5, 0.5833333, 0.8333333]),
    ("hardtanh", "/layer/torch/hardtanh", {}, ROW, [-1.0, -0.5, 0.0, 0.5, 1.0]),
    ("hardshrink", "/layer/torch/hardshrink", {}, ROW, [-2.0, 0.0, 0.0, 0.0, 2.0]),
    ("softshrink", "/layer/torch/softshrink", {}, ROW, [-1.5, 0.0, 0.0, 0.0, 1.5]),
    ("tanhshrink", "/layer/torch/tanhshrink", {}, ROW, [-1.0359724, -0.0378828, 0.0, 0.0378828, 1.0359724]),
    ("softsign", "/layer/torch/softsign", {}, ROW, [-0.6666667, -0.3333333, 0.0, 0.3333333, 0.6666667]),
    ("softplus", "/layer/torch/softplus", {"beta": 2.0}, ROW, [0.009075, 0.1566308, 0.3465736, 0.6566308, 2.009075]),
    ("sigmoid", "/layer/torch/sigmoid", {}, ROW, [0.1192029, 0.3775407, 0.5, 0.6224593, 0.8807971]),
    ("log_sigmoid", "/layer/torch/log_sigmoid", {}, ROW, [-2.126928, -0.974077, -0.6931472, -0.474077, -0.126928]),
    ("tanh", "/layer/torch/tanh", {}, ROW, [-0.9640276, -0.4621172, 0.0, 0.4621172, 0.9640276]),
    ("threshold", "/layer/torch/threshold", {"threshold": 0.1, "value": 20.0}, ROW, [20.0, 20.0, 20.0, 0.5, 2.0]),
    ("glu", "/layer/torch/glu", {}, (1.0, 2.0, 0.0, 0.0), [0.5, 1.0]),
    ("softmax", "/layer/torch/softmax", {}, (0.0, 0.0, 0.0, 0.0), [0.25, 0.25, 0.25, 0.25]),
    ("softmin", "/layer/torch/softmin", {}, (0.0, math.log(3.0)), [0.75, 0.25]),
    ("log_softmax", "/layer/torch/log_softmax", {}, (0.0, 0.0), [-math.log(2.0), -math.log(2.0)]),
]


@pytest.mark.parametrize("uri, params, inputs, expected", [row[1:] for row in TORCH], ids=[row[0] for row in TORCH])
def test_torch_layer_maps_its_inputs_to_the_expected_shape(uri, params, inputs, expected):
    layer = build(uri, **params)
    assert isinstance(layer, nn.Module)
    assert shapes_of(layer(*inputs)) == expected


@pytest.mark.parametrize("uri, params, inputs, expected", [row[1:] for row in EXACT], ids=[row[0] for row in EXACT])
def test_activation_computes_the_documented_function(uri, params, inputs, expected):
    layer = build(uri, **params).eval()
    assert layer(values(*inputs)).tolist()[0] == pytest.approx(expected, abs=1e-6)


def test_softmax2d_normalizes_over_the_channels_of_every_pixel():
    out = build("/layer/torch/softmax2d")(torch.tensor([[[[0.0]], [[math.log(3.0)]]]]))
    assert tuple(out.shape) == (1, 2, 1, 1) and out.reshape(-1).tolist() == pytest.approx([0.25, 0.75])


def test_the_tables_cover_every_torch_layer_and_the_kalfa_list_every_kalfa_layer():
    assert {row[1] for row in TORCH} == {uri for uri in LAYER_URIS if uri.startswith("/layer/torch/")}
    assert {row[1] for row in EXACT} <= {row[1] for row in TORCH}
    assert KALFA_URIS == [uri for uri in LAYER_URIS if uri.startswith("/layer/kalfa/")]
    assert len(LAYER_URIS) == 125


@pytest.mark.parametrize("uri", LAYER_URIS, ids=[uri[len("/layer/"):] for uri in LAYER_URIS])
def test_layer_alias_is_its_name(uri):
    aliases = registry.aliases()
    if uri in UNALIASED:
        assert registry.facts(uri).alias == ()
        assert uri not in aliases.values()
    else:
        assert aliases[SPECIAL_ALIASES.get(uri, uri.rsplit("/", 1)[1])] == uri


def test_the_short_names_of_tanh_and_threshold_belong_to_the_preprocessor_and_the_calibration():
    aliases = registry.aliases()
    assert aliases["tanh"] == "/pre/kalfa/tanh"
    assert aliases["threshold"] == "/calibrate/kalfa/threshold"
    assert aliases["linear"] == "/layer/kalfa/linear"
    assert aliases["unflatten"] == "/layer/kalfa/unflatten"


def test_convolution_is_lazy_without_in_channels_and_written_out_with_them():
    lazy = build("/layer/torch/conv2d", out_channels=4, kernel=3)
    assert isinstance(lazy, nn.LazyConv2d)
    lazy(floats(2, 3, 8, 8))
    assert isinstance(lazy, nn.Conv2d) and tuple(lazy.weight.shape) == (4, 3, 3, 3)
    fixed = build("/layer/torch/conv2d", out_channels=4, kernel=[3, 2], in_channels=3, stride=[2, 1],
                  dilation=1, groups=1, bias=False)
    assert type(fixed) is nn.Conv2d
    assert fixed.kernel_size == (3, 2) and fixed.stride == (2, 1) and fixed.bias is None
    assert tuple(fixed(floats(2, 3, 8, 8)).shape) == (2, 4, 3, 7)
    same = build("/layer/torch/conv1d", out_channels=2, kernel=3, padding="same", in_channels=3)
    assert tuple(same(floats(2, 3, 8)).shape) == (2, 2, 8)
    transposed = build("/layer/torch/conv_transpose2d", out_channels=2, kernel=3, stride=2, padding=1,
                       output_padding=1, in_channels=3)
    assert type(transposed) is nn.ConvTranspose2d and transposed.output_padding == (1, 1)
    assert tuple(transposed(floats(2, 3, 4, 4)).shape) == (2, 2, 8, 8)
    assert isinstance(build("/layer/torch/conv_transpose1d", out_channels=2, kernel=2), nn.LazyConvTranspose1d)
    assert isinstance(build("/layer/torch/conv3d", out_channels=2, kernel=2), nn.LazyConv3d)
    assert isinstance(build("/layer/torch/conv_transpose3d", out_channels=2, kernel=2), nn.LazyConvTranspose3d)


def test_distances_compute_the_cosine_and_the_p_norm_of_the_difference():
    cosine = build("/layer/torch/cosine_similarity")
    first = torch.tensor([[1.0, 0.0], [1.0, 0.0], [3.0, 4.0]])
    second = torch.tensor([[2.0, 0.0], [0.0, 5.0], [-3.0, -4.0]])
    assert cosine(first, second).tolist() == pytest.approx([1.0, 0.0, -1.0])
    assert build("/layer/torch/cosine_similarity", dim=0)(first, second).tolist() == pytest.approx(
        [(2.0 - 9.0) / (math.sqrt(11.0) * math.sqrt(13.0)), -16.0 / (4.0 * math.sqrt(41.0))])
    manhattan = build("/layer/torch/pairwise_distance", p=1)
    assert manhattan(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 5.0]])).tolist() == pytest.approx([5.0], abs=1e-4)
    euclid = build("/layer/torch/pairwise_distance", keepdim=True)
    kept = euclid(torch.tensor([[0.0, 0.0]]), torch.tensor([[3.0, 4.0]]))
    assert tuple(kept.shape) == (1, 1) and kept.item() == pytest.approx(5.0, abs=1e-4)


def test_dropout_is_the_identity_in_eval_mode_and_drops_whole_channels_in_train_mode():
    x = floats(2, 6)
    for uri in ("/layer/torch/dropout", "/layer/torch/alpha_dropout"):
        assert torch.equal(build(uri, p=0.5).eval()(x), x)
    torch.manual_seed(0)
    kept = build("/layer/torch/dropout", p=0.5).train()(torch.ones(1000))
    assert set(kept.unique().tolist()) == {0.0, 2.0}
    assert 400 < int((kept == 0.0).sum()) < 600
    torch.manual_seed(0)
    image = torch.ones(2, 8, 3, 3)
    channels = build("/layer/torch/dropout2d", p=0.5).train()(image)
    flat = channels.reshape(2, 8, 9)
    assert all(row.tolist() in ([0.0] * 9, [2.0] * 9) for row in flat.reshape(16, 9))
    assert 0 < int((flat[:, :, 0] == 0.0).sum()) < 16
    sequence = build("/layer/torch/dropout1d", p=0.5).train()(torch.ones(2, 8, 5))
    assert all(row.tolist() in ([0.0] * 5, [2.0] * 5) for row in sequence.reshape(16, 5))
    volume = build("/layer/torch/dropout3d", p=0.5).train()(torch.ones(2, 8, 2, 2, 2))
    assert all(row.tolist() in ([0.0] * 8, [2.0] * 8) for row in volume.reshape(16, 8))
    assert torch.equal(build("/layer/torch/feature_alpha_dropout", p=0.5).eval()(image), image)


def test_embedding_rows_are_looked_up_and_the_bag_reduces_them():
    embedding = build("/layer/torch/embedding", num=10, dim=4, padding_idx=0)
    assert type(embedding) is nn.Embedding and tuple(embedding.weight.shape) == (10, 4)
    assert embedding.weight[0].tolist() == [0.0] * 4
    looked = embedding(torch.tensor([[3, 0], [7, 7]]))
    assert torch.equal(looked[0, 0], embedding.weight[3]) and looked[0, 1].tolist() == [0.0] * 4
    assert torch.equal(looked[1, 0], looked[1, 1])
    bag = build("/layer/torch/embedding_bag", num=10, dim=4, mode="sum")
    assert type(bag) is nn.EmbeddingBag and bag.mode == "sum"
    tokens = torch.tensor([[3, 7], [1, 1]])
    assert torch.allclose(bag(tokens), bag.weight[tokens].sum(dim=1))
    mean = build("/layer/torch/embedding_bag", num=10, dim=4)
    assert mean.mode == "mean" and torch.allclose(mean(tokens), mean.weight[tokens].mean(dim=1))


def test_linear_layers_take_their_widths_written_out():
    linear = build("/layer/torch/linear", in_features=5, out_features=3, bias=False)
    assert type(linear) is nn.Linear and tuple(linear.weight.shape) == (3, 5) and linear.bias is None
    x = floats(2, 5)
    assert torch.allclose(linear(x), x @ linear.weight.T)
    bilinear = build("/layer/torch/bilinear", in1_features=5, in2_features=4, out_features=3)
    assert tuple(bilinear.weight.shape) == (3, 5, 4) and tuple(bilinear.bias.shape) == (3,)
    identity = build("/layer/torch/identity")
    assert identity(x) is x


def test_normalizations_are_lazy_in_the_width_and_reject_other_dims():
    batch = build("/layer/torch/batch_norm", momentum=0.2, eps=1e-3)
    assert isinstance(batch, nn.LazyBatchNorm1d) and batch.momentum == 0.2 and batch.eps == 1e-3
    batch(floats(4, 6))
    assert type(batch) is nn.BatchNorm1d and batch.num_features == 6
    assert type(build("/layer/torch/batch_norm", dims=2)) is nn.LazyBatchNorm2d
    assert type(build("/layer/torch/batch_norm", dims=3)) is nn.LazyBatchNorm3d
    with pytest.raises(ValueError, match=r"^batch_norm.dims must be 1, 2 or 3, got 4$"):
        build("/layer/torch/batch_norm", dims=4)
    instance = build("/layer/torch/instance_norm")
    assert type(instance) is nn.LazyInstanceNorm2d and instance.affine is False
    instance(floats(4, 3, 5, 5))
    assert type(instance) is nn.InstanceNorm2d and instance.num_features == 3 and instance.weight is None
    assert type(build("/layer/torch/instance_norm", dims=1)) is nn.LazyInstanceNorm1d
    assert type(build("/layer/torch/instance_norm", dims=3)) is nn.LazyInstanceNorm3d
    with pytest.raises(ValueError, match=r"^instance_norm.dims must be 1, 2 or 3, got 0$"):
        build("/layer/torch/instance_norm", dims=0)
    layer = build("/layer/torch/layer_norm", normalized_shape=3)
    assert type(layer) is nn.LayerNorm and layer.normalized_shape == (3,)
    scale = math.sqrt(1.5)
    assert layer(torch.tensor([[1.0, 2.0, 3.0]])).tolist()[0] == pytest.approx([-scale, 0.0, scale], abs=1e-4)
    assert build("/layer/torch/layer_norm", normalized_shape=[5, 6]).normalized_shape == (5, 6)
    rms = build("/layer/torch/rms_norm", normalized_shape=2)
    assert type(rms) is nn.RMSNorm and rms.eps is None
    assert rms(torch.tensor([[3.0, 4.0]])).tolist()[0] == pytest.approx([3.0 / 3.5355339, 4.0 / 3.5355339])
    group = build("/layer/torch/group_norm", num_groups=2, num_channels=6)
    assert group.num_groups == 2 and tuple(group.weight.shape) == (6,)
    local = build("/layer/torch/local_response_norm", size=3, alpha=0.5, beta=1.0, k=2.0)
    assert (local.size, local.alpha, local.beta, local.k) == (3, 0.5, 1.0, 2.0)


def test_pads_extend_a_sequence_the_way_their_name_says():
    sequence = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    assert build("/layer/torch/zero_pad1d", padding=1)(sequence).tolist() == [[[0.0, 1.0, 2.0, 3.0, 4.0, 0.0]]]
    assert build("/layer/torch/constant_pad1d", padding=[1, 2], value=7.0)(sequence).tolist() == \
        [[[7.0, 1.0, 2.0, 3.0, 4.0, 7.0, 7.0]]]
    assert build("/layer/torch/reflection_pad1d", padding=1)(sequence).tolist() == [[[2.0, 1.0, 2.0, 3.0, 4.0, 3.0]]]
    assert build("/layer/torch/replication_pad1d", padding=1)(sequence).tolist() == \
        [[[1.0, 1.0, 2.0, 3.0, 4.0, 4.0]]]
    assert build("/layer/torch/circular_pad1d", padding=1)(sequence).tolist() == [[[4.0, 1.0, 2.0, 3.0, 4.0, 1.0]]]
    image = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    assert build("/layer/torch/zero_pad", padding=[1, 0, 0, 1])(image).tolist() == \
        [[[[0.0, 1.0, 2.0], [0.0, 3.0, 4.0], [0.0, 0.0, 0.0]]]]
    assert build("/layer/torch/constant_pad", padding=1, value=-1.0)(image)[0, 0, 0].tolist() == [-1.0] * 4
    assert build("/layer/torch/reflection_pad", padding=1)(image)[0, 0, 0].tolist() == [4.0, 3.0, 4.0, 3.0]
    assert build("/layer/torch/replication_pad", padding=1)(image)[0, 0, 0].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert build("/layer/torch/circular_pad", padding=1)(image)[0, 0, 0].tolist() == [4.0, 3.0, 4.0, 3.0]
    volume = torch.arange(8.0).reshape(1, 1, 2, 2, 2)
    assert build("/layer/torch/zero_pad3d", padding=1)(volume)[0, 0, 1, 1].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert build("/layer/torch/constant_pad3d", padding=1, value=9.0)(volume)[0, 0, 0, 0].tolist() == [9.0] * 4
    assert build("/layer/torch/reflection_pad3d", padding=1)(volume)[0, 0, 0, 0].tolist() == [7.0, 6.0, 7.0, 6.0]
    assert build("/layer/torch/replication_pad3d", padding=1)(volume)[0, 0, 0, 0].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert build("/layer/torch/circular_pad3d", padding=1)(volume)[0, 0, 0, 0].tolist() == [7.0, 6.0, 7.0, 6.0]


def test_pools_reduce_windows_and_the_unpool_puts_the_maxima_back():
    grid = torch.tensor([[[[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]]])
    assert build("/layer/torch/maxpool", kernel=2)(grid).tolist() == [[[[4.0, 8.0], [0.0, 1.0]]]]
    assert build("/layer/torch/avgpool", kernel=2)(grid).tolist() == [[[[2.5, 6.5], [0.0, 1.0]]]]
    assert build("/layer/torch/maxpool", kernel=2, stride=1)(grid)[0, 0, 0].tolist() == [4.0, 7.0, 8.0]
    assert build("/layer/torch/maxpool", kernel=2, padding=1)(grid).shape == (1, 1, 3, 3)
    assert build("/layer/torch/adaptive_avgpool")(grid).tolist() == [[[[2.5]]]]
    assert build("/layer/torch/adaptive_maxpool", output_size=[1, 2])(grid).tolist() == [[[[4.0, 8.0]]]]
    assert build("/layer/torch/lppool", norm_type=2, kernel=2)(grid)[0, 0, 0].tolist() == pytest.approx(
        [math.sqrt(30.0), math.sqrt(174.0)])
    line = torch.tensor([[[1.0, 3.0, 2.0, 2.0]]])
    assert build("/layer/torch/maxpool1d", kernel=2)(line).tolist() == [[[3.0, 2.0]]]
    assert build("/layer/torch/avgpool1d", kernel=2)(line).tolist() == [[[2.0, 2.0]]]
    assert build("/layer/torch/adaptive_avgpool1d", output_size=1)(line).tolist() == [[[2.0]]]
    assert build("/layer/torch/adaptive_maxpool1d", output_size=2)(line).tolist() == [[[3.0, 2.0]]]
    assert build("/layer/torch/lppool1d", norm_type=1, kernel=2)(line).tolist() == [[[4.0, 4.0]]]
    cube = torch.arange(8.0).reshape(1, 1, 2, 2, 2)
    assert build("/layer/torch/maxpool3d", kernel=2)(cube).tolist() == [[[[[7.0]]]]]
    assert build("/layer/torch/avgpool3d", kernel=2)(cube).tolist() == [[[[[3.5]]]]]
    assert build("/layer/torch/adaptive_avgpool3d")(cube).tolist() == [[[[[3.5]]]]]
    assert build("/layer/torch/adaptive_maxpool3d")(cube).tolist() == [[[[[7.0]]]]]
    assert build("/layer/torch/lppool3d", norm_type=1, kernel=2)(cube).tolist() == [[[[[28.0]]]]]
    maxima, indices = nn.MaxPool2d(2, return_indices=True)(grid)
    restored = build("/layer/torch/max_unpool", kernel=2)(maxima, indices)
    assert restored.tolist() == [[[[0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 8.0], [0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0]]]]
    maxima, indices = nn.MaxPool1d(2, return_indices=True)(line)
    assert build("/layer/torch/max_unpool1d", kernel=2)(maxima, indices).tolist() == [[[0.0, 3.0, 2.0, 0.0]]]
    maxima, indices = nn.MaxPool3d(2, return_indices=True)(cube)
    assert build("/layer/torch/max_unpool3d", kernel=2)(maxima, indices).reshape(-1).tolist() == [0.0] * 7 + [7.0]
    fractional = build("/layer/torch/fractional_maxpool", kernel=2, output_ratio=0.5)
    assert fractional(floats(1, 1, 8, 8)).shape == (1, 1, 4, 4)
    assert build("/layer/torch/fractional_maxpool3d", kernel=2, output_size=3)(floats(1, 1, 8, 8, 8)).shape == \
        (1, 1, 3, 3, 3)


def test_recurrent_layers_take_the_input_width_from_the_first_batch():
    for uri, core_class in (("/layer/torch/gru", nn.GRU), ("/layer/torch/lstm", nn.LSTM), ("/layer/torch/rnn", nn.RNN)):
        layer = build(uri, hidden=6, layers=2, dropout=0.1)
        assert isinstance(layer, Recurrent) and isinstance(layer, LazyLayer) and layer.core is None
        assert tuple(layer(floats(2, 7, 5)).shape) == (2, 7, 6)
        assert type(layer.core) is core_class and layer.core.input_size == 5 and layer.core.num_layers == 2
        assert layer.core.dropout == 0.1 and layer.core.batch_first and not layer.core.bidirectional
    both = build("/layer/torch/gru", hidden=6, bidirectional=True)
    assert tuple(both(floats(2, 7, 5, seed=1).double()).shape) == (2, 7, 12)
    assert both.core.bidirectional and both.core.weight_ih_l0.dtype == torch.float64
    relu = build("/layer/torch/rnn", hidden=3, nonlinearity="relu")
    assert (relu(floats(2, 4, 5)) >= 0.0).all() and relu.core.nonlinearity == "relu"
    assert relu.extra == {"nonlinearity": "relu"}
    gru_cell = build("/layer/torch/gru_cell", input_size=5, hidden=6, bias=False)
    assert type(gru_cell) is nn.GRUCell and gru_cell.bias_ih is None
    assert tuple(gru_cell(floats(2, 5), torch.zeros(2, 6)).shape) == (2, 6)
    lstm_cell = build("/layer/torch/lstm_cell", input_size=5, hidden=6)
    hidden, cell = lstm_cell(floats(2, 5), (torch.zeros(2, 6), torch.zeros(2, 6)))
    assert type(lstm_cell) is nn.LSTMCell and tuple(hidden.shape) == (2, 6) and tuple(cell.shape) == (2, 6)
    rnn_cell = build("/layer/torch/rnn_cell", input_size=5, hidden=6)
    assert type(rnn_cell) is nn.RNNCell and rnn_cell.nonlinearity == "tanh"
    assert tuple(rnn_cell(floats(2, 5), torch.zeros(2, 6)).shape) == (2, 6)


def test_attention_layers_are_batch_first_and_count_their_layers():
    attention = build("/layer/torch/multihead_attention", embed_dim=8, heads=2, dropout=0.1, bias=False)
    assert attention.batch_first and attention.num_heads == 2 and attention.in_proj_bias is None
    query = floats(2, 7, 8)
    attended, weights = attention.eval()(query, query, query)
    assert weights.sum(dim=-1).reshape(-1).tolist() == pytest.approx([1.0] * 14, abs=1e-5)
    encoder = build("/layer/torch/transformer_encoder", d_model=8, heads=2, layers=3, dim_feedforward=16,
                    norm_first=True)
    assert encoder.num_layers == 3 and encoder.layers[0].norm_first and encoder.layers[0].self_attn.batch_first
    assert encoder.layers[0].linear1.out_features == 16
    layer = build("/layer/torch/transformer_encoder_layer", d_model=8, heads=2, dim_feedforward=16, activation="gelu")
    assert type(layer.activation) is not type(encoder.layers[0].activation)
    decoder = build("/layer/torch/transformer_decoder", d_model=8, heads=2, layers=2, dim_feedforward=16)
    assert decoder.num_layers == 2 and decoder.layers[1].multihead_attn.batch_first
    single = build("/layer/torch/transformer_decoder_layer", d_model=8, heads=2, dim_feedforward=16)
    assert single.self_attn.batch_first and single.linear1.out_features == 16
    whole = build("/layer/torch/transformer", d_model=8, heads=2, encoder_layers=1, decoder_layers=2,
                  dim_feedforward=16)
    assert whole.batch_first and whole.encoder.num_layers == 1 and whole.decoder.num_layers == 2


def test_shape_layers_rearrange_without_changing_the_values():
    first, second = torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0]])
    assert build("/layer/torch/concat")(first, second).tolist() == [[1.0, 2.0, 3.0]]
    assert build("/layer/torch/concat", dim=0)(first, first).tolist() == [[1.0, 2.0], [1.0, 2.0]]
    steps = floats(2, 7, 5)
    assert torch.equal(build("/layer/torch/last_step")(steps), steps[:, -1, :])
    image = torch.arange(48.0).reshape(2, 3, 4, 2)
    flat = build("/layer/torch/flatten")(image)
    assert torch.equal(flat, image.reshape(2, 24))
    assert torch.equal(build("/layer/torch/unflatten", dim=1, size=[3, 8])(flat), image.reshape(2, 3, 8))
    blocks = build("/layer/torch/unfold", kernel=2, stride=2)(image)
    assert tuple(blocks.shape) == (2, 12, 2)
    assert torch.equal(build("/layer/torch/fold", output_size=[4, 2], kernel=2, stride=2)(blocks), image)
    shuffled = build("/layer/torch/pixel_shuffle", factor=2)(torch.arange(16.0).reshape(1, 4, 2, 2))
    assert shuffled[0, 0, 0].tolist() == [0.0, 4.0, 1.0, 5.0]
    unshuffled = build("/layer/torch/pixel_unshuffle", factor=2)(shuffled)
    assert torch.equal(unshuffled, torch.arange(16.0).reshape(1, 4, 2, 2))
    channels = torch.arange(6.0).reshape(1, 6, 1, 1)
    assert build("/layer/torch/channel_shuffle", groups=2)(channels).reshape(-1).tolist() == [0.0, 3.0, 1.0, 4.0, 2.0,
                                                                                               5.0]
    nearest = build("/layer/torch/upsample", scale_factor=2)(torch.tensor([[[[1.0, 2.0]]]]))
    assert nearest.tolist() == [[[[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]]]
    linear = build("/layer/torch/upsample", size=3, mode="linear", align_corners=True)(torch.tensor([[[0.0, 2.0]]]))
    assert linear.tolist() == [[[0.0, 1.0, 2.0]]]


def test_linear_is_lazy_without_in_features():
    lazy = build("/layer/kalfa/linear", out_features=4)
    assert type(lazy) is nn.LazyLinear and lazy.out_features == 4
    assert tuple(lazy(floats(2, 3)).shape) == (2, 4)
    assert type(lazy) is nn.Linear and tuple(lazy.weight.shape) == (4, 3)
    fixed = build("/layer/kalfa/linear", out_features=4, in_features=3)
    assert type(fixed) is nn.Linear and tuple(fixed.weight.shape) == (4, 3) and tuple(fixed.bias.shape) == (4,)
    x = floats(2, 3)
    assert torch.allclose(fixed(x), x @ fixed.weight.T + fixed.bias)


def test_linear_relu_stacks_a_linear_layer_and_a_relu():
    block = build("/layer/kalfa/linear_relu", out_features=4)
    assert type(block) is nn.Sequential and [type(part) for part in block] == [nn.LazyLinear, nn.ReLU]
    x = floats(3, 5)
    out = block(x)
    assert tuple(out.shape) == (3, 4) and bool((out >= 0.0).all())
    assert torch.equal(out, torch.relu(block[0](x)))
    fixed = build("/layer/kalfa/linear_relu", out_features=4, in_features=5)
    assert [type(part) for part in fixed] == [nn.Linear, nn.ReLU] and tuple(fixed[0].weight.shape) == (4, 5)


def test_mlp_is_a_linear_activation_dropout_stack_per_width():
    block = build("/layer/kalfa/mlp", widths=[8, 4], activation="gelu", dropout=0.1, out_features=1)
    assert [type(part).__name__ for part in block] == ["LazyLinear", "GELU", "Dropout", "LazyLinear", "GELU",
                                                       "Dropout", "LazyLinear"]
    assert block[2].p == 0.1 and block[0].out_features == 8 and block[3].out_features == 4
    assert tuple(block(floats(3, 5)).shape) == (3, 1) and tuple(block[6].weight.shape) == (1, 4)
    plain = build("/layer/kalfa/mlp", widths=[8], in_features=5)
    assert [type(part).__name__ for part in plain] == ["Linear", "ReLU"] and tuple(plain[0].weight.shape) == (8, 5)
    assert tuple(plain(floats(3, 5)).shape) == (3, 8)
    head = build("/layer/kalfa/mlp", widths=[], out_features=2, in_features=5)
    assert [type(part).__name__ for part in head] == ["Linear"] and tuple(head[0].weight.shape) == (2, 5)
    hidden = build("/layer/kalfa/mlp", widths=[6, 6], in_features=5, dropout=0.0)
    assert [type(part).__name__ for part in hidden] == ["Linear", "ReLU", "LazyLinear", "ReLU"]
    assert hidden[2].out_features == 6 and tuple(hidden(floats(3, 5)).shape) == (3, 6)
    assert tuple(hidden[2].weight.shape) == (6, 6)


@pytest.mark.parametrize("name, kind", [("relu", nn.ReLU), ("gelu", nn.GELU), ("silu", nn.SiLU), ("tanh", nn.Tanh),
                                        ("leaky_relu", nn.LeakyReLU), ("elu", nn.ELU), ("selu", nn.SELU),
                                        ("mish", nn.Mish), ("sigmoid", nn.Sigmoid)])
def test_mlp_activation_names_its_layer(name, kind):
    block = build("/layer/kalfa/mlp", widths=[4], activation=name)
    assert type(block[1]) is kind


def test_mlp_refuses_an_unknown_activation():
    expected = ("mlp.activation must be one of ['elu', 'gelu', 'leaky_relu', 'mish', 'relu', 'selu', 'sigmoid', "
                "'silu', 'tanh'], got 'swish'")
    with pytest.raises(ValueError) as caught:
        build("/layer/kalfa/mlp", widths=[4], activation="swish")
    assert str(caught.value) == expected


def test_polynomial_expands_the_features_by_their_products():
    x = torch.tensor([[2.0, 3.0]])
    assert build("/layer/kalfa/polynomial")(x).tolist() == [[2.0, 3.0, 4.0, 6.0, 9.0]]
    assert build("/layer/kalfa/polynomial", interaction_only=True)(x).tolist() == [[2.0, 3.0, 6.0]]
    assert build("/layer/kalfa/polynomial", bias=True)(x).tolist() == [[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]]
    assert build("/layer/kalfa/polynomial", keep=False)(x).tolist() == [[4.0, 6.0, 9.0]]
    assert build("/layer/kalfa/polynomial", keep=False, bias=True)(x).tolist() == [[1.0, 4.0, 6.0, 9.0]]
    assert build("/layer/kalfa/polynomial", degree=3)(x).tolist() == \
        [[2.0, 3.0, 4.0, 6.0, 9.0, 8.0, 12.0, 18.0, 27.0]]
    assert build("/layer/kalfa/polynomial", degree=3, interaction_only=True)(x).tolist() == [[2.0, 3.0, 6.0]]
    wide = torch.tensor([[1.0, 2.0, 3.0]])
    assert build("/layer/kalfa/polynomial", degree=3, interaction_only=True)(wide).tolist() == \
        [[1.0, 2.0, 3.0, 2.0, 3.0, 6.0, 6.0]]
    assert build("/layer/kalfa/polynomial")(x.reshape(1, 2, 1)).tolist() == [[2.0, 3.0, 4.0, 6.0, 9.0]]
    two = build("/layer/kalfa/polynomial")(torch.tensor([[2.0, 3.0], [1.0, -1.0]]))
    assert two.tolist() == [[2.0, 3.0, 4.0, 6.0, 9.0], [1.0, -1.0, 1.0, -1.0, 1.0]]


def test_polynomial_needs_a_degree_of_two_or_more():
    with pytest.raises(ValueError, match=r"^degree must be 2 or more, got 1$"):
        build("/layer/kalfa/polynomial", degree=1)


def test_l2_normalize_divides_every_sample_by_its_own_norm():
    layer = build("/layer/kalfa/l2_normalize")
    out = layer(torch.tensor([[3.0, 4.0], [0.0, 0.0], [-1.0, 0.0]]))
    assert out.reshape(-1).tolist() == pytest.approx([0.6, 0.8, 0.0, 0.0, -1.0, 0.0])
    assert layer.eps == 1e-12
    flat = layer(torch.ones(2, 2, 2))
    assert tuple(flat.shape) == (2, 4) and flat.reshape(-1).tolist() == pytest.approx([0.5] * 8)
    assert build("/layer/kalfa/l2_normalize", eps=10.0)(torch.tensor([[3.0, 4.0]])).tolist()[0] == \
        pytest.approx([0.3, 0.4])


def test_select_picks_the_positions_of_the_feature_axis_in_the_order_written():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer = build("/layer/kalfa/select", index=[2, 0])
    assert layer(x).tolist() == [[3.0, 1.0], [6.0, 4.0]]
    assert layer.index.dtype == torch.long and layer.dim == 1
    assert list(layer.state_dict()) == ["index"] and layer.state_dict()["index"].tolist() == [2, 0]
    assert build("/layer/kalfa/select", index=[1], dim=0)(x).tolist() == [[4.0, 5.0, 6.0]]
    assert build("/layer/kalfa/select", index=[1, 1])(x).tolist() == [[2.0, 2.0], [5.0, 5.0]]


def test_select_takes_a_feature_index_component_built_from_the_train_loader():
    deferred = Deferred("/data/kalfa/feature_index", {"columns": ["x2", "x0"]},
                        registry.resolve("/data/kalfa/feature_index"))
    layer = build("/layer/kalfa/select", index=deferred)
    assert isinstance(layer, DeferredLayer) and layer.params == {"index": deferred, "dim": 1}
    loader = SimpleNamespace(dataset=SimpleNamespace(frame=SimpleNamespace(features=["x0", "x1", "x2"])))
    built = layer.build(loader=loader, prep=None)
    assert built.index.tolist() == [2, 0]
    assert built(torch.tensor([[1.0, 2.0, 3.0]])).tolist() == [[3.0, 1.0]]


def test_wires_combine_elementwise_and_broadcast():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    row = torch.tensor([10.0, 20.0, 30.0])
    assert build("/layer/kalfa/add")(x, row).tolist() == [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]
    assert build("/layer/kalfa/add")(x, x, x).tolist() == [[3.0, 6.0, 9.0], [12.0, 15.0, 18.0]]
    assert torch.equal(build("/layer/kalfa/add")(x), x)
    assert build("/layer/kalfa/subtract")(x, row).tolist() == [[-9.0, -18.0, -27.0], [-6.0, -15.0, -24.0]]
    assert build("/layer/kalfa/subtract")(torch.tensor([[1.0], [2.0]]), torch.tensor([[1.0, 2.0, 3.0]])).tolist() == \
        [[0.0, -1.0, -2.0], [1.0, 0.0, -1.0]]
    gate = torch.tensor([[0.0, 1.0, 0.0]])
    assert build("/layer/kalfa/multiply")(x, gate).tolist() == [[0.0, 2.0, 0.0], [0.0, 5.0, 0.0]]
    assert build("/layer/kalfa/multiply")(x, gate, torch.tensor(2.0)).tolist() == [[0.0, 4.0, 0.0], [0.0, 10.0, 0.0]]
    assert build("/layer/kalfa/divide")(x, torch.tensor([[2.0], [4.0]])).tolist() == [[0.5, 1.0, 1.5],
                                                                                        [1.0, 1.25, 1.5]]
    assert build("/layer/kalfa/negate")(x).tolist() == [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]


def test_divide_by_zero_gives_an_infinity_as_torch_does():
    out = build("/layer/kalfa/divide")(torch.tensor([[1.0, -1.0, 0.0]]), torch.zeros(1, 3))
    assert out[0, 0].item() == math.inf and out[0, 1].item() == -math.inf and math.isnan(out[0, 2].item())


def test_multipliers_start_every_lambda_at_its_entry_and_ignore_their_input():
    constraints = {"mae_lin": {"epsilon": 0.5, "lmbda_init": 0.25, "scale": 1.0}, "ws.huber_hv": {"epsilon": 0.3}}
    layer = build("/layer/kalfa/multipliers", names=constraints)
    assert layer.names == ["mae_lin", "ws.huber_hv"]
    assert isinstance(layer.lmbda, nn.Parameter) and layer.lmbda.dtype == torch.float32 and layer.lmbda.requires_grad
    assert layer.lmbda.tolist() == [0.25, 0.0]
    assert [name for name, _ in layer.named_parameters()] == ["lmbda"]
    assert layer(floats(4, 3)) is layer.lmbda and layer() is layer.lmbda
    assert build("/layer/kalfa/multipliers", names=["p", "q", "r"], init=1.5).lmbda.tolist() == [1.5, 1.5, 1.5]
    mixed = build("/layer/kalfa/multipliers", names={"a": None, "b": {"lmbda_init": -2}}, init=3.0)
    assert mixed.names == ["a", "b"] and mixed.lmbda.tolist() == [3.0, -2.0]


@pytest.mark.parametrize("names", [{}, [], None], ids=["mapping", "list", "none"])
def test_multipliers_need_at_least_one_name(names):
    expected = "multipliers needs names: the constraints mapping of the mdmm loss, or a list of names"
    with pytest.raises(ValueError) as caught:
        build("/layer/kalfa/multipliers", names=names)
    assert str(caught.value) == expected


def test_l1_distance_is_the_mean_absolute_difference_per_sample():
    layer = build("/layer/kalfa/l1_distance")
    first = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    second = torch.tensor([[2.0, 2.0, 5.0], [1.0, -1.0, 4.0]])
    assert layer(first, second).tolist() == [1.0, 2.0]
    assert layer(torch.ones(2, 2, 2), torch.zeros(2, 2, 2)).tolist() == [1.0, 1.0]


def test_reparam_samples_in_train_mode_and_returns_mu_in_eval_mode():
    layer = build("/layer/kalfa/reparam")
    mu = torch.tensor([[1.0, -1.0]])
    logvar = torch.tensor([[0.0, math.log(4.0)]])
    assert layer.eval()(mu, logvar) is mu
    torch.manual_seed(3)
    noise = torch.randn(1, 2)
    torch.manual_seed(3)
    sampled = layer.train()(mu, logvar)
    assert torch.equal(sampled, mu + torch.tensor([[1.0, 2.0]]) * noise)
    assert not torch.equal(layer(mu, logvar), sampled)
    assert torch.equal(layer(mu, torch.full((1, 2), -math.inf)), mu)


def test_unflatten_reshapes_the_features_of_every_sample():
    layer = build("/layer/kalfa/unflatten", shape=[3, 4])
    assert type(layer) is nn.Unflatten and layer.dim == 1 and layer.unflattened_size == (3, 4)
    x = torch.arange(24.0).reshape(2, 12)
    out = layer(x)
    assert tuple(out.shape) == (2, 3, 4) and torch.equal(out.reshape(2, 12), x)
    assert out[1, 2].tolist() == [20.0, 21.0, 22.0, 23.0]
