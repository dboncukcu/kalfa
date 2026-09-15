"""The torch layer catalog and kalfa's blocks build and run on a batch."""

import pytest
import torch

import kalfa  # noqa: F401
from cirak.registry import registry
from kalfa.std import STD_URIS
from kalfa.std.common.deferred import LazyLayer
from kalfa.std.layer.kalfa.blocks import mlp
from kalfa.std.layer.torch.recurrent import Recurrent


def build(uri, **params):
    return registry.resolve(uri)(**params)


def run(uri, shape, **params):
    layer = build(uri, **params)
    return layer(torch.randn(*shape))


ACTIVATIONS = ["relu", "relu6", "leaky_relu", "prelu", "rrelu", "elu", "celu", "selu", "gelu", "silu", "mish",
               "hardswish", "hardsigmoid", "hardtanh", "hardshrink", "softshrink", "tanhshrink", "softsign", "softplus",
               "sigmoid", "log_sigmoid", "tanh", "softmax", "softmin", "log_softmax"]


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_every_activation_keeps_the_shape(name):
    assert run(f"/layer/torch/{name}", (2, 6)).shape == (2, 6)


def test_the_special_activations():
    assert run("/layer/torch/glu", (2, 6)).shape == (2, 3)
    assert run("/layer/torch/threshold", (2, 6), threshold=0.1, value=20.0).shape == (2, 6)
    assert run("/layer/torch/softmax2d", (2, 3, 4, 4)).shape == (2, 3, 4, 4)
    assert run("/layer/torch/identity", (2, 6)).shape == (2, 6)


def test_the_normalizations_are_lazy_in_the_width():
    assert run("/layer/torch/batch_norm", (4, 6)).shape == (4, 6)
    assert run("/layer/torch/batch_norm", (4, 3, 5, 5), dims=2).shape == (4, 3, 5, 5)
    assert run("/layer/torch/instance_norm", (4, 3, 5, 5)).shape == (4, 3, 5, 5)
    assert run("/layer/torch/instance_norm", (4, 3, 7), dims=1).shape == (4, 3, 7)
    assert run("/layer/torch/layer_norm", (4, 6), normalized_shape=6).shape == (4, 6)
    assert run("/layer/torch/rms_norm", (4, 6), normalized_shape=6).shape == (4, 6)
    assert run("/layer/torch/group_norm", (4, 6, 5, 5), num_groups=2, num_channels=6).shape == (4, 6, 5, 5)
    assert run("/layer/torch/local_response_norm", (4, 6, 5, 5), size=2).shape == (4, 6, 5, 5)
    with pytest.raises(ValueError, match="dims"):
        build("/layer/torch/batch_norm", dims=4)


def test_the_dropouts_and_the_pools():
    for name in ("dropout", "alpha_dropout"):
        assert run(f"/layer/torch/{name}", (2, 6), p=0.2).shape == (2, 6)
    assert run("/layer/torch/dropout1d", (2, 3, 7)).shape == (2, 3, 7)
    assert run("/layer/torch/dropout2d", (2, 3, 5, 5)).shape == (2, 3, 5, 5)
    assert run("/layer/torch/dropout3d", (2, 3, 4, 4, 4)).shape == (2, 3, 4, 4, 4)
    assert run("/layer/torch/feature_alpha_dropout", (2, 3, 5, 5)).shape == (2, 3, 5, 5)
    assert run("/layer/torch/maxpool", (2, 3, 8, 8), kernel=2).shape == (2, 3, 4, 4)
    assert run("/layer/torch/maxpool1d", (2, 3, 8), kernel=2).shape == (2, 3, 4)
    assert run("/layer/torch/maxpool3d", (2, 3, 8, 8, 8), kernel=2).shape == (2, 3, 4, 4, 4)
    assert run("/layer/torch/avgpool", (2, 3, 8, 8), kernel=2, stride=2).shape == (2, 3, 4, 4)
    assert run("/layer/torch/avgpool1d", (2, 3, 8), kernel=2).shape == (2, 3, 4)
    assert run("/layer/torch/avgpool3d", (2, 3, 8, 8, 8), kernel=2).shape == (2, 3, 4, 4, 4)
    assert run("/layer/torch/adaptive_avgpool", (2, 3, 8, 8)).shape == (2, 3, 1, 1)
    assert run("/layer/torch/adaptive_avgpool1d", (2, 3, 8), output_size=2).shape == (2, 3, 2)
    assert run("/layer/torch/adaptive_avgpool3d", (2, 3, 8, 8, 8)).shape == (2, 3, 1, 1, 1)
    assert run("/layer/torch/adaptive_maxpool", (2, 3, 8, 8), output_size=[2, 3]).shape == (2, 3, 2, 3)
    assert run("/layer/torch/adaptive_maxpool1d", (2, 3, 8)).shape == (2, 3, 1)
    assert run("/layer/torch/adaptive_maxpool3d", (2, 3, 8, 8, 8)).shape == (2, 3, 1, 1, 1)
    assert run("/layer/torch/lppool", (2, 3, 8, 8), norm_type=2, kernel=2).shape == (2, 3, 4, 4)
    assert run("/layer/torch/lppool1d", (2, 3, 8), norm_type=2, kernel=2).shape == (2, 3, 4)
    assert run("/layer/torch/lppool3d", (2, 3, 8, 8, 8), norm_type=2, kernel=2).shape == (2, 3, 4, 4, 4)
    assert run("/layer/torch/fractional_maxpool", (2, 3, 8, 8), kernel=2, output_size=5).shape == (2, 3, 5, 5)
    volume = run("/layer/torch/fractional_maxpool3d", (2, 3, 8, 8, 8), kernel=2, output_ratio=0.5)
    assert volume.shape == (2, 3, 4, 4, 4)
    pooled, indices = torch.nn.MaxPool2d(2, return_indices=True)(torch.randn(2, 3, 8, 8))
    assert build("/layer/torch/max_unpool", kernel=2)(pooled, indices).shape == (2, 3, 8, 8)
    pooled, indices = torch.nn.MaxPool1d(2, return_indices=True)(torch.randn(2, 3, 8))
    assert build("/layer/torch/max_unpool1d", kernel=2)(pooled, indices).shape == (2, 3, 8)
    pooled, indices = torch.nn.MaxPool3d(2, return_indices=True)(torch.randn(2, 3, 8, 8, 8))
    assert build("/layer/torch/max_unpool3d", kernel=2)(pooled, indices).shape == (2, 3, 8, 8, 8)


def test_the_pads():
    for name in ("zero_pad", "constant_pad", "reflection_pad", "replication_pad", "circular_pad"):
        assert run(f"/layer/torch/{name}", (2, 3, 4, 4), padding=1).shape == (2, 3, 6, 6)
        assert run(f"/layer/torch/{name}1d", (2, 3, 4), padding=[1, 2]).shape == (2, 3, 7)
        assert run(f"/layer/torch/{name}3d", (2, 3, 4, 4, 4), padding=1).shape == (2, 3, 6, 6, 6)
    assert float(run("/layer/torch/constant_pad", (1, 1, 2, 2), padding=1, value=7.0)[0, 0, 0, 0]) == 7.0


def test_the_convolutions_are_lazy_without_in_channels():
    assert run("/layer/torch/conv1d", (2, 3, 8), out_channels=4, kernel=3, padding=1).shape == (2, 4, 8)
    assert run("/layer/torch/conv2d", (2, 3, 8, 8), out_channels=4, kernel=3).shape == (2, 4, 6, 6)
    assert run("/layer/torch/conv2d", (2, 3, 8, 8), out_channels=4, kernel=3, in_channels=3).shape == (2, 4, 6, 6)
    assert run("/layer/torch/conv3d", (2, 3, 6, 6, 6), out_channels=4, kernel=3).shape == (2, 4, 4, 4, 4)
    assert run("/layer/torch/conv_transpose1d", (2, 3, 8), out_channels=4, kernel=2, stride=2).shape == (2, 4, 16)
    assert run("/layer/torch/conv_transpose2d", (2, 3, 8, 8), out_channels=4, kernel=2,
               stride=2).shape == (2, 4, 16, 16)
    assert run("/layer/torch/conv_transpose2d", (2, 3, 8, 8), out_channels=4, kernel=2, stride=2,
               in_channels=3).shape == (2, 4, 16, 16)
    assert run("/layer/torch/conv_transpose3d", (2, 3, 4, 4, 4), out_channels=4, kernel=2,
               stride=2).shape == (2, 4, 8, 8, 8)


def test_the_shapes():
    assert run("/layer/torch/flatten", (2, 3, 4, 4)).shape == (2, 48)
    assert run("/layer/torch/flatten", (2, 3, 4, 4), start_dim=2).shape == (2, 3, 16)
    assert run("/layer/torch/unflatten", (2, 12), dim=1, size=[3, 4]).shape == (2, 3, 4)
    assert build("/layer/torch/concat")(torch.zeros(2, 3), torch.ones(2, 4)).shape == (2, 7)
    assert run("/layer/torch/last_step", (2, 7, 5)).shape == (2, 5)
    blocks = run("/layer/torch/unfold", (2, 3, 4, 4), kernel=2)
    assert blocks.shape == (2, 12, 9)
    assert build("/layer/torch/fold", output_size=[4, 4], kernel=2)(blocks).shape == (2, 3, 4, 4)
    assert run("/layer/torch/pixel_shuffle", (2, 12, 4, 4), factor=2).shape == (2, 3, 8, 8)
    assert run("/layer/torch/pixel_unshuffle", (2, 3, 8, 8), factor=2).shape == (2, 12, 4, 4)
    assert run("/layer/torch/channel_shuffle", (2, 6, 4, 4), groups=2).shape == (2, 6, 4, 4)
    assert run("/layer/torch/upsample", (2, 3, 4, 4), scale_factor=2).shape == (2, 3, 8, 8)
    assert run("/layer/torch/upsample", (2, 3, 4, 4), size=[6, 6], mode="bilinear").shape == (2, 3, 6, 6)


def test_the_recurrent_layers_take_the_width_from_the_first_batch():
    for name in ("gru", "lstm", "rnn"):
        layer = build(f"/layer/torch/{name}", hidden=6, layers=2, dropout=0.1)
        assert isinstance(layer, Recurrent) and isinstance(layer, LazyLayer) and layer.core is None
        assert layer(torch.randn(2, 7, 5)).shape == (2, 7, 6)
        assert layer.core.input_size == 5 and layer.core.num_layers == 2
    both = build("/layer/torch/gru", hidden=6, bidirectional=True)
    assert both(torch.randn(2, 7, 5)).shape == (2, 7, 12)
    assert build("/layer/torch/rnn", hidden=6, nonlinearity="relu")(torch.randn(2, 7, 5)).shape == (2, 7, 6)
    hidden = build("/layer/torch/gru_cell", input_size=5, hidden=6)(torch.randn(2, 5), torch.zeros(2, 6))
    assert hidden.shape == (2, 6)
    hidden, cell = build("/layer/torch/lstm_cell", input_size=5, hidden=6)(torch.randn(2, 5))
    assert hidden.shape == (2, 6) and cell.shape == (2, 6)
    assert build("/layer/torch/rnn_cell", input_size=5, hidden=6)(torch.randn(2, 5)).shape == (2, 6)


def test_the_attention_layers():
    query = torch.randn(2, 7, 8)
    attended, weights = build("/layer/torch/multihead_attention", embed_dim=8, heads=2)(query, query, query)
    assert attended.shape == (2, 7, 8) and weights.shape == (2, 7, 7)
    encoder = build("/layer/torch/transformer_encoder", d_model=8, heads=2, layers=2, dim_feedforward=16)
    assert encoder(query).shape == (2, 7, 8) and encoder.num_layers == 2
    assert run("/layer/torch/transformer_encoder_layer", (2, 7, 8), d_model=8, heads=2, dim_feedforward=16,
               norm_first=True).shape == (2, 7, 8)
    memory = torch.randn(2, 5, 8)
    layer = build("/layer/torch/transformer_decoder_layer", d_model=8, heads=2, dim_feedforward=16)
    assert layer(query, memory).shape == (2, 7, 8)
    decoder = build("/layer/torch/transformer_decoder", d_model=8, heads=2, layers=2, dim_feedforward=16)
    assert decoder(query, memory).shape == (2, 7, 8)
    whole = build("/layer/torch/transformer", d_model=8, heads=2, encoder_layers=1, decoder_layers=1,
                  dim_feedforward=16)
    assert whole(memory, query).shape == (2, 7, 8)


def test_embeddings_distances_and_linear_layers():
    ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    assert build("/layer/torch/embedding", num=10, dim=4, padding_idx=0)(ids).shape == (2, 3, 4)
    assert build("/layer/torch/embedding_bag", num=10, dim=4, mode="sum")(ids).shape == (2, 4)
    assert build("/layer/torch/cosine_similarity")(torch.randn(2, 5), torch.randn(2, 5)).shape == (2,)
    assert build("/layer/torch/pairwise_distance", p=1)(torch.randn(2, 5), torch.randn(2, 5)).shape == (2,)
    assert run("/layer/torch/linear", (2, 5), in_features=5, out_features=3, bias=False).shape == (2, 3)
    assert build("/layer/torch/bilinear", in1_features=5, in2_features=4, out_features=3)(
        torch.randn(2, 5), torch.randn(2, 4)).shape == (2, 3)


def test_the_mlp_block_is_one_node_for_a_whole_perceptron():
    block = mlp([8, 4], activation="gelu", dropout=0.1, out_features=1)
    kinds = [type(part).__name__ for part in block]
    assert kinds == ["LazyLinear", "GELU", "Dropout", "LazyLinear", "GELU", "Dropout", "LazyLinear"]
    assert block(torch.randn(3, 5)).shape == (3, 1)
    plain = mlp([8], in_features=5)
    assert [type(part).__name__ for part in plain] == ["Linear", "ReLU"]
    assert plain(torch.randn(3, 5)).shape == (3, 8)
    assert mlp([], out_features=2, in_features=5)(torch.randn(3, 5)).shape == (3, 2)
    with pytest.raises(ValueError, match="mlp.activation"):
        mlp([8], activation="swish")


def test_every_layer_lego_has_an_alias_in_a_pack_table():
    from kalfa.config import pack_tables

    listed = {name for table in pack_tables().values() for name in table}
    for uri in STD_URIS:
        if not uri.startswith("/layer/"):
            continue
        for alias in registry.lookup(uri).facts.alias:
            assert alias in listed, (uri, alias)
