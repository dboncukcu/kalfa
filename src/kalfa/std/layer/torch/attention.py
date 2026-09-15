from torch import nn


def multihead_attention(embed_dim, heads, dropout=0.0, bias=True, batch_first=True):
    return nn.MultiheadAttention(int(embed_dim), int(heads), dropout=float(dropout), bias=bool(bias),
                                 batch_first=bool(batch_first))


def transformer_encoder_layer(d_model, heads, dim_feedforward=2048, dropout=0.1, activation="relu", norm_first=False,
                              batch_first=True):
    return nn.TransformerEncoderLayer(int(d_model), int(heads), dim_feedforward=int(dim_feedforward),
                                      dropout=float(dropout), activation=activation, norm_first=bool(norm_first),
                                      batch_first=bool(batch_first))


def transformer_encoder(d_model, heads, layers, dim_feedforward=2048, dropout=0.1, activation="relu",
                        norm_first=False, batch_first=True):
    layer = transformer_encoder_layer(d_model, heads, dim_feedforward, dropout, activation, norm_first, batch_first)
    return nn.TransformerEncoder(layer, num_layers=int(layers), enable_nested_tensor=False)


def transformer_decoder_layer(d_model, heads, dim_feedforward=2048, dropout=0.1, activation="relu", norm_first=False,
                              batch_first=True):
    return nn.TransformerDecoderLayer(int(d_model), int(heads), dim_feedforward=int(dim_feedforward),
                                      dropout=float(dropout), activation=activation, norm_first=bool(norm_first),
                                      batch_first=bool(batch_first))


def transformer_decoder(d_model, heads, layers, dim_feedforward=2048, dropout=0.1, activation="relu",
                        norm_first=False, batch_first=True):
    layer = transformer_decoder_layer(d_model, heads, dim_feedforward, dropout, activation, norm_first, batch_first)
    return nn.TransformerDecoder(layer, num_layers=int(layers))


def transformer(d_model=512, heads=8, encoder_layers=6, decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                activation="relu", norm_first=False, batch_first=True):
    return nn.Transformer(int(d_model), int(heads), int(encoder_layers), int(decoder_layers),
                          dim_feedforward=int(dim_feedforward), dropout=float(dropout), activation=activation,
                          norm_first=bool(norm_first), batch_first=bool(batch_first))
