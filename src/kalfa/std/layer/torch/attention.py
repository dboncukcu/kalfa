from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/multihead_attention", alias="multihead_attention",
      description="torch.nn.MultiheadAttention over (batch, steps, embed_dim); inputs query, key and value, "
                  "outputs the attended values and the weights")
def multihead_attention(embed_dim, heads, dropout=0.0, bias=True, batch_first=True):
    return nn.MultiheadAttention(int(embed_dim), int(heads), dropout=float(dropout), bias=bool(bias),
                                 batch_first=bool(batch_first))


@lego("/layer/torch/transformer_encoder_layer", alias="transformer_encoder_layer",
      description="torch.nn.TransformerEncoderLayer over (batch, steps, d_model)")
def transformer_encoder_layer(d_model, heads, dim_feedforward=2048, dropout=0.1, activation="relu", norm_first=False,
                              batch_first=True):
    return nn.TransformerEncoderLayer(int(d_model), int(heads), dim_feedforward=int(dim_feedforward),
                                      dropout=float(dropout), activation=activation, norm_first=bool(norm_first),
                                      batch_first=bool(batch_first))


@lego("/layer/torch/transformer_encoder", alias="transformer_encoder",
      description="torch.nn.TransformerEncoder of layers encoder layers over (batch, steps, d_model)")
def transformer_encoder(d_model, heads, layers, dim_feedforward=2048, dropout=0.1, activation="relu",
                        norm_first=False, batch_first=True):
    layer = transformer_encoder_layer(d_model, heads, dim_feedforward, dropout, activation, norm_first, batch_first)
    return nn.TransformerEncoder(layer, num_layers=int(layers), enable_nested_tensor=False)


@lego("/layer/torch/transformer_decoder_layer", alias="transformer_decoder_layer",
      description="torch.nn.TransformerDecoderLayer; inputs the target sequence and the memory")
def transformer_decoder_layer(d_model, heads, dim_feedforward=2048, dropout=0.1, activation="relu", norm_first=False,
                              batch_first=True):
    return nn.TransformerDecoderLayer(int(d_model), int(heads), dim_feedforward=int(dim_feedforward),
                                      dropout=float(dropout), activation=activation, norm_first=bool(norm_first),
                                      batch_first=bool(batch_first))


@lego("/layer/torch/transformer_decoder", alias="transformer_decoder",
      description="torch.nn.TransformerDecoder of layers decoder layers; inputs the target sequence and the memory")
def transformer_decoder(d_model, heads, layers, dim_feedforward=2048, dropout=0.1, activation="relu",
                        norm_first=False, batch_first=True):
    layer = transformer_decoder_layer(d_model, heads, dim_feedforward, dropout, activation, norm_first, batch_first)
    return nn.TransformerDecoder(layer, num_layers=int(layers))


@lego("/layer/torch/transformer", alias="transformer",
      description="torch.nn.Transformer, the encoder and the decoder; inputs the source and the target sequences")
def transformer(d_model=512, heads=8, encoder_layers=6, decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                activation="relu", norm_first=False, batch_first=True):
    return nn.Transformer(int(d_model), int(heads), int(encoder_layers), int(decoder_layers),
                          dim_feedforward=int(dim_feedforward), dropout=float(dropout), activation=activation,
                          norm_first=bool(norm_first), batch_first=bool(batch_first))
