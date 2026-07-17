from torch import Tensor, nn

from helpers import freeze_params, subsequent_mask
from transformer.transformer_layers import PositionalEncoding, TransformerDecoderLayer


class Decoder(nn.Module):
    """
    Base decoder class
    """

    # pylint: disable=abstract-method

    @property
    def output_size(self):
        """
        Return the output size (size of the target vocabulary)

        :return:
        """
        return self._output_size


class TransformerDecoder(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_size: int = 512,
        ff_size: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        output_size: int = 1,
        freeze: bool = False,
        **kwargs,
    ):
        """
        Initialize a Transformer decoder.

        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param output_size: size of the output
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs:
        """
        super().__init__()

        self._hidden_size = hidden_size
        self._output_size = output_size

        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    alpha=kwargs.get("alpha", 1.0),
                    layer_norm=kwargs.get("layer_norm", "post"),
                    activation=kwargs.get("activation", "relu"),
                )
                for _ in range(num_layers)
            ]
        )

        self.pe = PositionalEncoding(hidden_size)
        self.layer_norm = (
            nn.LayerNorm(hidden_size, eps=1e-6)
            if kwargs.get("layer_norm", "post") == "pre"
            else None
        )

        self.emb_dropout = nn.Dropout(p=emb_dropout)

        if freeze:
            freeze_params(self)

    def forward(
        self,
        trg_embed: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor,
        trg_mask: Tensor,
        return_attention: bool = False,
    ):
        """
        Transformer decoder forward pass.

        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param src_mask:
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :param return_attention:
        :return:
            - decoder_output: shape (batch_size, seq_len, vocab_size)
            - decoder_hidden: shape (batch_size, seq_len, emb_size)
            - att_probs: shape (batch_size, trg_length, src_length),
            - None
        """
        assert trg_mask is not None, "trg_mask required for Transformer"

        x = self.pe(trg_embed)  # add position encoding to word embedding
        x = self.emb_dropout(x)

        trg_mask = trg_mask & subsequent_mask(trg_embed.size(1)).type_as(trg_mask)

        last_layer = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x, att = layer(
                x=x,
                memory=encoder_output,
                src_mask=src_mask,
                trg_mask=trg_mask,
                return_attention=(return_attention and i == last_layer),
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x, att

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_layers={len(self.layers)}, "
            f"num_heads={self.layers[0].trg_trg_att.num_heads}, "
            f"alpha={self.layers[0].alpha}, "
            f'layer_norm="{self.layers[0]._layer_norm_position}", '
            f"activation={self.layers[0].feed_forward.pwff_layer[1]})"
        )
