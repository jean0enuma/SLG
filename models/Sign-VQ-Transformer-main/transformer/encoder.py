from typing import Tuple
from torch import Tensor, nn

from helpers import freeze_params
from transformer.transformer_layers import PositionalEncoding, TransformerEncoderLayer


class Encoder(nn.Module):
    """
    Base encoder class
    """

    # pylint: disable=abstract-method
    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size


class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        **kwargs,
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super().__init__()

        self._output_size = hidden_size

        # build all (num_layers) layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    alpha=kwargs.get("alpha", 1.0),
                    layer_norm=kwargs.get("layer_norm", "pre"),
                    activation=kwargs.get("activation", "relu"),
                )
                for _ in range(num_layers)
            ]
        )

        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.layer_norm = (
            nn.LayerNorm(hidden_size, eps=1e-6)
            if kwargs.get("layer_norm", "post") == "pre"
            else None
        )

        if freeze:
            freeze_params(self)

    def forward(
        self, src_embed: Tensor, src_length: Tensor, mask: Tensor = None  # unused
    ) -> Tuple[Tensor, Tensor]:
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param src_embed: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, 1, src_len)
        :param kwargs:
        :return:
            - output: hidden states with shape (batch_size, max_length, hidden)
            - None
        """
        # pylint: disable=unused-argument
        x = self.pe(src_embed)  # add position encoding to word embeddings
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_layers={len(self.layers)}, "
            f"num_heads={self.layers[0].src_src_att.num_heads}, "
            f"alpha={self.layers[0].alpha}, "
            f'layer_norm="{self.layers[0]._layer_norm_position}", '
            f"activation={self.layers[0].feed_forward.pwff_layer[1]})"
        )
