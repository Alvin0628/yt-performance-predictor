"""Joint Transformer encoder for multimodal sequence."""
import torch.nn as nn


class JointTransformer(nn.Module):
    def __init__(self, d, nhead, num_layers, dim_feedforward=None, dropout=0.2):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d * 4
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,   # Pre-LN, stable on small data
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers,
            enable_nested_tensor=False,   # avoid warning & not supported with mask
        )

    def forward(self, x, padding_mask=None):
        return self.encoder(x, src_key_padding_mask=padding_mask)