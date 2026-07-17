from torch import nn
class AdaLN(nn.Module):
    """
    LayerNorm + time conditioning
    x: (B, T, D)
    t_emb: (B, time_dim)
    """
    def __init__(self, d_model, time_dim):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, d_model * 3)
        )
        # zero-init so scale=shift=gate=0 at start → stable identity at init
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(self,t_emb):
        scale, shift,gate = self.to_scale_shift(t_emb).chunk(3, dim=-1)
        scale = scale.unsqueeze(1)   # (B,1,D)
        shift = shift.unsqueeze(1)   # (B,1,D)
        gate=gate.unsqueeze(1)     # (B,1,D)
        return scale, shift,gate