import copy
import torch

class EMA(torch.nn.Module):
    """モデル重みの指数移動平均を保持する.
    - update(): 毎step, optimizer.step() の後に呼ぶ
    - ema_model: 評価・保存・生成に使う側 (勾配不要)
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999,
                 warmup_steps: int = 10):
        super().__init__()
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def _decay_now(self) -> float:
        # 序盤は decay を小さくして生の重みに素早く追従させる
        return min(self.decay, (1 + self.step) / (self.warmup_steps + self.step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.step += 1
        d = self._decay_now()
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            if p.requires_grad:
                ema_p.lerp_(p.detach(), 1.0 - d)   # ema = d*ema + (1-d)*p
            else:
                ema_p.copy_(p)                      # 凍結VAEはそのままコピー
        # BatchNorm統計等のbufferは平均せずコピーが安全
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)