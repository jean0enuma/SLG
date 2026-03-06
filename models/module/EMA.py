import torch
class EMA:
    def __init__(self, model, beta=0.999):
        self.beta = beta
        self.model = model
        self.ema_model = self._clone_model()

    def _clone_model(self):
        ema = type(self.model)(*[])  # 簡略化。実際は deepcopy 推奨
        ema.load_state_dict(self.model.state_dict())
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self):
        for ema_p, p in zip(self.ema_model.parameters(),
                            self.model.parameters()):
            ema_p.data = self.beta * ema_p.data + (1 - self.beta) * p.data