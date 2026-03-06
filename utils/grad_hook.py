# モデルの指定されたレイヤーの出力と勾配を保存するクラス
class SaveOutput:
    def __init__(self, model, target_layer):  # 引数：モデル, 対象のレイヤー
        self.model = model
        self.layer_output = []
        self.layer_grad = []

        # 特徴マップを取るためのregister_forward_hookを設定
        self.feature_handle = target_layer.register_forward_hook(self.feature)
        # 勾配を取るためのregister_forward_hookを設定
        self.grad_handle = target_layer.register_forward_hook(self.gradient)

    # self.feature_handleの定義時に呼び出されるメソッド
    ## モデルの指定されたレイヤーの出力（特徴マップ）を保存する
    def feature(self, model, input, output):
        activation = output
        if type(activation) is tuple or type(activation) is list:
            activation = [act.to("cpu").detach() for act in activation]
            #activation = [flow.to("cpu").detach() for flow in activation]
            self.layer_output.append(activation)
        else:
            self.layer_output.append(activation.to("cpu").detach())

    # self.grad_handleの定義時に呼び出されるメソッド
    ## モデルの指定されたレイヤーの勾配を保存する
    ## 勾配が存在しない場合や勾配が必要ない場合は処理をスキップ
    def gradient(self, model, input, output):
        # 勾配が無いとき
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return  # ここでメソッド終了

        # 勾配を取得
        def _hook(grad):
            # gradが定義されていないが、勾配が計算されると各テンソルのgrad属性に保存されるっぽい（詳細未確認）
            self.layer_grad.append(grad.to("cpu").detach())

        # PyTorchのregister_hookメソッド（https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html）
        output.register_hook(_hook)

        # メモリの解放を行うメソッド、フックを解除してメモリを解放する

    def release(self):
        self.feature_handle.remove()
        self.grad_handle.remove()
