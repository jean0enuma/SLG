import torch
import torch.optim as optim
from models.module.MBartSkeleton import MBartText2Pose

def test_mbart_text2pose():
    # --- 1. 設定項目 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_dim = 150  # 例: 50関節 x 3次元
    seq_len = 20  # 20フレームの動作
    batch_size = 2
    model_name = "facebook/mbart-large-50-many-to-many-mmt"

    print(f"🚀 Testing on {device}...")

    # --- 2. モデルの初期化 ---
    # メモリ節約のため、テスト時はPEFTを有効化
    model = MBartText2Pose(model_name=model_name, pose_dim=pose_dim, use_peft=True)
    model.to(device)
    model.train()

    # --- 3. ダミーデータの作成 ---
    # 入力テキスト（日本語と英語を混ぜて多言語対応を確認）
    input_texts = [
        "素早く右手を上げる",
        "A person walking slowly"
    ]

    # 正解ポーズデータ (Batch, Seq_Len, Pose_Dim)
    target_poses = torch.randn(batch_size, seq_len, pose_dim).to(device)

    # マスクデータ (パディングなしと想定して全て1)
    target_mask = torch.ones(batch_size, seq_len).to(device)

    # --- 4. 順伝播 (Forward Pass) ---
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    loss, predicted_poses = model(input_texts, target_poses)

    # --- 5. 検証内容の出力 ---
    print("-" * 30)
    print(f"✅ Loss: {loss.item():.4f}")
    print(f"✅ Predicted Pose Shape: {predicted_poses.shape}")  # (2, 20, 150)

    # 形状の整合性チェック
    assert predicted_poses.shape == (batch_size, seq_len, pose_dim), "Shape mismatch!"
    assert loss.item() > 0, "Loss should be positive!"
    assert loss.grad_fn is not None, "Loss should have a gradient function (Check if gradients are tracked)."

    # --- 6. 逆伝播 (Backward Pass) のテスト ---
    loss.backward()
    optimizer.step()
    print("✅ Backward pass successful.")
    print("-" * 30)
    print("🎉 Test Passed: The model can compute MSE loss and update parameters.")


if __name__ == "__main__":
    # 前述の MBartText2PoseTrainer クラスが定義されている前提で実行
    try:
        test_mbart_text2pose()
    except Exception as e:
        print(f"❌ Test Failed: {e}")