import glob
import pandas as pd
from utils.load_corpus import convert_gloss_to_class
from Parameter.Parameter import *
def load_phoenix_dataset():
    train_path = sorted(glob.glob(f"{TRAIN_DATADIR}/*/1"))
    # 13April_2011_Wednesday_tagesschau_default-14だけnanがでる(削除検討)
    train_path.remove(f"{TRAIN_DATADIR}/13April_2011_Wednesday_tagesschau_default-14/1")
    # train_pathからランダムにデータを選択(データ数を減らす試験用)
    # _, train_path = train_test_split(train_path, test_size=10, random_state=42)
    # 訓練コーパスをロード(csv)
    train_corpus = pd.read_csv(TEXT_TRAIN_PATH, delimiter="|")
    max_targets_length = max([len(i.split(" ")) for i in train_corpus["annotation"].values])

    print(f"最大ラベル長:{max_targets_length}")
    print(f"# of train data:{len(train_path)}")

    # テストデータのロード
    # test_path=sorted(glob.glob(f"{Parameter.TEST_DATADIR}/*/1"))
    test_corpus = pd.read_csv(TEXT_TEST_PATH, delimiter="|")

    # 評価データのロード
    dev_path = sorted(glob.glob(f"{DEV_DATADIR}/*/1"))
    # dev_pathからランダムに100個のデータを選択(データ数を減らす試験用)
    # _, dev_path = train_test_split(dev_path, test_size=10, random_state=42)
    dev_corpus = pd.read_csv(TEXT_DEV_PATH, delimiter="|")

    max_dev_target_length = max([len(i.split(" ")) for i in dev_corpus["annotation"].values])
    print(f"# of eval data:{len(dev_path)}")
    gloss2class, class2gloss = convert_gloss_to_class(pd.concat([train_corpus, dev_corpus, test_corpus]))
    return {"train_path": train_path, "train_corpus": train_corpus, "max_targets_length": max_targets_length, "test_corpus": test_corpus, "dev_path": dev_path, "dev_corpus": dev_corpus, "max_dev_target_length": max_dev_target_length, "gloss2class": gloss2class, "class2gloss": class2gloss}