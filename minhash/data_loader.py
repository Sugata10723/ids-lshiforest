import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import numpy.typing as npt
import config
import pickle

## List of main function
# load_cic_for_minhash
# load_unsw_for_minhash
# load_nsl_for_minhash
# load_unsw_binned
# load_nsl_binned


def load_cic_for_minhash() -> (
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]
):
    """CIC-IDS-2018のminhash用のデータセットを読み込み"""
    # _load_big_cic_data()
    with open("../data/cicids2018/alldata.pkl", "rb") as file:
        df: pd.DataFrame = pickle.load(file)

    df_sample = df.sample(n=50000, random_state=42).reset_index(drop=True)
    df_y = df_sample["Label"]
    X_train, X_test, y_train, y_cat = train_test_split(
        df_sample, df_y, test_size=0.3, random_state=42
    )
    X_train = X_train[X_train["Label"] == "Benign"]  # 正常データだけ
    y_train = y_train[X_train.index]
    y_test = pd.Series([0 if label == "Benign" else 1 for label in y_cat])
    X_train = X_train.reset_index(drop=True)
    y_cat = y_cat.reset_index(drop=True)

    X_train = X_train[config.categorical_columns_cic]
    X_test = X_test[config.categorical_columns_cic]

    X_train_minhash = _process_for_minhash(X_train)
    X_test_minhash = _process_for_minhash(X_test)

    return X_train_minhash, y_train, X_test_minhash, y_test, y_cat


def load_unsw_for_minhash() -> (
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]
):
    """UNSWのminhash用のデータセットを読み込み"""
    train_df = pd.read_csv(
        "../data/unsw_nb15/UNSW_NB15_testing-set.csv"
    )  # テストとトレーニングが逆
    test_df = pd.read_csv("../data/unsw_nb15/UNSW_NB15_training-set.csv")
    y_cat = test_df["attack_cat"]  # のちの分析用に用意

    train_df = train_df[train_df["label"] == 0]  # 正常通信だけで学習
    train_df = train_df.reset_index()
    train_df.drop(["id", "attack_cat", "rate"], axis=1, inplace=True)
    test_df.drop(["id", "attack_cat", "rate"], axis=1, inplace=True)

    y_train = train_df["label"]
    y_test = test_df["label"]
    X_train = train_df.drop("label", axis=1)
    X_test = test_df.drop("label", axis=1)

    X_train = X_train[config.categorical_columns_unsw]
    X_test = X_test[config.categorical_columns_unsw]

    X_train_minhash = _process_for_minhash(X_train)
    X_test_minhash = _process_for_minhash(X_test)

    return X_train_minhash, y_train, X_test_minhash, y_test, y_cat


def load_nsl_for_minhash() -> (
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]
):
    """nsl-kddデータセットのminhash用に変換したものを読み込み"""
    train_df = pd.read_csv(
        "../data/nsl_kdd/KDDTrain+.txt", header=None, names=config.columns_nsl
    )
    test_df = pd.read_csv(
        "../data/nsl_kdd/KDDTest+.txt", header=None, names=config.columns_nsl
    )
    train_df = train_df[train_df["class"] == "normal"]
    train_df = train_df.reset_index()

    y_train = train_df["class"]
    y_cat = test_df["class"]
    y_test = pd.Series([0 if label == "normal" else 1 for label in y_cat])
    X_train = train_df.drop("class", axis=1)
    X_test = test_df.drop("class", axis=1)

    X_train = X_train[config.categorical_columns_nsl]
    X_test = X_test[config.categorical_columns_nsl]

    X_train_minhash = _process_for_minhash(X_train)
    X_test_minhash = _process_for_minhash(X_test)

    return X_train_minhash, y_train, X_test_minhash, y_test, y_cat


def load_unsw_binned() -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """数値データもビニングして使う"""
    train_df = pd.read_csv("../data/unsw_nb15/UNSW_NB15_testing-set.csv")
    test_df = pd.read_csv("../data/unsw_nb15/UNSW_NB15_training-set.csv")
    train_df = train_df[train_df["label"] == 0]  # 正常通信だけで学習
    train_df = train_df.reset_index()

    y_cat = test_df["attack_cat"]  # 攻撃手法分析用

    train_df.drop(["id", "attack_cat", "rate"], axis=1, inplace=True)
    test_df.drop(["id", "attack_cat", "rate"], axis=1, inplace=True)

    y_train = train_df["label"]
    y_test = test_df["label"]
    X_train = train_df.drop("label", axis=1)
    X_test = test_df.drop("label", axis=1)

    # Run standard scaler and PCA to numerical data
    X_train_bs, X_test_bs = _process_sc_pca(
        X_train_num=X_train[config.numerical_columns_unsw],
        X_test_num=X_test[config.numerical_columns_unsw],
        num_column_names=config.numerical_columns_unsw,
        n_pca=5,
    )

    # Reset categorical columns index and concatenate them.
    X_train_comed = pd.concat(
        [X_train[config.categorical_columns_unsw].reset_index(drop=True), X_train_bs],
        axis=1,
    )
    X_test_comed = pd.concat(
        [X_test[config.categorical_columns_unsw].reset_index(drop=True), X_test_bs],
        axis=1,
    )

    X_train_minhash = _process_for_minhash(X_train_comed)
    X_test_minhash = _process_for_minhash(X_test_comed)

    return X_train_minhash, y_train, X_test_minhash, y_test, y_cat


def load_nsl_binned() -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """数値をビニングして使用"""
    train_df = pd.read_csv(
        "../data/nsl_kdd/KDDTrain+.txt", header=None, names=config.columns_nsl
    )
    test_df = pd.read_csv(
        "../data/nsl_kdd/KDDTest+.txt", header=None, names=config.columns_nsl
    )
    train_df = train_df[train_df["class"] == "normal"].reset_index(drop=True)

    y_train = train_df["class"]
    y_cat = test_df["class"]
    y_test = pd.Series([0 if label == "normal" else 1 for label in y_cat])
    X_train = train_df.drop(["class", "difficulty"], axis=1)
    X_test = test_df.drop(["class", "difficulty"], axis=1)

    # process standard scaler and pca to numerical data
    X_train_bs, X_test_bs = _process_sc_pca(
        X_train_num=X_train[config.numerical_columns_nsl],
        X_test_num=X_test[config.numerical_columns_nsl],
        num_column_names=config.numerical_columns_nsl,
        n_pca=5,
    )

    X_train_comed = pd.concat(
        [X_train_bs, X_train[config.categorical_columns_nsl].reset_index(drop=True)],
        axis=1,
    )
    X_test_comed = pd.concat(
        [X_test_bs, X_test[config.categorical_columns_nsl].reset_index(drop=True)],
        axis=1,
    )

    X_train_minhash = _process_for_minhash(X_train_comed)
    X_test_minhash = _process_for_minhash(X_test_comed)

    return X_train_minhash, y_train, X_test_minhash, y_test, y_cat


def _process_for_minhash(df: pd.DataFrame) -> pd.Series:  # よりよい方法あるかもしれない
    return df.astype(str).apply(
        lambda row: {f"{col}:{val}" for col, val in row.items()}, axis=1
    )


def _process_sc_pca(
    X_train_num: pd.DataFrame,
    X_test_num: pd.DataFrame,
    num_column_names: List[str],
    n_pca: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    sc = StandardScaler()
    X_train_ss: pd.DataFrame = pd.DataFrame(
        sc.fit_transform(X_train_num),
        columns=num_column_names,
    )
    X_test_ss: pd.DataFrame = pd.DataFrame(
        sc.transform(X_test_num),
        columns=num_column_names,
    )

    pca = PCA(n_components=n_pca)
    _column_pca = [f"pca_{i+1}" for i in range(n_pca)]
    _train_pca: npt.ArrayLike = pca.fit_transform(X_train_ss)
    _test_pca: npt.ArrayLike = pca.transform(X_test_ss)
    X_train_pca: pd.DataFrame = pd.DataFrame(
        data=_train_pca, columns=_column_pca
    )  # Reset index
    X_test_pca: pd.DataFrame = pd.DataFrame(data=_test_pca, columns=_column_pca)

    X_train_bs: pd.DataFrame = X_train_pca.copy()
    X_test_bs: pd.DataFrame = X_test_pca.copy()
    for col in X_test_pca:
        binned_series, bin_edges = pd.cut(
            X_train_pca[col],
            bins=10,
            retbins=True,
            labels=False,
            duplicates="drop",
        )
        X_train_bs[col] = binned_series

        X_test_bs[col] = pd.cut(
            X_test_pca[col], bins=bin_edges, labels=False, include_lowest=True
        )
        X_test_bs[col] = X_test_bs[col].fillna(-1).astype(int)

    return X_train_bs, X_test_bs


def _load_big_cic_data() -> None:
    df_1 = pd.read_csv("../data/cicids2018/02-14-2018.csv")
    df_2 = pd.read_csv("../data/cicids2018/02-15-2018.csv")
    df_3 = pd.read_csv("../data/cicids2018/02-16-2018.csv")
    df_4 = pd.read_csv("../data/cicids2018/02-20-2018.csv")
    df_5 = pd.read_csv("../data/cicids2018/02-21-2018.csv")
    df_6 = pd.read_csv("../data/cicids2018/02-22-2018.csv")
    df_7 = pd.read_csv("../data/cicids2018/02-23-2018.csv")
    df_8 = pd.read_csv("../data/cicids2018/02-28-2018.csv")
    df_9 = pd.read_csv("../data/cicids2018/03-01-2018.csv")
    df_10 = pd.read_csv("../data/cicids2018/03-02-2018.csv")

    df = pd.concat(
        [df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10], axis=0
    )
    with open("../data/cicids2018/alldata.pkl", "wb") as file:
        pickle.dump(df, file)
