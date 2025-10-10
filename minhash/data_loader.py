import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import config

# load_unsw_for_minhash
# load_nsl_for_minhash
# load_unsw_binned
# load_nsl_binned


def load_unsw_for_minhash() -> (
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]
):
    """UNSWのminhash用のデータセットを読み込み"""
    train_df = pd.read_csv(
        "data/unsw_nb15/UNSW_NB15_testing-set.csv"
    )  # テストとトレーニングが逆
    test_df = pd.read_csv("data/unsw_nb15/UNSW_NB15_training-set.csv")
    y_cat = test_df["attack_cat"]  # のちの分析用に用意
    train_df.drop(["id", "attack_cat", "rate"], axis=1, inplace=True)
    test_df.drop(["id", "attack_cat", "rate"], axis=1, inplace=True)

    y_train = train_df["label"]
    y_test = test_df["label"]
    X_train = train_df.drop("label", axis=1)
    X_test = test_df.drop("label", axis=1)

    X_train = X_train[config.categorical_columns_unsw]
    X_test = X_test[config.categorical_columns_unsw]

    X_train_minhash = process_for_minhash(X_train)
    X_test_minhash = process_for_minhash(X_test)

    return X_train_minhash, y_train, X_test_minhash, y_test, y_cat


def load_nsl_for_minhash() -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """nsl-kddデータセットのminhash用に変換したものを読み込み"""

    train_df = pd.read_csv(
        "data/nsl_kdd/KDDTrain+.txt", header=None, names=config.columns_nsl
    )
    test_df = pd.read_csv(
        "data/nsl_kdd/KDDTest+.txt", header=None, names=config.columns_nsl
    )

    y_train = train_df["class"]
    y_test = test_df["class"]
    X_train = train_df.drop("class", axis=1)
    X_test = test_df.drop("class", axis=1)

    X_train = X_train[config.categorical_columns_nsl]
    X_test = X_test[config.categorical_columns_nsl]

    X_train_minhash = process_for_minhash(X_train)
    X_test_minhash = process_for_minhash(X_test)

    return X_train_minhash, y_train, X_test_minhash, y_test


def load_unsw_binned() -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """数値データもビニングして使う"""
    train_df = pd.read_csv("data/unsw_nb15/UNSW_NB15_testing-set.csv")
    test_df = pd.read_csv("data/unsw_nb15/UNSW_NB15_training-set.csv")

    y_cat = test_df["attack_cat"]  # 攻撃手法分析用

    train_df.drop(["id", "attack_cat", "rate"], axis=1, inplace=True)
    test_df.drop(["id", "attack_cat", "rate"], axis=1, inplace=True)

    y_train = train_df["label"]
    y_test = test_df["label"]
    X_train = train_df.drop("label", axis=1)
    X_test = test_df.drop("label", axis=1)

    X_train_binned = X_train.copy()
    X_test_binned = X_test.copy()

    sc = StandardScaler()
    X_train_bs = pd.DataFrame(
        sc.fit_transform(X_train_binned[config.numerical_columns_unsw]),
        columns=config.numerical_columns_unsw,
    )  # bs = binned scaled
    X_test_bs = pd.DataFrame(
        sc.transform(X_test_binned[config.numerical_columns_unsw]),
        columns=config.numerical_columns_unsw,
    )

    for col in config.numerical_columns_unsw:
        binned_series, bin_edges = pd.cut(
            X_train_bs[col],
            bins=10,
            retbins=True,
            labels=False,
            duplicates="drop",
        )

        X_train_bs[col] = binned_series
        X_test_bs[col] = pd.cut(
            X_test_bs[col], bins=bin_edges, labels=False, include_lowest=True
        )
        X_test_bs[col] = X_test_bs[col].fillna(-1).astype(int)

    # Reset Index (add 2025/09/16)
    X_train_categorical = X_train[config.categorical_columns_unsw].reset_index(
        drop=True
    )
    X_test_categorical = X_test[config.categorical_columns_unsw].reset_index(drop=True)

    X_train_comed = pd.concat([X_train_categorical, X_train_bs], axis=1)
    X_test_comed = pd.concat([X_test_categorical, X_test_bs], axis=1)

    X_train_minhash = process_for_minhash(X_train_comed)
    X_test_minhash = process_for_minhash(X_test_comed)

    return X_train_minhash, y_train, X_test_minhash, y_test, y_cat


def load_nsl_binned() -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """数値をビニングして使用"""
    train_df = pd.read_csv(
        "data/nsl_kdd/KDDTrain+.txt", header=None, names=config.columns_nsl
    )
    test_df = pd.read_csv(
        "data/nsl_kdd/KDDTest+.txt", header=None, names=config.columns_nsl
    )

    y_train = train_df["class"]
    y_test = test_df["class"]
    X_train = train_df.drop(["class", "difficulty"], axis=1)
    X_test = test_df.drop(["class", "difficulty"], axis=1)

    sc = StandardScaler()
    X_train_bs = pd.DataFrame(
        sc.fit_transform(X_train[config.numerical_columns_nsl]),
        columns=config.numerical_columns_nsl,
    )  # bs = binned scaled
    X_test_bs = pd.DataFrame(
        sc.transform(X_test[config.numerical_columns_nsl]),
        columns=config.numerical_columns_nsl,
    )

    for col in config.numerical_columns_nsl:
        binned_series, bin_edges = pd.cut(
            X_train_bs[col],
            bins=10,
            retbins=True,
            labels=False,
            duplicates="drop",
        )

        X_train_bs[col] = binned_series
        X_test_bs[col] = pd.cut(
            X_test_bs[col], bins=bin_edges, labels=False, include_lowest=True
        )
        X_test_bs[col] = X_test_bs[col].fillna(-1).astype(int)

    X_train_comed = pd.concat(
        [X_train_bs, X_train[config.categorical_columns_nsl]], axis=1
    )
    X_test_comed = pd.concat(
        [X_test_bs, X_test[config.categorical_columns_nsl]], axis=1
    )

    X_train_minhash = process_for_minhash(X_train_comed)
    X_test_minhash = process_for_minhash(X_test_comed)

    return X_train_minhash, y_train, X_test_minhash, y_test


# convert x: pd.DataFrame to set
# よりよい方法あるかもしれない
def process_for_minhash(df: pd.DataFrame) -> pd.Series:
    return df.astype(str).apply(
        lambda row: {f"{col}:{val}" for col, val in row.items()}, axis=1
    )
