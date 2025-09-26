import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Set, Tuple, Any
from IPython.display import display
from abc import ABC, abstractmethod


class Data(ABC):
    def __init__(self) -> None:
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load(self) -> None:
        pass

    def preprocess(self) -> None:
        pass


class Data_unsw(Data):

    def load(self):


    def preprocess_minhash():


    def preprocess_

    def load(self) -> None:
        train_df = pd.read_csv(
            "data/unsw_nb15/UNSW_NB15_testing-set.csv"
        )  # テストとトレーニングが逆
        test_df = pd.read_csv("data/unsw_nb15/UNSW_NB15_training-set.csv")
        y_cat = test_df["attack_cat"]  # のちの分析用に用意
        train_df.drop(["id", "attack_cat", "rate"], axis=1, inplace=True)
        test_df.drop(["id", "attack_cat", "rate"], axis=1, inplace=True)

        train_df, test_df = sample(
            train_df=train_df, test_df=test_df, sample_frac=sample_frac
        )

        y_train = train_df["label"]
        y_test = test_df["label"]
        X_train = train_df.drop("label", axis=1)
        X_test = test_df.drop("label", axis=1)

        features_to_select = [
            "proto",
            "service",
            "state",
            "is_ftp_login",
            "is_sm_ips_ports",
        ]
        X_train = X_train[features_to_select]
        X_test = X_test[features_to_select]

    def preprocess(self) -> None:
        return super().preprocess()

    @staticmethod
    def load_unsw_nb15_for_minhash(
        sample_frac: float = 0.1,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """unsw_nb15データセットの読み込み"""
        train_df = pd.read_csv(
            "data/unsw_nb15/UNSW_NB15_testing-set.csv"
        )  # テストとトレーニングが逆
        test_df = pd.read_csv("data/unsw_nb15/UNSW_NB15_training-set.csv")
        y_cat = test_df["attack_cat"]  # のちの分析用に用意
        train_df.drop(["id", "attack_cat", "rate"], axis=1, inplace=True)
        test_df.drop(["id", "attack_cat", "rate"], axis=1, inplace=True)

        train_df, test_df = sample(
            train_df=train_df, test_df=test_df, sample_frac=sample_frac
        )

        y_train = train_df["label"]
        y_test = test_df["label"]
        X_train = train_df.drop("label", axis=1)
        X_test = test_df.drop("label", axis=1)

        features_to_select = [
            "proto",
            "service",
            "state",
            "is_ftp_login",
            "is_sm_ips_ports",
        ]
        X_train = X_train[features_to_select]
        X_test = X_test[features_to_select]

        # デバック用
        display(X_train)

        X_train = X_train.apply(row_to_set, axis=1, raw=False, result_type="reduce")
        X_test = X_test.apply(row_to_set, axis=1, raw=False, result_type="reduce")

        return X_train, y_train, X_test, y_test, y_cat

    @staticmethod
    def load_nsl_kdd_for_minhash(
        sample_frac: float = 0.3,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """nsl-kddデータセットの読み込み"""
        train_df = pd.read_csv("data/nsl_kdd/KDDTrain+.txt", header=None)
        test_df = pd.read_csv("data/nsl_kdd/KDDTest+.txt", header=None)

        train_df, test_df = sample(
            train_df=train_df, test_df=test_df, sample_frac=sample_frac
        )

        columns = [
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "logged_in",
            "num_compromised",
            "root_shell",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
            "is_host_login",
            "is_guest_login",
            "count",
            "srv_count",
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
            "class",
            "difficulty",
        ]
        categorical_columns = [
            "protocol_type",
            "service",
            "flag",
            "is_host_login",
            "is_guest_login",
            "logged_in",
        ]

        train_df.columns = columns
        test_df.columns = columns

        y_train = train_df["class"]
        y_test = test_df["class"]
        X_train = train_df.drop("class", axis=1)
        X_test = test_df.drop("class", axis=1)

        X_train = X_train[categorical_columns]
        X_test = X_test[categorical_columns]

        X_train = X_train.apply(row_to_set, axis=1, raw=False, result_type="reduce")
        X_test = X_test.apply(row_to_set, axis=1, raw=False, result_type="reduce")

        return X_train, y_train, X_test, y_test

    @staticmethod
    def load_unsw_nb15_with_binned_features(
        sample_frac: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
        """数値データもビニングして使う"""
        train_df = pd.read_csv("data/unsw_nb15/UNSW_NB15_testing-set.csv")
        test_df = pd.read_csv("data/unsw_nb15/UNSW_NB15_training-set.csv")

        train_df, test_df = sample(
            train_df=train_df, test_df=test_df, sample_frac=sample_frac
        )

        numerical_columns = [
            "dur",
            "spkts",
            "dpkts",
            "sbytes",
            "dbytes",
            "sttl",
            "dttl",
            "sload",
            "dload",
            "sloss",
            "dloss",
            "sinpkt",
            "dinpkt",
            "sjit",
            "djit",
            "swin",
            "stcpb",
            "dtcpb",
            "dwin",
            "tcprtt",
            "synack",
            "ackdat",
            "smean",
            "dmean",
            "trans_depth",
            "response_body_len",
            "ct_srv_src",
            "ct_state_ttl",
            "ct_dst_ltm",
            "ct_src_dport_ltm",
            "ct_dst_sport_ltm",
            "ct_dst_src_ltm",
            "ct_ftp_cmd",
            "ct_flw_http_mthd",
            "ct_src_ltm",
            "ct_srv_dst",
        ]
        categorical_columns = [
            "proto",
            "service",
            "state",
            "is_ftp_login",
            "is_sm_ips_ports",
        ]

        # 攻撃手法分析用
        y_cat = test_df["attack_cat"]

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
            sc.fit_transform(X_train_binned[numerical_columns]),
            columns=numerical_columns,
        )  # bs = binned scaled
        X_test_bs = pd.DataFrame(
            sc.transform(X_test_binned[numerical_columns]), columns=numerical_columns
        )

        for col in numerical_columns:
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
        X_train_categorical = X_train[categorical_columns].reset_index(drop=True)
        X_test_categorical = X_test[categorical_columns].reset_index(drop=True)

        X_train_comed = pd.concat([X_train_categorical, X_train_bs], axis=1)
        X_test_comed = pd.concat([X_test_categorical, X_test_bs], axis=1)

        # for debug
        print("before binning:")
        display(X_train)
        print("after binning: ")
        display(X_train_comed)
        print("describe: ")
        X_train_comed.describe(include="all")

        # Convert each row into a set of "feature:value" strings
        # X_train_sets = X_train_comed.apply(row_to_set, axis=1)
        # X_test_sets = X_test_comed.apply(row_to_set, axis=1)

        return X_train_comed, y_train, X_test_comed, y_test, y_cat

    @staticmethod
    def load_nsl_kdd_with_binned_features(
        sample_frac: float = 0.3,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """数値をビニングして使用"""
        train_df = pd.read_csv("data/nsl_kdd/KDDTrain+.txt", header=None)
        test_df = pd.read_csv("data/nsl_kdd/KDDTest+.txt", header=None)

        columns = [
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "logged_in",
            "num_compromised",
            "root_shell",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
            "is_host_login",
            "is_guest_login",
            "count",
            "srv_count",
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
            "class",
            "difficulty",
        ]
        numerical_columns = [
            "duration",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "num_compromised",
            "root_shell",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
            "count",
            "srv_count",
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
        ]
        categorical_columns = [
            "protocol_type",
            "service",
            "flag",
            "is_host_login",
            "is_guest_login",
            "logged_in",
        ]

        train_df.columns = columns
        test_df.columns = columns

        train_df, test_df = sample(
            train_df=train_df, test_df=test_df, sample_frac=sample_frac
        )

        y_train = train_df["class"]
        y_test = test_df["class"]
        X_train = train_df.drop(["class", "difficulty"], axis=1)
        X_test = test_df.drop(["class", "difficulty"], axis=1)

        X_train_binned = X_train.copy()
        X_test_binned = X_test.copy()

        sc = StandardScaler()
        X_train_bs = pd.DataFrame(
            sc.fit_transform(X_train_binned[numerical_columns]),
            columns=numerical_columns,
        )  # bs = binned scaled
        X_test_bs = pd.DataFrame(
            sc.transform(X_test_binned[numerical_columns]), columns=numerical_columns
        )

        for col in numerical_columns:
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

        X_train_comed = pd.concat([X_train_bs, X_train[categorical_columns]], axis=1)
        X_test_comed = pd.concat([X_test_bs, X_test[categorical_columns]], axis=1)

        X_train_sets = X_train_comed.apply(row_to_set, axis=1)
        X_test_sets = X_test_comed.apply(row_to_set, axis=1)

        return X_train_sets, y_train, X_test_sets, y_test


def sample(
    train_df: pd.DataFrame, test_df: pd.DataFrame, sample_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if sample_frac < 1.0:
        train_df = train_df.sample(frac=sample_frac, random_state=42)
        test_df = test_df.sample(frac=sample_frac, random_state=42)
    return train_df, test_df


# convert x: pd.DataFrame to set
# よりよい方法あるかもしれない
def row_to_set(row: pd.Series) -> Set[str]:
    return {f"{col}:{val}" for col, val in row.items()}
