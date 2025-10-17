import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import Tuple
import config
import pickle

# load data type
# load_nsl_kdd
# load_unsw_nb15
# load_cic_ids


def load_nsl_kdd() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    # Load the NSL-KDD dataset
    train_df = pd.read_csv(
        "../data/nsl_kdd/KDDTrain+.txt", header=None, names=config.columns_nsl
    )
    train_df = train_df[train_df["class"] == "normal"]
    train_df = train_df.reset_index(drop=True)
    test_df = pd.read_csv(
        "../data/nsl_kdd/KDDTest+.txt", header=None, names=config.columns_nsl
    )

    # delete uneccesary features
    train_df.drop(
        ["difficulty", "protocol_type", "service", "flag"], axis=1, inplace=True
    )
    test_df.drop(
        ["difficulty", "protocol_type", "service", "flag"], axis=1, inplace=True
    )

    # encode the attack column to 0/1
    train_df["class"] = train_df["class"].apply(lambda x: 0 if x == "normal" else 1)
    test_df["class"] = test_df["class"].apply(lambda x: 0 if x == "normal" else 1)

    # Separate features and labels
    y_train = train_df["class"]
    y_test = test_df["class"]
    X_train = train_df.drop("class", axis=1)
    X_test = test_df.drop("class", axis=1)

    X_train, X_test = preprocess(X_train, X_test, n=10)

    return X_train, y_train, X_test, y_test


def load_unsw_nb15() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    # Load the UNSW-NB15 dataset
    train_df = pd.read_csv("../data/unsw_nb15/UNSW_NB15_testing-set.csv")  # traversal
    train_df = train_df[train_df["label"] == 0]  # Train on normal traffic
    train_df = train_df.reset_index(drop=True)
    test_df = pd.read_csv("../data/unsw_nb15/UNSW_NB15_training-set.csv")

    # Preprocess the data
    train_df.drop(["id", "attack_cat", "rate"], axis=1, inplace=True)  # rate
    test_df.drop(["id", "attack_cat", "rate"], axis=1, inplace=True)

    # eliminate categorical columns
    train_df.drop(
        ["proto", "service", "state", "is_ftp_login", "is_sm_ips_ports"],
        axis=1,
        inplace=True,
    )
    test_df.drop(
        ["proto", "service", "state", "is_ftp_login", "is_sm_ips_ports"],
        axis=1,
        inplace=True,
    )

    # Separate features and labels
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]
    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    X_train, X_test = preprocess(X_train, X_test, n=10)

    return X_train, y_train, X_test, y_test


def load_cic_ids() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    CIC-IDS-2018のminhash用のデータセットを読み込み
    https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
    """
    with open("../data/cicids2018/alldata.pkl", "rb") as file:
        df = pickle.load(file)

    df_sample = df.sample(n=50000, random_state=42).reset_index(drop=True)
    df_y = df_sample["Label"]
    X_train, X_test, y_train, y_test = train_test_split(
        df_sample, df_y, test_size=0.3, random_state=42
    )
    X_train = X_train[X_train["Label"] == "Benign"]

    X_train = X_train[config.numerical_columns_cic]
    X_test = X_test[config.numerical_columns_cic]

    X_train, X_test = preprocess(X_train, X_test, n=10)

    return X_train, y_train, X_test, y_test


def preprocess(
    X_train: pd.DataFrame, X_test: pd.DataFrame, n: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Normalize the data
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca = pca.transform(X_test_sc)

    pca_columns = [f"pca_{i}" for i in range(n)]
    X_train = pd.DataFrame(data=X_train_pca, columns=pca_columns)
    X_test = pd.DataFrame(data=X_test_pca, columns=pca_columns)

    return X_train, X_test
