import pandas as pd
import sweetviz as sv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv(
    "data/unsw_nb15/UNSW_NB15_testing-set.csv"
)  # テストとトレーニングが逆
test_df = pd.read_csv("data/unsw_nb15/UNSW_NB15_training-set.csv")

train_df.drop(
    [
        "id",
        "attack_cat",
        "rate",
        "proto",
        "service",
        "state",
        "is_ftp_login",
        "is_sm_ips_ports",
    ],
    axis=1,
    inplace=True,
)
test_df.drop(
    [
        "id",
        "attack_cat",
        "rate",
        "proto",
        "service",
        "state",
        "is_ftp_login",
        "is_sm_ips_ports",
    ],
    axis=1,
    inplace=True,
)
y_train = train_df["label"]
y_test = test_df["label"]
X_train = train_df.drop("label", axis=1)
X_test = test_df.drop("label", axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

pca_cols = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]

X_train_final = pd.DataFrame(X_train_pca, columns=pca_cols)
X_train_final["label"] = y_train.reset_index(drop=True)

X_test_final = pd.DataFrame(X_test_pca, columns=pca_cols)
X_test_final["label"] = y_test.reset_index(drop=True)

# 1. 比較レポートを作成
compare_report = sv.compare([X_train_final, "Train Data"], [X_test_final, "Test Data"])

# 2. レポートを表示
compare_report.show_html("UNSW-PCA.html")
