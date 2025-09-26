import pandas as pd
import sweetviz as sv
from data_loader import DataLoader
from sklearn.decomposition import PCA


X_train, y_train, X_test, y_test, y_cat = DataLoader.load_unsw_nb15(sample_frac=1)

pca = PCA(n_components=0.5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# 1. 比較レポートを作成
compare_report = sv.compare([X_train, "Train Data"], [X_test, "Test Data"])

# 2. レポートを表示
compare_report.show_html("UNSW-PCA.html")
