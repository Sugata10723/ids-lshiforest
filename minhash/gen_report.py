import pandas as pd
import sweetviz as sv

train_df = pd.read_csv("data/cicids2018/02-14-2018.csv")
test_df = pd.read_csv("data/cicids2018/02-15-2018.csv")

y_train = train_df["Label"]
y_test = test_df["Label"]
X_train = train_df.drop(["Flow Byts/s", "Flow Pkts/s"], axis=1)
X_test = test_df.drop(["Flow Byts/s", "Flow Pkts/s"], axis=1)

# 1. 比較レポートを作成
compare_report = sv.compare([X_train, "Train Data"], [X_test, "Test Data"])

# 2. レポートを表示
compare_report.show_html("CIC2018.html")
