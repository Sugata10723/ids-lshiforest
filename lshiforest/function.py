import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from src.detectors import LSHiForest
import data_loader as dl
import config
import numpy as np


def jikken(X_train: pd.DataFrame, X_test: pd.DataFrame, y_test:pd.Series) -> :
num_ensemblers = 100
classifiers = [
    ("sklearn.ISO", IsolationForest(n_estimators=num_ensemblers)), 
    ("L1SH", LSHiForest('L1SH', num_ensemblers)),
    ("L2SH", LSHiForest('L2SH', num_ensemblers)), 
    ("ALSH", LSHiForest('ALSH', num_ensemblers))
    ]
    
for i, (clf_name, clf) in enumerate(classifiers):
	print("\n"+clf_name+":")
	start_time = time.time()
	clf.fit(X_train)
	train_time = time.time()-start_time

	y_pred = clf.decision_function(X_test)
	test_time = time.time()-start_time-train_time
	auc = roc_auc_score(y_test, -y_pred) # なぜ反転させている？
	
	print("\tAUC score:\t", auc)
	print("\tTraining time:\t", train_time) 
	print("\tTesting time:\t", test_time)
