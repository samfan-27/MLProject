import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
import joblib

DATA_PATH = "data/p4_clean.parquet"


def print_fold_result(i, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    tn, fp, fn, tp = cm.ravel()

    print(f"Fold {i}:")
    print(f"    Precision(1): {report['1']['precision']:.4f}")
    print(f"    Recall(1)   : {report['1']['recall']:.4f}")
    print(f"    F1(1)       : {report['1']['f1-score']:.4f}")
    print(f"    Confusion Matrix:")
    print(f"        TN {tn:5d} | FP {fp:5d}")
    print(f"        FN {fn:5d} | TP {tp:5d}\n")

    return report['1']


# ============================
# MAIN
# ============================
df = pd.read_parquet(DATA_PATH)
print(f"=== DATA LOADED ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Y_class distribution: {df['Y_class'].value_counts().to_dict()}")
print("-----------------------------\n")

Y = df["Y_class"]
X = df.drop(columns=["Y_class", "Y_reg"])

tscv = TimeSeriesSplit(n_splits=5)


# ======================================
# Logistic Regression
# ======================================
print("=== Logistic Regression (5-Fold TimeSeriesSplit) ===\n")

lr_metrics = []

for i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    metrics = print_fold_result(i, y_test, y_pred)
    lr_metrics.append(metrics)

final_lr = (model, scaler)
os.makedirs("models", exist_ok=True)
joblib.dump(final_lr[0], "models/log_reg_classifier.joblib")
joblib.dump(final_lr[1], "models/log_reg_scaler.joblib")

print("=== Logistic Regression Summary ===")
print(f"Avg Precision(1): {np.mean([m['precision'] for m in lr_metrics]):.4f}")
print(f"Avg Recall(1)   : {np.mean([m['recall']    for m in lr_metrics]):.4f}")
print(f"Avg F1(1)       : {np.mean([m['f1-score']  for m in lr_metrics]):.4f}\n")


# ======================================
# XGBoost Classifier
# ======================================
print("\n=== XGBoost Classifier (5-Fold TimeSeriesSplit) ===\n")

xgb_metrics = []

for i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = print_fold_result(i, y_test, y_pred)
    xgb_metrics.append(metrics)

joblib.dump(model, "models/xgb_classifier.json")

print("=== XGBoost Summary ===")
print(f"Avg Precision(1): {np.mean([m['precision'] for m in xgb_metrics]):.4f}")
print(f"Avg Recall(1)   : {np.mean([m['recall']    for m in xgb_metrics]):.4f}")
print(f"Avg F1(1)       : {np.mean([m['f1-score']  for m in xgb_metrics]):.4f}\n")

print("complete.")
