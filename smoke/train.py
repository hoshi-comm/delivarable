# train.py
# 喫煙2値分類：前処理 + 3モデル比較（LogReg / RF / AdaBoost）
# ベストモデルと可視化用メタ情報を models/model.joblib に保存

import os
import joblib
import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ========= 1) データ読み込み =========
DF_PATH = "data/train.csv"
TARGET = "smoking"

df = pd.read_csv(DF_PATH)
print("元データ shape:", df.shape)

# ========= 2) y / X 分割 =========
y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET, "id"], errors="ignore")

# ========= 3) 列型の自動判定 =========
num_cols = [c for c in X.columns if ptypes.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]
print("数値列:", num_cols)
print("カテゴリ列:", cat_cols)

# ========= 4) 前処理 =========
num_pre = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler())
])
cat_pre = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])
pre = ColumnTransformer([
    ("num", num_pre, num_cols),
    ("cat", cat_pre, cat_cols),
])

# ========= 5) 学習/検証分割 =========
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========= 6) 比較するモデル =========
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
}

results = {}
pipes = {}

for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    results[name] = {
        "Accuracy": round(accuracy_score(y_te, pred), 4),
        "AUC":      round(roc_auc_score(y_te, proba), 4),
        "F1":       round(f1_score(y_te, pred), 4),
    }
    pipes[name] = pipe

# ========= 7) 比較結果の表示 =========
df_results = pd.DataFrame(results).T.sort_values("AUC", ascending=False)
print("\n=== モデル比較結果 ===")
print(df_results)

best_name = df_results.index[0]
best_pipe = pipes[best_name]

# ========= 8) ベストモデルの可視化用データ作成 =========
proba_best = best_pipe.predict_proba(X_te)[:, 1]
fpr, tpr, _ = roc_curve(y_te, proba_best)
prec, rec, _ = precision_recall_curve(y_te, proba_best)

# OneHot後の列名を復元
ohe = None
try:
    ohe = best_pipe.named_steps["pre"].named_transformers_["cat"].named_steps.get("ohe", None)
except Exception:
    pass
cat_names = ohe.get_feature_names_out(cat_cols).tolist() if (ohe is not None and cat_cols) else []
feat_names = num_cols + cat_names

# 重要度/係数の抽出（モデルによりどちらかが入る）
clf = best_pipe.named_steps["clf"]
importance = getattr(clf, "feature_importances_", None)
if importance is not None:
    importance = importance.tolist()
coef = getattr(clf, "coef_", None)
if isinstance(coef, np.ndarray):
    coef = coef[0].tolist()

# ========= 9) 保存（辞書バンドル） =========
bundle = {
    "model": best_pipe,
    "best_model_name": best_name,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "features": feat_names,
    "importance": importance,   # RF など
    "coef": coef,               # ロジ回帰など
    "eval": {
        "y_true": y_te.tolist(),
        "proba":  proba_best.tolist(),
        "fpr":    fpr.tolist(),
        "tpr":    tpr.tolist(),
        "prec":   prec.tolist(),
        "rec":    rec.tolist(),
    },
    "compare": df_results.reset_index().rename(columns={"index": "model"}).to_dict(orient="list"),
}

os.makedirs("models", exist_ok=True)
joblib.dump(bundle, "models/model.joblib")
print(f"\nベストモデルを保存 -> models/model.joblib ({best_name})")