# app.py
# 学習バンドル models/model.joblib を読み込み、可視化＆操作するUI

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="Smoker ML Report", layout="wide")
st.title("🚬 Smoker Classification – 学習レポート & デモ")

# ===== 0) モデル/メタ情報の読み込み =====
obj = joblib.load("models/model.joblib")

# （保険）パイプライン単体が保存されている場合にも動くように
if isinstance(obj, dict) and "model" in obj:
    bundle = obj
    pipe        = bundle["model"]
    best_name   = bundle.get("best_model_name", "BestModel")
    num_cols    = bundle.get("num_cols", [])
    cat_cols    = bundle.get("cat_cols", [])
    feat_names  = bundle.get("features", num_cols + cat_cols)
    importance  = bundle.get("importance", None)
    coef        = bundle.get("coef", None)
    compare_tbl = pd.DataFrame(bundle.get("compare", {}))
    ev          = bundle.get("eval", {})
    y_true = np.array(ev.get("y_true", []))
    proba  = np.array(ev.get("proba", []))
    fpr    = np.array(ev.get("fpr", []))
    tpr    = np.array(ev.get("tpr", []))
    prec   = np.array(ev.get("prec", []))
    rec    = np.array(ev.get("rec", []))
else:
    # Pipelineだけ保存されている場合（最低限のUI）
    pipe = obj
    best_name = "(pipeline)"
    pre = pipe.named_steps.get("pre", None)
    num_cols, cat_cols = [], []
    if pre is not None and hasattr(pre, "transformers_"):
        for name, trans, cols in pre.transformers_:
            if name == "num": num_cols = list(cols)
            elif name == "cat": cat_cols = list(cols)
    feat_names  = num_cols + cat_cols
    importance  = getattr(pipe.named_steps.get("clf", object()), "feature_importances_", None)
    coef_arr    = getattr(pipe.named_steps.get("clf", object()), "coef_", None)
    coef        = coef_arr[0].tolist() if isinstance(coef_arr, np.ndarray) else None
    compare_tbl = pd.DataFrame([{"model": best_name, "AUC": None, "Accuracy": None, "F1": None}])
    y_true = proba = fpr = tpr = prec = rec = np.array([])

expected = num_cols + cat_cols

# ===== タブ構成 =====
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["概要", "モデル比較", "しきい値＆混同行列", "重要度/係数", "バッチ洞察", "個別予測"]
)

with tab1:
    st.subheader("データ・前処理の概要")
    st.markdown(f"""
- ベストモデル: **{best_name}**  
- 数値列: `{num_cols}`  
- カテゴリ列: `{cat_cols}`  
- 前処理: 数値=欠損中央値→標準化、カテゴリ=欠損最頻値→OneHot  
""")
    # ROC / PR 可視化（数値から描画できる場合のみ）
    if len(fpr) > 0:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC")
            st.pyplot(fig)
            st.caption(f"AUC (best): {roc_auc_score(y_true, proba):.3f}")
        with c2:
            fig2, ax2 = plt.subplots()
            ax2.plot(rec, prec)
            ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("PR Curve")
            st.pyplot(fig2)
    else:
        st.info("評価曲線データが見つかりません。先に `python train.py` を実行してください。")

with tab2:
    st.subheader("モデル比較（AUC / Accuracy / F1）")
    if not compare_tbl.empty:
        st.dataframe(compare_tbl)
    else:
        st.info("比較表がありません。`train.py` を実行してください。")

with tab3:
    st.subheader("ビジネス要件に合わせたしきい値の調整")
    if len(proba) == 0:
        st.info("評価用の確率がありません。`train.py` を実行してください。")
    else:
        th = st.slider("判定しきい値", 0.0, 1.0, 0.5, 0.01)
        y_pred = (proba >= th).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Precision(1)", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
        with c2: st.metric("Recall(1)",    f"{recall_score(y_true, y_pred):.3f}")
        with c3: st.metric("F1",           f"{f1_score(y_true, y_pred):.3f}")

        st.markdown("**Confusion Matrix (labels=[0,1])**")
        st.dataframe(pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"]))

with tab4:
    st.subheader("どの特徴が効いたか（寄与の可視化）")
    if importance is not None:
        df_imp = pd.DataFrame({"feature": feat_names, "importance": importance}).sort_values("importance", ascending=False)
        st.bar_chart(df_imp.set_index("feature").head(10))
        st.caption("※ ツリー系の重要度：誤差減少寄与の合計")
    elif coef is not None:
        df_coef = pd.DataFrame({"feature": feat_names, "coef": coef})
        df_coef["abs"] = df_coef["coef"].abs()
        df_coef = df_coef.sort_values("abs", ascending=False).drop(columns="abs")
        st.bar_chart(df_coef.set_index("feature").head(10))
        st.caption("※ ロジスティック回帰の係数：符号=方向、絶対値=寄与強度")
    else:
        st.info("このモデルは簡易重要度を提供しません。")

with tab5:
    st.subheader("CSVアップロードで傾向を可視化")
    f = st.file_uploader("学習時の列を含む CSV をアップロードしてください", type=["csv"])
    if f is not None:
        df_u = pd.read_csv(f)
        Xb = df_u.reindex(columns=expected)
        prob_b = pipe.predict_proba(Xb)[:, 1]
        pred_b = (prob_b >= 0.5).astype(int)

        st.write("プレビュー")
        st.dataframe(df_u.head())

        # 喫煙確率の分布
        st.markdown("**喫煙確率の分布**")
        fig, ax = plt.subplots()
        ax.hist(prob_b, bins=20, range=(0, 1))
        ax.set_xlabel("Probability")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # TOP特徴（現ベストモデル基準）
        st.markdown("**特徴の寄与（TOP10）**")
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp_df = pd.DataFrame({"feature": feat_names, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
            st.bar_chart(imp_df.set_index("feature").head(10))
        elif hasattr(clf, "coef_"):
            cf = clf.coef_[0]
            cdf = pd.DataFrame({"feature": feat_names, "coef": cf})
            cdf["abs"] = cdf["coef"].abs()
            st.bar_chart(cdf.sort_values("abs", ascending=False).set_index("feature").head(10)[["coef"]])

with tab6:
    st.subheader("個別予測（実運用の雰囲気）")
    with st.sidebar:
        st.markdown("### 入力フォーム（1人分）")
        user = {}
        for c in num_cols:
            user[c] = st.number_input(c, value=0.0, step=1.0, format="%.2f")
        for c in cat_cols:
            user[c] = st.text_input(c, value="")
    xin = pd.DataFrame([user]).reindex(columns=expected)
    p = float(pipe.predict_proba(xin)[0, 1])
    st.metric("喫煙者確率", f"{p*100:.1f}%")
    st.write("判定:", "喫煙者" if p >= 0.5 else "非喫煙者")