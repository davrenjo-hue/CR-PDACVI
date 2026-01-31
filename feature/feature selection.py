import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ==================== 配置区 ====================
feature_files = {
    2: r"Radiomics_label2.csv",
    3: r"Radiomics_label3.csv",
    4: r"Radiomics_label4.csv",
    5: r"Radiomics_label5.csv",
    6: r"Radiomics_label6.csv"
}

OUTPUT_ROOT = r""
os.makedirs(OUTPUT_ROOT, exist_ok=True)

target_col = "label"
filename_col = "filename"

# ==========================================================
def run_lasso_for_organ(organ_id, input_csv):
    print(f"\n{'='*70}")
    print(f"🔹 正在处理器官 {organ_id}  → {input_csv}")
    print(f"{'='*70}")

    # 1. 读取数据
    df = pd.read_csv(input_csv)
    drop_cols = ["time_used"] + [c for c in df.columns if c.lower().startswith("diagnostics_")]
    df = df.drop(columns=drop_cols, errors="ignore")

    if target_col not in df.columns or filename_col not in df.columns:
        raise ValueError(f"文件缺少 '{target_col}' 或 '{filename_col}'")

    feature_cols = [c for c in df.columns if c not in [target_col, filename_col, "group"]]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col].values
    filenames = df[filename_col].values

    # 2. 预处理
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))

    # 3. LASSO 带交叉验证
    lasso = LassoCV(cv=10, random_state=42, max_iter=20000, n_jobs=-1)
    lasso.fit(X_scaled, y)

    best_alpha = lasso.alpha_
    print(f"✅ 最优 α: {best_alpha:.6f}")

    # 4. 提取非零系数
    coef = pd.Series(lasso.coef_, index=feature_cols)
    selected = coef[coef != 0].index.tolist()
    print(f"保留特征数: {len(selected)} / {len(feature_cols)}")

    # gpt. 保存降维后的特征
    df_out = df[[filename_col, target_col] + selected]
    out_dir = os.path.join(OUTPUT_ROOT, str(organ_id))
    os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(os.path.join(out_dir, "reduced_features.csv"), index=False)

    # 6. 保存所有可视化数据（不出图）
    result_data = {
        "organ": organ_id,
        "alphas": lasso.alphas_.tolist(),
        "mse_path": lasso.mse_path_.tolist(),
        "coef": lasso.coef_.tolist(),
        "feature_cols": feature_cols,
        "selected_features": selected,
        "alpha_best": float(best_alpha)
    }
    with open(os.path.join(out_dir, "lasso_info.json"), "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    # 保存模型对象（供后续复现）
    joblib.dump({
        "lasso": lasso,
        "scaler": scaler,
        "imputer": imputer
    }, os.path.join(out_dir, "lasso_model.pkl"))

    # 7. 保存非零特征表
    coef_df = pd.DataFrame({
        "Feature": coef.index,
        "Coefficient": coef.values,
        "Abs_Coefficient": np.abs(coef.values)
    }).query("Coefficient != 0").sort_values("Abs_Coefficient", ascending=False)
    coef_df.to_csv(os.path.join(out_dir, "lasso_coef.csv"), index=False)

    # 8. 验证（可选）
    X_train, X_test, y_train, y_test = train_test_split(df_out[selected], y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    print(f"🎯 器官 {organ_id} 完成，结果保存至：{out_dir}")

# ==========================================================
if __name__ == "__main__":
    for organ, path in feature_files.items():
        try:
            run_lasso_for_organ(organ, path)
        except Exception as e:
            print(f"❌ 器官 {organ} 失败：{e}")

    print("\n✅ 所有器官 LASSO 完成！")
    print("📦 结果已保存，可用于后续绘图与融合。")
