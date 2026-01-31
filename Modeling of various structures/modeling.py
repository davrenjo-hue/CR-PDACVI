import os
import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, recall_score, f1_score
)
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")


# ================= 特异度 ==================
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, _, _ = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    return tn / (tn + fp + 1e-7)


# ================= 配置 ==================
use_lasso_features = True

if use_lasso_features:
    feature_files = {
        2: r"aligned_features_label2.csv",
        3: r"aligned_features_label3.csv",
        4: r"aligned_features_label4.csv",
        5: r"aligned_features_label5.csv",
        6: r"aligned_features_label6.csv"#Samples after alignment
    }

output_dir = r""
os.makedirs(output_dir, exist_ok=True)

organs = [2, 3, 4, 5, 6]


# ================= 数据加载 ==================
def load_organ_data(organ_id):
    df = pd.read_csv(feature_files[organ_id])
    feature_cols = [c for c in df.columns if c not in ["filename", "label"] and not c.startswith("diagnostics_")]

    X = df[feature_cols].values
    y = df["label"].values

    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    return X, y


# =======================================================
#           Optuna 超参数搜索 - 每个模型一个 objective
# =======================================================

def objective_logr(trial, X_train, y_train, X_val, y_val):
    C = trial.suggest_float("C", 1e-3, 10.0, log=True)
    clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_val)[:,1]
    return roc_auc_score(y_val, prob)


def objective_svm(trial, X_train, y_train, X_val, y_val):
    C = trial.suggest_float("C", 1e-3, 10, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 1, log=True)
    clf = SVC(probability=True, C=C, gamma=gamma, random_state=42)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_val)[:,1]
    return roc_auc_score(y_val, prob)


def objective_rf(trial, X_train, y_train, X_val, y_val):
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_val)[:,1]
    return roc_auc_score(y_val, prob)


def objective_xgb(trial, X_train, y_train, X_val, y_val):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "eval_metric": "logloss",
        "objective": "binary:logistic",
        "random_state": 42
    }
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_val)[:,1]
    return roc_auc_score(y_val, prob)


def objective_lgb(trial, X_train, y_train, X_val, y_val):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "random_state": 42
    }
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_val)[:,1]
    return roc_auc_score(y_val, prob)


def objective_mlp(trial, X_train, y_train, X_val, y_val):
    hidden = trial.suggest_int("hidden", 50, 300)
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden,),
        max_iter=500,
        random_state=42
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_val)[:,1]
    return roc_auc_score(y_val, prob)


model_optuna_objectives = {
    "LogR": objective_logr,
    "SVM": objective_svm,
    "RF": objective_rf,
    "XGB": objective_xgb,
    "LGBM": objective_lgb,
    "MLP": objective_mlp
}


# ==================== 运行 Optuna + 测试集评估 ====================
def evaluate_models_on_organ(organ_id):
    print(f"\n==================== Organ {organ_id} ====================")

    X, y = load_organ_data(organ_id)

    # 外层 train/test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 内层 train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2,
        stratify=y_train_full, random_state=42
    )

    results = []

    for model_name, objective in model_optuna_objectives.items():

        print(f"\n⭐ 开始 {model_name} 的 Optuna 搜索...")

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val),
                       n_trials=50, show_progress_bar=False)

        print(f"➡ 最优 AUC = {study.best_value:.4f}")
        print(f"➡ 最优参数 = {study.best_params}")

        # 用最佳参数重新训练
        best_params = study.best_params

        if model_name == "LogR":
            clf = LogisticRegression(max_iter=2000, **best_params)
        elif model_name == "SVM":
            clf = SVC(probability=True, **best_params)
        elif model_name == "RF":
            clf = RandomForestClassifier(random_state=42, **best_params)
        elif model_name == "XGB":
            clf = xgb.XGBClassifier(objective="binary:logistic",
                                    eval_metric="logloss",
                                    random_state=42, **best_params)
        elif model_name == "LGBM":
            clf = lgb.LGBMClassifier(random_state=42, **best_params)
        elif model_name == "MLP":
            clf = MLPClassifier(hidden_layer_sizes=(best_params["hidden"],),
                               max_iter=500, random_state=42)

        clf.fit(X_train_full, y_train_full)
        prob = clf.predict_proba(X_test)[:,1]
        pred = (prob >= 0.5).astype(int)

        # 指标
        metrics = {
            "organ": organ_id,
            "model": model_name,
            "accuracy": accuracy_score(y_test, pred),
            "auc": roc_auc_score(y_test, prob),
            "sensitivity": recall_score(y_test, pred),
            "specificity": specificity_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "best_params": str(best_params),
            "y_test": str(list(y_test)),
            "y_prob": str(list(prob))
        }

        results.append(metrics)

    return results


# ==================== 主流程 ====================
def main():
    all_results = []

    for organ_id in organs:
        res = evaluate_models_on_organ(organ_id)
        all_results.extend(res)

    df = pd.DataFrame(all_results).sort_values(by=["organ", "auc"], ascending=[True, False])


    save_path = os.path.join(output_dir, "optuna_model_results.csv")
    df.to_csv(save_path, index=False)

    print("\n所有结果已保存：", save_path)


if __name__ == "__main__":
    main()
