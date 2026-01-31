# optuna_stacking_full.py
import os
import json
import joblib
import time
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")

import optuna

# ==================== 配置区域 ====================
DATA_DIR = r""#Samples after alignment
OUTPUT_DIR = r""
# 你选择了“自定义路径(B)”，我在 OUTPUT_DIR 下创建 optuna_custom 文件夹作为默认位置。
# 如果你想放到别的地方，请直接修改下面这个 OPTUNA_DIR 为你想要的绝对路径。
OPTUNA_DIR = os.path.join(OUTPUT_DIR, "optuna_custom")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OPTUNA_DIR, exist_ok=True)

ORGANS = [2,3,4,5,6]
N_TRIALS = 50   # 你确认的每器官每模型 30 次

# 基模型定义 (不带超参)
BASE_MODELS = {
    "LogR": LogisticRegression(max_iter=2000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "RF": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGB": xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        use_label_encoder=False
    ),
    "LGBM": lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbose=-1,
        min_gain_to_split=0.0
    ),
    "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

# meta-models（如需，这里保留与你先前脚本一致的配置）
META_MODELS = {
    "LogR": LogisticRegression(max_iter=2000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "RF": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGB": xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        use_label_encoder=False
    ),
    "LGBM": lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbose=-1,
        min_gain_to_split=0.0
    ),
    "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

COMBO_SIZES = [1, 2, 3, 4, 5, 6]
# ==================================================

# 工具函数
def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, alpha=0.05):
    scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_score[idx]))
    sorted_scores = np.sort(scores)
    if len(sorted_scores) == 0:
        return (np.nan, np.nan, np.nan)
    ci_low = sorted_scores[int(alpha / 2 * len(sorted_scores))]
    ci_high = sorted_scores[int((1 - alpha / 2) * len(sorted_scores))-1]
    return np.mean(scores), ci_low, ci_high

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def load_and_align_data():
    dfs = {}
    for org in ORGANS:
        path = os.path.join(DATA_DIR, f"aligned_features_label{org}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        df = pd.read_csv(path)
        if not {"filename", "label"}.issubset(df.columns):
            raise ValueError(f"{path} 缺少 filename 或 label")
        dfs[org] = df

    # 取共同 filename 对齐
    common = set(dfs[ORGANS[0]]["filename"])
    for org in ORGANS[1:]:
        common &= set(dfs[org]["filename"])
    common = sorted(list(common))
    print(f"共同样本: {len(common)}")

    aligned, y_all = {}, None
    for org in ORGANS:
        df = dfs[org]
        df = df[df["filename"].isin(common)].set_index("filename").loc[common].reset_index()
        feat_cols = [c for c in df.columns if c not in ["filename", "label"] and not c.startswith("diagnostics_")]
        X = df[feat_cols].select_dtypes(include=[np.number])
        aligned[org] = {"X": X, "feat_cols": feat_cols}
        if y_all is None:
            y_all = df["label"].values
    print(f"标签分布: 健康={sum(y_all==0)}, 疾病={sum(y_all==1)}")
    return aligned, y_all

# ========== Optuna 搜索空间定义 ==========

def suggest_params_logr(trial):
    # Logistic: C (inverse regularization), penalty l2, solver liblinear 或 saga
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    penalty = trial.suggest_categorical("penalty", ["l2"])
    solver = "liblinear" if penalty=="l2" else "saga"
    return {"C": C, "penalty": penalty, "solver": solver, "max_iter":2000, "random_state":42}

def suggest_params_svm(trial):
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
    if kernel == "rbf":
        gamma = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
    else:
        gamma = "scale"
    return {"C": C, "kernel": kernel, "gamma": gamma, "probability": True, "random_state":42}

def suggest_params_rf(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    return {"n_estimators": n_estimators, "max_depth": max_depth,
            "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf,
            "random_state": 42, "n_jobs": -1}

def suggest_params_xgb(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    lr = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 10.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 10.0)
    return {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": lr,
            "subsample": subsample, "colsample_bytree": colsample, "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda, "verbosity":0, "use_label_encoder":False, "random_state":42}

def suggest_params_lgb(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    lr = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
    num_leaves = trial.suggest_int("num_leaves", 8, 256)
    min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 10.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 10.0)
    return {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": lr,
            "num_leaves": num_leaves, "min_child_samples": min_child_samples,
            "reg_alpha": reg_alpha, "reg_lambda": reg_lambda, "random_state":42}

def suggest_params_mlp(trial, input_dim=None):
    # MLP: 两层结构和学习率
    n_units1 = trial.suggest_int("n_units1", 32, 300)
    n_units2 = trial.suggest_int("n_units2", 16, min(300, n_units1))
    lr = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)
    alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
    return {"hidden_layer_sizes": (n_units1, n_units2), "learning_rate_init": lr, "alpha": alpha,
            "max_iter": 500, "random_state":42}

# 选择建议器映射
SUGGEST_FN = {
    "LogR": suggest_params_logr,
    "SVM": suggest_params_svm,
    "RF": suggest_params_rf,
    "XGB": suggest_params_xgb,
    "LGBM": suggest_params_lgb,
    "MLP": suggest_params_mlp
}

# ========== 每器官每模型的 Optuna 目标函数 ==========

def objective_factory(org, model_name, X, y, n_splits=3):
    # 注意：为了节约时间，这里用 n_splits=3 做内部验证（如果你想更稳健可改为5）
    def objective(trial):
        # suggest params according to model
        if model_name == "MLP":
            params = SUGGEST_FN[model_name](trial, input_dim=X.shape[1])
        else:
            params = SUGGEST_FN[model_name](trial)

        # build model with suggested params
        if model_name == "LogR":
            clf = LogisticRegression(**params)
        elif model_name == "SVM":
            clf = SVC(**params)
        elif model_name == "RF":
            clf = RandomForestClassifier(**params)
        elif model_name == "XGB":
            clf = xgb.XGBClassifier(**params)
        elif model_name == "LGBM":
            clf = lgb.LGBMClassifier(**params)
        elif model_name == "MLP":
            clf = MLPClassifier(**params)
        else:
            raise ValueError(model_name)

        # 简单的内部交叉验证：计算平均 AUC（使用 StratifiedKFold）
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, val_idx in cv.split(X, y):
            X_tr = X[tr_idx]; X_v = X[val_idx]
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(imputer.fit_transform(X_tr))
            X_v = scaler.transform(imputer.transform(X_v))

            try:
                clf.fit(X_tr, y[tr_idx])
                prob = clf.predict_proba(X_v)[:,1]
            except Exception as e:
                # 若模型没有 predict_proba（极少数情况），用 decision_function 通过 sigmoid 估算
                try:
                    dec = clf.decision_function(X_v)
                    prob = 1/(1+np.exp(-dec))
                except:
                    return 0.5  # 非常糟的参数
            # 如果只有单一类，会抛错，跳过
            if len(np.unique(y[val_idx])) < 2:
                continue
            try:
                aucs.append(roc_auc_score(y[val_idx], prob))
            except:
                continue
        if len(aucs)==0:
            return 0.5
        return float(np.mean(aucs))
    return objective

# ========== 批量运行 Optuna（每器官 × 每模型） ==========
def run_optuna_for_all(aligned_data, y, n_trials=N_TRIALS):
    best_params_store = {}  # best_params_store[org][model_name] = params
    os.makedirs(OPTUNA_DIR, exist_ok=True)

    for org in ORGANS:
        best_params_store[org] = {}
        X_org = aligned_data[org]["X"].values
        print(f"\n=== Org {org} Optuna tuning start ===")
        for model_name in BASE_MODELS.keys():
            print(f"  - Tuning {model_name} for organ {org} ...")
            study_name = f"org{org}_{model_name}"
            study_path = os.path.join(OPTUNA_DIR, f"{study_name}_study.pkl")
            json_path = os.path.join(OPTUNA_DIR, f"{study_name}_best_params.json")

            # 如果已经有结果，直接加载（断点续训友好）
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    best_params = json.load(f)
                print(f"    loaded existing params for {study_name}")
                best_params_store[org][model_name] = best_params
                continue

            # study = optuna.create_study(direction="maximize", study_name=study_name)
            sampler = optuna.samplers.TPESampler(seed=42)  # 固定采样器seed
            study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler)


            obj = objective_factory(org, model_name, X_org, y, n_splits=3)
            study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

            best_params = study.best_params
            # Optuna 可能返回 numpy types，转为原生
            def make_serializable(d):
                out = {}
                for k,v in d.items():
                    try:
                        json.dumps({k:v})
                        out[k]=v
                    except TypeError:
                        out[k]=float(v) if hasattr(v,'item') else str(v)
                return out

            best_params = make_serializable(best_params)
            best_params_store[org][model_name] = best_params

            # 保存 study 和 best params
            joblib.dump(study, study_path)
            with open(json_path, "w") as f:
                json.dump(best_params, f, indent=2)
            print(f"    saved best params to {json_path}")

    return best_params_store

# ========== 用最佳参数构造 per-organ base model 集合 ==========
def build_base_models_from_optuna(best_params_store):
    # 返回结构： per_org_models[org][model_name] = estimator instance
    per_org_models = {}
    for org in ORGANS:
        per_org_models[org] = {}
        for model_name, base in BASE_MODELS.items():
            params = best_params_store.get(org, {}).get(model_name, None)
            if params is None:
                # fallback to default
                per_org_models[org][model_name] = clone(base)
            else:
                # 特殊处理 gamma 如果为字符串 "scale"
                if model_name == "SVM" and params.get("gamma", None) == "scale":
                    params["gamma"] = "scale"
                # sklearn / xgboost / lightgbm 接受 dict
                try:
                    if model_name == "LogR":
                        per_org_models[org][model_name] = LogisticRegression(**params)
                    elif model_name == "SVM":
                        per_org_models[org][model_name] = SVC(**params)
                    elif model_name == "RF":
                        per_org_models[org][model_name] = RandomForestClassifier(**params)
                    elif model_name == "XGB":
                        per_org_models[org][model_name] = xgb.XGBClassifier(**params)
                    elif model_name == "LGBM":
                        per_org_models[org][model_name] = lgb.LGBMClassifier(**params)
                    elif model_name == "MLP":
                        per_org_models[org][model_name] = MLPClassifier(**params)
                    else:
                        per_org_models[org][model_name] = clone(base)
                except Exception as e:
                    print(f"Warning: constructing {model_name} with best params failed: {e}. Using default.")
                    per_org_models[org][model_name] = clone(base)
    return per_org_models

# ========== stacking 流程（使用 per_org_models 作为 base 模型） ==========
def run_stacking_with_models(aligned_data, y, per_org_models, selected_models, meta_model):
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    y_train, y_test = y[train_idx], y[test_idx]

    meta_features = {"train": [], "test": []}
    feature_names = []

    for org in ORGANS:
        X_org = aligned_data[org]["X"].values
        X_train, X_test = X_org[train_idx], X_org[test_idx]

        for name in selected_models:
            model = per_org_models[org][name]
            col_name = f"Organ{org}_{name}"
            feature_names.append(col_name)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            train_probs = np.zeros(len(X_train))
            test_probs_list = []

            for tr_idx, val_idx_fold in cv.split(X_train, y_train):
                imputer = SimpleImputer(strategy="median")
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(imputer.fit_transform(X_train[tr_idx]))
                X_v = scaler.transform(imputer.transform(X_train[val_idx_fold]))
                clf = clone(model)
                clf.fit(X_tr, y_train[tr_idx])
                # predict_proba
                try:
                    train_probs[val_idx_fold] = clf.predict_proba(X_v)[:, 1]
                except:
                    # fallback to decision_function
                    dec = clf.decision_function(X_v)
                    train_probs[val_idx_fold] = 1/(1+np.exp(-dec))

                X_test_f = scaler.transform(imputer.transform(X_test))
                try:
                    test_probs_list.append(clf.predict_proba(X_test_f)[:, 1])
                except:
                    dec = clf.decision_function(X_test_f)
                    test_probs_list.append(1/(1+np.exp(-dec)))

            test_probs = np.mean(test_probs_list, axis=0)
            meta_features["train"].append(train_probs)
            meta_features["test"].append(test_probs)

    X_meta_train = np.column_stack(meta_features["train"])
    X_meta_test = np.column_stack(meta_features["test"])

    # 训练 Meta 模型
    clf = clone(meta_model)
    clf.fit(X_meta_train, y_train)
    prob_train = clf.predict_proba(X_meta_train)[:, 1]
    pred_train = (prob_train >= 0.5).astype(int)
    prob_test = clf.predict_proba(X_meta_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(int)

    def get_metrics(y_true, prob, pred):
        acc = accuracy_score(y_true, pred)
        auc_score = roc_auc_score(y_true, prob)
        acc_ci = bootstrap_ci(y_true, pred, accuracy_score)
        auc_ci = bootstrap_ci(y_true, prob, roc_auc_score)
        return {
            "ACC": acc,
            "ACC_95CI": f"[{acc_ci[1]:.3f}, {acc_ci[2]:.3f}]",
            "AUC": auc_score,
            "AUC_95CI": f"[{auc_ci[1]:.3f}, {auc_ci[2]:.3f}]",
            "Sensitivity": recall_score(y_true, pred),
            "Specificity": specificity_score(y_true, pred),
            "F1": f1_score(y_true, pred)
        }

    train_metrics = get_metrics(y_train, prob_train, pred_train)
    test_metrics = get_metrics(y_test, prob_test, pred_test)
    # return train_metrics, test_metrics, clf, feature_names, y_test, prob_test, X_meta_test
    return (
        train_metrics,
        test_metrics,
        clf,
        feature_names,
        y_train,
        X_meta_train,
        y_test,
        prob_test,
        X_meta_test
    )


# ========== 主程序 ==========
def main():
    print("加载数据...")
    aligned_data, y = load_and_align_data()

    # 1) Run Optuna to obtain best params per organ/model
    print("\nStart Optuna tuning for all organs/models...")
    start_time = time.time()
    best_params_store = run_optuna_for_all(aligned_data, y, n_trials=N_TRIALS)
    print(f"Optuna tuning finished in {time.time()-start_time:.1f}s")

    # 保存 best_params_store 整体文件以备查验
    with open(os.path.join(OPTUNA_DIR, "best_params_store.json"), "w") as f:
        json.dump(best_params_store, f, indent=2)

    # 2) 构造 per-organ 最优 base models
    per_org_models = build_base_models_from_optuna(best_params_store)

    # 3) 对所有 base model 组合与 meta models 做 stacking（与你原先流程一致）
    print("\nStart stacking with tuned base models...")
    results = []
    combo_id = 0
    model_names = list(BASE_MODELS.keys())

    for meta_model_name, meta_model in META_MODELS.items():
        print(f"\n=== Meta model: {meta_model_name} ===")
        for size in COMBO_SIZES:
            combo_list = list(combinations(model_names, size))
            for combo in combo_list:
                combo_id += 1
                combo_name = "+".join(combo)
                print(f"→ Training {combo_name} (ID: {combo_id})")

                # train_metrics, test_metrics, meta_clf, feat_names, y_test, prob_test, X_meta_test = \
                (
                    train_metrics,
                    test_metrics,
                    meta_clf,
                    feat_names,
                    y_train,
                    X_meta_train,
                    y_test,
                    prob_test,
                    X_meta_test
                ) =\
                run_stacking_with_models(aligned_data, y, per_org_models, combo, meta_model)

                row = {
                    "Combo_ID": combo_id,
                    "Meta_Model": meta_model_name,
                    "Base_Combo": combo_name,
                    "Combo_Size": size,
                    "Total_Features": len(combo) * len(ORGANS)
                }
                for k, v in train_metrics.items():
                    row[f"Train_{k}"] = v
                for k, v in test_metrics.items():
                    row[f"Test_{k}"] = v
                results.append(row)

                # 保存模型包
                joblib.dump({
                    "meta_model": meta_clf,
                    "features": feat_names,
                    "combo": combo_name,
                    "meta": meta_model_name,
                    "y_test": y_test,
                    "prob_test": prob_test,
                    "X_meta_test": X_meta_test,
                    "y_train": y_train,  # ← 新增
                    "X_meta_train": X_meta_train  # ← 新增
                }, os.path.join(OUTPUT_DIR, f"model_{combo_id}_{combo_name}_{meta_model_name}.pkl"))

    df = pd.DataFrame(results)
    df = df.sort_values(by="Test_AUC", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(OUTPUT_DIR, "results_with_optuna.csv")
    df.to_csv(csv_path, index=False)

    best = df.iloc[0]
    print("\n" + "=" * 80)
    print("最佳模型：")
    print(f"Combo_ID: {best['Combo_ID']}")
    print(f"Meta: {best['Meta_Model']}")
    print(f"Base: {best['Base_Combo']}")
    print(f"Test AUC: {best['Test_AUC']:.3f} {best['Test_AUC_95CI']}")
    print(f"模型保存目录: {OUTPUT_DIR}")
    print(f"Optuna 保存目录: {OPTUNA_DIR}")
    print(f"结果表格: {csv_path}")

if __name__ == "__main__":
    main()
