# ─── 0) 라이브러리 ─────────────────────────────────────────────
import os, json, warnings, joblib, optuna
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ─── 상수 설정 ────────────────────────────────────────────────
RANDOM_STATE = 42
N_SPLITS      = 5
SCORING       = "f1"
N_TRIALS_RF   = 200

path_name   = "economic_clustered_data.csv"
CLUSTER_COL = "cluster_k2"          # 타깃 컬럼
RESULT_DIR  = "model/model_result/feature"

# ─── 1) 데이터 로드 ────────────────────────────────────────────
df_path = os.path.join("model/model_result/clustering", path_name)
df = pd.read_csv(df_path, index_col=0)
X, y = df.drop(columns=[CLUSTER_COL]), df[CLUSTER_COL]

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ─── 2) Optuna objective ─────────────────────────────────────
def rf_objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 1000),
        "max_depth":         trial.suggest_int("max_depth", 2, 10, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap":         trial.suggest_categorical("bootstrap", [True, False]),
        "n_jobs":            -1,
        "random_state":      RANDOM_STATE,
    }
    clf = RandomForestClassifier(**params)
    return cross_val_score(clf, X, y, cv=cv,
                           scoring=SCORING, n_jobs=-1).mean()

# ─── 3) 하이퍼파라미터 탐색 ───────────────────────────────────
print("▶ RandomForest 하이퍼파라미터 탐색 중…")
study_rf = optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_rf.optimize(rf_objective, n_trials=N_TRIALS_RF, n_jobs=1, show_progress_bar=True)

best_rf = RandomForestClassifier(**study_rf.best_params,
                                 n_jobs=-1, random_state=RANDOM_STATE)

# ─── 4) 교차검증 (precision·recall·f1·roc_auc) ───────────────
multi_scoring = {"precision": "precision",
                 "recall":    "recall",
                 "f1":        "f1",
                 "roc_auc":   "roc_auc"}

cv_res = cross_validate(best_rf, X, y, cv=cv,
                        scoring=multi_scoring,
                        n_jobs=-1, return_train_score=False)

# ─── 5) 메트릭 요약 데이터프레임 ──────────────────────────────
metrics = ["precision", "recall", "f1", "roc_auc"]
summary = {f"{m}_mean": cv_res[f"test_{m}"].mean()
           for m in metrics}
summary.update({f"{m}_std": cv_res[f"test_{m}"].std()
                for m in metrics})

print("\n◆ 교차검증 결과 (mean ± std)")
for m in metrics:
    print(f"{m:<9}: {summary[f'{m}_mean']:.4f} ± {summary[f'{m}_std']:.4f}")

# ─── 6) 결과 저장 폴더 생성 ──────────────────────────────────
os.makedirs(RESULT_DIR, exist_ok=True)

# ── 6-A) 메트릭 JSON 저장 (f1, precision, recall) ────────────
metrics_path = os.path.join(RESULT_DIR, "economic_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump({k: round(v, 6) for k, v in summary.items()
               if k.startswith(("f1_", "precision_", "recall_"))},
              f, ensure_ascii=False, indent=4)

# ── 6-B) 전체 데이터 학습 후 feature importance 저장 ────────
best_rf.fit(X, y)
fi_df = (pd.Series(best_rf.feature_importances_, index=X.columns, name="importance")
           .sort_values(ascending=False)
           .reset_index()
           .rename(columns={"index": "feature"}))

fi_path = os.path.join(RESULT_DIR, "economic_feature_importance.csv")
fi_df.to_csv(fi_path, index=False, encoding="utf-8-sig")

print(f"\n■ 저장 완료")
print(f"  • 메트릭  → {metrics_path}")
print(f"  • 중요도  → {fi_path}")