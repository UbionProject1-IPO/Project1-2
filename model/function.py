# ─── 0) 라이브러리 ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import json
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


def evaluate_kmeans(data, k_min=2, k_max=10, random_state=42):
    inertias, silhouettes = [], []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(data, labels))
    return inertias, silhouettes

def clustering(df) -> pd.DataFrame:
    # ─── 2) 데이터 스케일링 ────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
    df_scaled.to_csv("data/주가수익률/주가수익률_Temp_scaled.csv", encoding="utf-8-sig", index=True)

    # ─── 4) K 범위 평가 ────────────────────────────────────────
    K_MIN, K_MAX = 2, 10
    inertias, silhouettes = evaluate_kmeans(X_scaled, K_MIN, K_MAX)

    # ─── 5) Elbow & 실루엣 그래프 ─────────────────────────────
    plt.figure(figsize=(10, 4))

    # (a) Elbow
    plt.subplot(1, 2, 1)
    plt.plot(range(K_MIN, K_MAX + 1), inertias, marker="o")
    plt.title("Elbow Method (Inertia)")
    plt.xlabel("k"), plt.ylabel("Inertia")

    # (b) Silhouette
    plt.subplot(1, 2, 2)
    plt.plot(range(K_MIN, K_MAX + 1), silhouettes, marker="o")
    plt.title("Silhouette Score")
    plt.xlabel("k"), plt.ylabel("Score")

    plt.tight_layout()
    plt.show()

    # ─── 6) 기본 추천 k 계산 ───────────────────────────────────
    best_k_sil = int(np.argmax(silhouettes) + K_MIN)           # 실루엣 최대
    diff2 = np.diff(inertias, n=2)                             # Elbow 추정
    best_k_elbow = int(np.argmin(diff2) + K_MIN + 1)

    print(f"[실루엣 최대] k = {best_k_sil}")
    print(f"[Elbow]     k = {best_k_elbow}")

    # ─── 7) 사용자 입력으로 k 결정 ─────────────────────────────
    while True:
        choice = input(
            f"\n클러스터 개수(k)를 직접 선택하세요 [{K_MIN}~{K_MAX}] "
            f"(엔터를 누르면 실루엣 기준 k={best_k_sil} 사용): "
        ).strip()

        if choice == "":
            k_final = best_k_sil
            print(f"▶ 실루엣 기준 k={k_final} 사용")
            break

        if not choice.isdigit():
            print("⚠️ 숫자를 입력해주세요.")
            continue

        k_user = int(choice)
        if K_MIN <= k_user <= K_MAX:
            k_final = k_user
            print(f"▶ 사용자가 선택한 k={k_final} 사용")
            break
        else:
            print(f"⚠️ {K_MIN}~{K_MAX} 범위 내에서 입력해주세요.")

    # ─── 8) 최종 K-means 학습 & 컬럼 추가 ─────────────────────
    kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init="auto")
    df[f"cluster_k{k_final}"] = kmeans_final.fit_predict(X_scaled)
    return df, k_final

def feature_engineering(df, k_final, file_name):
    import os, json, optuna, numpy as np, pandas as pd
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.exceptions import UndefinedMetricWarning
    import warnings

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    CLUSTER_COL = f"cluster_k{k_final}"
    RESULT_DIR  = "model/model_result"

    RANDOM_STATE = 42
    N_SPLITS     = 5
    N_TRIALS_RF  = 200

    # ─── 1) 데이터·결측치 처리 ────────────────────────────────
    X, y = df.drop(columns=[CLUSTER_COL]), df[CLUSTER_COL]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)

    n_classes = y.nunique()
    if n_classes < 2:
        print(f"[{file_name}] 단일 클래스 → feature importance 산출 불가")
        return pd.DataFrame()

    # ─── 2) 스코어링 지정(다중/이진 분기) ─────────────────────
    if n_classes == 2:
        primary_score = "f1"
        multi_scoring = {
            "precision": "precision",
            "recall":    "recall",
            "f1":        "f1",
            "roc_auc":   "roc_auc",
        }
    else:
        primary_score = "f1_macro"        # Optuna 목적함수
        multi_scoring = {
            "precision": "precision_macro",
            "recall":    "recall_macro",
            "f1":        "f1_macro",
            "roc_auc":   "roc_auc_ovr_weighted",   # 필요 없으면 삭제
        }

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # ─── 3) Optuna objective ─────────────────────────────────
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
        score = cross_val_score(clf, X, y, cv=cv,
                                scoring=primary_score,
                                n_jobs=-1, error_score=np.nan)
        if np.isnan(score).all():
            raise ValueError("all scores are NaN")   # trial 실패 처리
        return np.nanmean(score)

    print("▶ RandomForest 하이퍼파라미터 탐색 중…")
    study_rf = optuna.create_study(direction="maximize",
                                   sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_rf.optimize(rf_objective, n_trials=N_TRIALS_RF,
                      catch=(Exception,), show_progress_bar=True)

    # ─── 4) 최적 모델 정의 & 교차검증 ─────────────────────────
    if not study_rf.best_trials:          # 모든 trial 실패
        print(f"[{file_name}] Optuna 성공 trial 없음 → 건너뜀")
        return pd.DataFrame()

    best_rf = RandomForestClassifier(**study_rf.best_params,
                                     n_jobs=-1, random_state=RANDOM_STATE)
    cv_res = cross_validate(best_rf, X, y, cv=cv,
                            scoring=multi_scoring,
                            n_jobs=-1, return_train_score=False)

    # ─── 5) 결과 요약&저장 ───────────────────────────────────
    metrics = list(multi_scoring.keys())
    summary = {f"{m}_mean": cv_res[f"test_{m}"].mean() for m in metrics}
    summary.update({f"{m}_std": cv_res[f"test_{m}"].std() for m in metrics})

    os.makedirs(RESULT_DIR, exist_ok=True)
    metrics_path = os.path.join(RESULT_DIR, f"{file_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    best_rf.fit(X, y)
    fi_df = (pd.Series(best_rf.feature_importances_, index=X.columns, name="importance")
               .sort_values(ascending=False)
               .reset_index()
               .rename(columns={"index": "feature"}))

    fi_path = os.path.join(RESULT_DIR, f"{file_name}_feature_importance.csv")
    fi_df.to_csv(fi_path, index=False, encoding="utf-8-sig")

    print(f"\n■ 저장 완료")
    print(f"  • 메트릭  → {metrics_path}")
    print(f"  • 중요도  → {fi_path}")
    return fi_df
