# ─── 0) 라이브러리 ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

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

    # ─── 3) Elbow & 실루엣 평가 함수 ───────────────────────────

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

def feautre_engineering(df, k_final, file_name):
    CLUSTER_COL = f"cluster_k{k_final}" 
    RESULT_DIR  = "model/model_result"

    RANDOM_STATE = 42
    N_SPLITS      = 5
    SCORING       = "f1"
    N_TRIALS_RF   = 20

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
    metrics_path = os.path.join(RESULT_DIR, f"{file_name}_metrics.json")
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

    fi_path = os.path.join(RESULT_DIR, f"{file_name}_feature_importance.csv")
    fi_df.to_csv(fi_path, index=False, encoding="utf-8-sig")

    print(f"\n■ 저장 완료")
    print(f"  • 메트릭  → {metrics_path}")
    print(f"  • 중요도  → {fi_path}")

    return fi_df