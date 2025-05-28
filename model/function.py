# ─── 0) 라이브러리 ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import json
import optuna
import shap
import platform

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from kmodes.kprototypes import KPrototypes

# 운영체제별 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac OS
    plt.rc('font', family='AppleGothic')
else: 
    plt.rc('font', family='NanumGothic')

warnings.filterwarnings("ignore")

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


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
    N_TRIALS_RF  = 50

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

    analyze_shap_values(best_rf, X, feature_names=X.columns)

    print(f"\n■ 저장 완료")
    print(f"  • 메트릭  → {metrics_path}")
    print(f"  • 중요도  → {fi_path}")
    return fi_df



# Helper function to find best k based on silhouette (similar to your K-means)
def _get_user_k_choice(k_min, k_max, scores, score_name="Silhouette Score", higher_is_better=True):
    if higher_is_better:
        best_k_metric = int(np.argmax(scores) + k_min)
    else:
        best_k_metric = int(np.argmin(scores) + k_min)

    print(f"\n--- {score_name} for k={k_min} to k={k_max} ---")
    for i, score in enumerate(scores):
        print(f"k={k_min + i}: {score:.4f}")
    print(f"[{score_name} {'max' if higher_is_better else 'min'}] k = {best_k_metric}")

    while True:
        choice = input(
            f"\n클러스터 개수(k)를 직접 선택하세요 [{k_min}~{k_max}] "
            f"(엔터를 누르면 {score_name} 기준 k={best_k_metric} 사용): "
        ).strip()

        if choice == "":
            k_final = best_k_metric
            print(f"▶ {score_name} 기준 k={k_final} 사용")
            break
        if not choice.isdigit():
            print("⚠️ 숫자를 입력해주세요.")
            continue
        k_user = int(choice)
        if k_min <= k_user <= k_max:
            k_final = k_user
            print(f"▶ 사용자가 선택한 k={k_final} 사용")
            break
        else:
            print(f"⚠️ {k_min}~{k_max} 범위 내에서 입력해주세요.")
    return k_final

## 1. Gaussian Mixture Models (GMM)
def cluster_gmm(df: pd.DataFrame, k_min=2, k_max=10, random_state=42) -> tuple[pd.DataFrame, int]:
    print("--- Gaussian Mixture Model (GMM) Clustering ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)
    
    silhouettes, bics, aics = [], [], []
    possible_k_values = list(range(k_min, k_max + 1))

    for k_gmm in possible_k_values:
        gmm = GaussianMixture(n_components=k_gmm, random_state=random_state, n_init=10)
        labels = gmm.fit_predict(X_scaled)
        if len(np.unique(labels)) < 2: # Silhouette score needs at least 2 labels
            silhouettes.append(-1) # Or some other indicator of invalid silhouette
        else:
            silhouettes.append(silhouette_score(X_scaled, labels))
        bics.append(gmm.bic(X_scaled))
        aics.append(gmm.aic(X_scaled))

    print("\n--- Evaluation Metrics ---")
    print("K\tSilhouette\tBIC\t\tAIC")
    for i, k_gmm in enumerate(possible_k_values):
        print(f"{k_gmm}\t{silhouettes[i]:.4f}\t\t{bics[i]:.2f}\t{aics[i]:.2f}")

    # --- Choose k based on Silhouette (user can override) ---
    # Note: For BIC/AIC, lower is better. For Silhouette, higher is better.
    # We will primarily use Silhouette for user choice here for consistency.
    print("\n결정: 실루엣 점수 (높을수록 좋음), BIC/AIC (낮을수록 좋음)를 종합적으로 고려하세요.")
    k_final = _get_user_k_choice(k_min, k_max, silhouettes, "Silhouette Score (GMM)", higher_is_better=True)

    final_gmm = GaussianMixture(n_components=k_final, random_state=random_state, n_init=10)
    df[f"cluster_gmm_{k_final}"] = final_gmm.fit_predict(X_scaled)
    print(f"GMM clustering complete. Labels added to 'cluster_gmm_{k_final}'.")
    return df, k_final

## 2. Agglomerative Hierarchical Clustering
def cluster_hierarchical(df: pd.DataFrame, k_min=2, k_max=10, linkage_method='ward') -> tuple[pd.DataFrame, int]:
    print(f"--- Agglomerative Hierarchical Clustering (Linkage: {linkage_method}) ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    silhouettes = []
    possible_k_values = list(range(k_min, k_max + 1))

    for k_hc in possible_k_values:
        # Linkage must be 'ward' if affinity is 'euclidean' (default) for n_clusters != None
        if linkage_method == 'ward':
            hc = AgglomerativeClustering(n_clusters=k_hc, linkage=linkage_method)
        else:
             hc = AgglomerativeClustering(n_clusters=k_hc, linkage=linkage_method, affinity='euclidean')
        labels = hc.fit_predict(X_scaled)
        if len(np.unique(labels)) < 2:
            silhouettes.append(-1)
        else:
            silhouettes.append(silhouette_score(X_scaled, labels))
    
    k_final = _get_user_k_choice(k_min, k_max, silhouettes, "Silhouette Score (Hierarchical)", higher_is_better=True)

    final_hc = AgglomerativeClustering(n_clusters=k_final, linkage=linkage_method)
    df[f"cluster_hc_{k_final}"] = final_hc.fit_predict(X_scaled)
    print(f"Hierarchical clustering complete. Labels added to 'cluster_hc_{k_final}'.")
    return df, k_final

## 3. DBSCAN
def cluster_dbscan(df: pd.DataFrame, eps=0.5, min_samples=5) -> tuple[pd.DataFrame, int]:
    print(f"--- DBSCAN Clustering (eps={eps}, min_samples={min_samples}) ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_points = list(labels).count(-1)

    print(f"Estimated number of clusters: {n_clusters_found}")
    print(f"Estimated number of noise points: {n_noise_points}")

    if n_clusters_found > 1 and len(labels) > n_noise_points : # Check if non-noise points exist to calculate silhouette
        # Exclude noise points for silhouette calculation
        X_scaled_no_noise = X_scaled[labels != -1]
        labels_no_noise = labels[labels != -1]
        if len(np.unique(labels_no_noise)) > 1: # Silhouette score needs at least 2 labels among non-noise points
             score = silhouette_score(X_scaled_no_noise, labels_no_noise)
             print(f"Silhouette Coefficient (excluding noise): {score:.4f}")
        else:
            print("Silhouette Coefficient: Not enough clusters among non-noise points to calculate.")
    else:
        print("Silhouette Coefficient: Not enough clusters found or too much noise to calculate.")

    df[f"cluster_dbscan_eps{str(eps).replace('.', '')}_ms{min_samples}"] = labels
    print(f"DBSCAN clustering complete. Labels added to 'cluster_dbscan_eps{str(eps).replace('.', '')}_ms{min_samples}'.")
    return df, n_clusters_found

## 4. K-Prototypes
def cluster_kprototypes(df_original: pd.DataFrame, categorical_cols_indices: list, 
                        k_min=2, k_max=10, random_state=42, n_init_val=10) -> tuple[pd.DataFrame, int]:
    print("--- K-Prototypes Clustering ---")
    if not categorical_cols_indices:
        print("⚠️ K-Prototypes_Warning: No categorical column indices provided. Consider using K-Means if all data is numerical.")
        # Fallback or error, as K-Prototypes needs categorical features.
        # For this example, we'll proceed, but it might not be optimal.

    X = df_original.values # K-Prototypes handles numerical scaling internally if needed, based on its logic
                           # or expects numericals to be somewhat comparable or user scales them prior.
                           # For consistency, it's often better to scale numerical features beforehand.
                           # However, the `kmodes` library documentation isn't explicit about auto-scaling.
                           # Let's assume for now the user might pass unscaled numericals or scaled ones.
                           # For a robust function, explicit scaling of numerical columns before passing to KPrototypes is better.

    costs = [] # K-Prototypes has a 'cost_' attribute (sum of dissimilarities)
    silhouettes = [] # Silhouette can be tricky with mixed types, depends on metric used
    possible_k_values = list(range(k_min, k_max + 1))

    for k_kp in possible_k_values:
        kproto = KPrototypes(n_clusters=k_kp, init='Cao', verbose=0, random_state=random_state, n_init=n_init_val)
        # Ensure categorical_cols_indices are valid
        valid_categorical_indices = [idx for idx in categorical_cols_indices if idx < X.shape[1]]
        if len(valid_categorical_indices) != len(categorical_cols_indices):
            print(f"⚠️ K-Prototypes_Warning: Some categorical indices are out of bounds for data with {X.shape[1]} columns.")
        
        clusters = kproto.fit_predict(X, categorical=valid_categorical_indices)
        costs.append(kproto.cost_)
        
        # Silhouette score for K-Prototypes is not straightforward as it uses mixed distance metrics.
        # The library itself doesn't provide a direct way to calculate a meaningful silhouette score
        # that correctly combines Hamming and Euclidean distances in the way silhouette_score from sklearn expects.
        # We will skip Silhouette for K-Prototypes for simplicity here, focusing on its own cost.
        # If you need it, you'd have to implement a custom distance metric or find a specialized package.
        silhouettes.append(np.nan) # Placeholder

    print("\n--- Evaluation Metrics (K-Prototypes) ---")
    print("K\tCost (Lower is Better)")
    for i, k_kp in enumerate(possible_k_values):
        print(f"{k_kp}\t{costs[i]:.2f}")
    
    k_final = _get_user_k_choice(k_min, k_max, costs, "Cost (K-Prototypes)", higher_is_better=False)

    final_kproto = KPrototypes(n_clusters=k_final, init='Cao', verbose=0, random_state=random_state, n_init=n_init_val)
    valid_categorical_indices = [idx for idx in categorical_cols_indices if idx < X.shape[1]]
    df_original[f"cluster_kproto_{k_final}"] = final_kproto.fit_predict(X, categorical=valid_categorical_indices)
    print(f"K-Prototypes clustering complete. Labels added to 'cluster_kproto_{k_final}'.")
    return df_original, k_final



def analyze_shap_values(model, X_features, feature_names=None):
    """
    RandomForest 계열 모델과 학습용 데이터(X_features)를 입력으로 받아
    각 피처의 SHAP 값을 계산하고, global importance(bar)와 summary plot을 출력합니다.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier or RandomForestRegressor
        학습된 RandomForest 모델
    X_features : pandas.DataFrame or numpy.ndarray
        SHAP 값을 계산할 피처 데이터 (train_data)
    feature_names : list of str, optional
        피처 이름 리스트. None이면 X_features.columns 사용
    """
    # 1) Explainer 생성
    explainer = shap.TreeExplainer(model)

    # 2) SHAP 값 계산
    shap_values = explainer.shap_values(X_features)

    # 3) 분류 모델일 경우 양성 클래스(1)의 shap value만 선택
    if isinstance(shap_values, list) and len(shap_values) > 1:
        # shap_values[0]: 음성 클래스, shap_values[1]: 양성 클래스
        shap_val = shap_values[1]
    else:
        shap_val = shap_values

    # 4) feature 이름 설정
    names = feature_names if feature_names is not None else (
        X_features.columns.tolist() if hasattr(X_features, 'columns') else None
    )

    # 5) Bar 형태의 global importance
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_val, X_features, feature_names=names, plot_type="bar", show=False)
    plt.title("Feature Importance (SHAP)")
    plt.tight_layout()
    plt.show()

    # 6) Beeswarm 형태의 summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_val, X_features, feature_names=names, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.show()