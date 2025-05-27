# ─── 0) 라이브러리 ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import json
import optuna
import function

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


RESULT_DIR  = "model/model_result"

# ─── 1) (예시) 데이터프레임 로드 ────────────────────────────
df = pd.read_csv("data/주가수익률_final.csv", index_col=0)
df.drop(columns=["회사명", "stock_code"], inplace=True)
df = df.dropna().reset_index(drop=True)

df, k_final = function.clustering(df)

CLUSTER_COL = f"cluster_k{k_final}" 

# ─── 2) 결과 확인 ─────────────────────────────────────────
print("\n[샘플 결과]")
print(df.head())

# ─── 3) 각 군집별 기업 수 확인 ──────────────────────────
label_counts = df[f"cluster_k{k_final}"].value_counts().sort_index()
print("\n[군집별 기업 수]")
for lbl, cnt in label_counts.items():
    print(f"  라벨 {lbl}: {cnt}개")


# ─── 11) 결과 확인 ─────────────────────────────────────────
print("\n[샘플 결과]")
print(df.head())

fi_df = function.feautre_engineering(df, k_final, "국면_clustering")

# ── top-10 리스트 추출 ─────────────────────────────────────────
top10_features = fi_df.head(10)["feature"].tolist()

# ── 조건 분기: 원하는 컬럼이 top-10 내부에 있는지 확인 ───────
TARGET_FEATURE = "국면"      # <- 여기에 확인할 컬럼명을 입력

if TARGET_FEATURE in top10_features:
    print(f"『{TARGET_FEATURE}』이(가) Top-10 안에 있습니다. 후속 코드를 실행합니다.")
    
    # ── 0) 준비 ────────────────────────────────────────────────────
    import os, pandas as pd, matplotlib.pyplot as plt

    # 결과를 넣을 하위 폴더 생성
    STAT_DIR = os.path.join(RESULT_DIR, "target0_vs_1")
    os.makedirs(STAT_DIR, exist_ok=True)

    # ── 1) target 값별 데이터프레임 분리 ───────────────────────────
    mask0, mask1 = (y == 0), (y == 1)
    X0, X1 = X[mask0], X[mask1]

    # ── 2) 기초 통계량 계산 & CSV 저장 ────────────────────────────
    stats0 = X0.describe().T          # target=0
    stats1 = X1.describe().T          # target=1

    stats_combined = stats0.add_suffix("_0").join(
                    stats1.add_suffix("_1"))

    stats_path = os.path.join(STAT_DIR, "basic_stats_target0_vs_1.csv")
    stats_combined.to_csv(stats_path, encoding="utf-8-sig")
    print(f"■ 기초 통계량 CSV 저장 → {stats_path}")

    # ── 3) 컬럼별 box-plot 저장 ──────────────────────────────────
    plot_dir = os.path.join(STAT_DIR, "boxplots")
    os.makedirs(plot_dir, exist_ok=True)

    for col in X.columns:
        plt.figure(figsize=(10, 5))
        plt.boxplot([X0[col].dropna(), X1[col].dropna()],
                    labels=["target=0", "target=1"])
        plt.title(f"{col} (target별 분포)")
        plt.ylabel(col)
        plt.tight_layout()

        plot_path = os.path.join(plot_dir, f"{col}_boxplot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    print(f"■ box-plot PNG 저장 → {plot_dir} 폴더 내 다수 파일")

    X0_clustered, X0_k_final = function.clustering(X0)
    X1_clustered, X1_k_final = function.clustering(X1)

    X0_fi_df = function.feautre_engineering(X0_clustered, X0_k_final, "국면0_clustering")
    X1_fi_df = function.feautre_engineering(X1_clustered, X1_k_final, "국면1_clustering")

else:
    print(f"『{TARGET_FEATURE}』이(가) Top-10 안에 없으므로 후속 코드를 건너뜁니다.")
    