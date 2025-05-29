import os, pandas as pd, matplotlib.pyplot as plt
import function

df = pd.read_csv("data/주가수익률/주가수익률_CSI.csv")
df.drop(columns=["회사명", "stock_code", "상장일"], inplace=True)

STAT_DIR = os.path.join("model/model_result_before")
os.makedirs(STAT_DIR, exist_ok=True)

TARGET_FEATURE = "경기국면"

# ── 1) target 값별 데이터프레임 분리 ───────────────────────────
mask0, mask1 = df[TARGET_FEATURE] == 0, df[TARGET_FEATURE] == 1
X0, X1 = df.loc[mask0].copy(), df.loc[mask1].copy()

# 타깃-열 & 불필요 열 제거 (선택)
drop_cols = [TARGET_FEATURE, "cluster_k2"]  # 있으면 제거
X0.drop(columns=[c for c in drop_cols if c in X0.columns], inplace=True)
X1.drop(columns=[c for c in drop_cols if c in X1.columns], inplace=True)

# ── 2) 기초 통계량 계산 & CSV 저장 ────────────────────────────
stats0 = X0.describe().T.add_suffix("_0")
stats1 = X1.describe().T.add_suffix("_1")
stats_combined = stats0.join(stats1)
stats_path = os.path.join(STAT_DIR, "basic_stats_target0_vs_1.csv")
stats_combined.to_csv(stats_path, encoding="utf-8-sig")
print(f"■ 기초 통계량 CSV 저장 → {stats_path}")

# ── 3) 수치형 컬럼만 box-plot 저장 ────────────────────────────
numeric_cols = X0.select_dtypes(include="number").columns   # ← 핵심!
plot_dir = function.get_next_boxplot_dir(STAT_DIR)
os.makedirs(plot_dir, exist_ok=True)

for col in numeric_cols:
    s0 = pd.to_numeric(X0[col], errors="coerce").dropna()
    s1 = pd.to_numeric(X1[col], errors="coerce").dropna()
    if s0.empty and s1.empty:         # 둘 다 값이 없으면 스킵
        continue

    plt.figure(figsize=(10, 5))
    plt.boxplot([s0, s1], labels=["target=0", "target=1"])
    plt.title(f"{col} (target별 분포)")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{col}_boxplot.png"), dpi=300)
    plt.close()

print(f"■ box-plot PNG 저장 → {plot_dir} 폴더 내 다수 파일")

# ── 4) 클러스터링·피처 중요도 ────────────────────────────────
#  ※ clustering() 내부에서 필요-없는 열 제거를 하지 않는다면 이미 제거한 X0/X1 그대로 사용

raw_data = pd.read_csv("data/주가수익률_raw/주가수익률_base_selected.csv")
raw_data.drop(columns=["회사명", "stock_code", "상장일"], inplace=True)

X0_clustered, k0 = function.clustering(X0.copy())
X0_clustered = raw_data.join(
    X0_clustered[f"cluster_k{k0}"], how="inner"  # ← 인덱스 교집합만 유지
)
X0_clustered.to_csv("data/clustered_before/국면0_clustering.csv", encoding="utf-8-sig")
fi0 = function.feature_engineering(X0_clustered, k0, "국면0_clustering")

X1_clustered, k1 = function.clustering(X1.copy())
X1_clustered = raw_data.join(
    X1_clustered[f"cluster_k{k1}"], how="inner"  # ← 인덱스 교집합만 유지
)
X1_clustered.to_csv("data/clustered_before/국면1_clustering.csv", encoding="utf-8-sig")
fi1 = function.feature_engineering(X1_clustered, k1, "국면1_clustering")
