# ─── 0) 라이브러리 ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ─── 1) (예시) 데이터프레임 로드 ────────────────────────────
df = pd.read_csv("data/경기종합지수.csv", index_col=0)

# ─── 2) 데이터 스케일링 ────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.values)

# ─── 3) Elbow & 실루엣 평가 함수 ───────────────────────────
def evaluate_kmeans(data, k_min=2, k_max=10, random_state=42):
    inertias, silhouettes = [], []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(data, labels))
    return inertias, silhouettes

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

# ─── 9) 결과 확인 ─────────────────────────────────────────
print("\n[샘플 결과]")
print(df.head())

# ─── 10) 각 군집별 기업 수 확인 ──────────────────────────
label_counts = df[f"cluster_k{k_final}"].value_counts().sort_index()
print("\n[군집별 기업 수]")
for lbl, cnt in label_counts.items():
    print(f"  라벨 {lbl}: {cnt}개")

# 0‧1만 따로 보고 싶다면
if 0 in label_counts.index:
    print(f"\n라벨 0 개수: {label_counts[0]}")
if 1 in label_counts.index:
    print(f"라벨 1 개수: {label_counts[1]}")


# ─── 11) 결과 확인 ─────────────────────────────────────────
print("\n[샘플 결과]")
print(df.head())

df.to_csv("model/model_result/clustering/economic_clustered_data.csv", index=False)