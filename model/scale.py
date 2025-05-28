#!/usr/bin/env python3
# ───────────────────────── scale_for_kmeans.py ─────────────────────────
"""
K-means 클러스터링용 추천 전처리
 - 식별자·범주형 제외
 - 왜도 큰 양수형 변수에 log1p
 - 모든 수치형 변수 StandardScaler
 - '경기국면'은 결과에 그대로 붙여둠
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─── 상수 (필요 시 수정) ───────────────────────────────────────────
INPUT_PATH  = "./data/주가수익률/주가수익률_CSI_noFundmental.csv"
OUTPUT_PATH = "./data/주가수익률/주가수익률_CSI_noFundmental_kmeans_scaled.csv"

def main():
    # 1) 데이터 로드
    df = pd.read_csv(INPUT_PATH)
    print(f"[로드] {INPUT_PATH}  shape={df.shape}")

    # 2) 제외할 컬럼
    id_cols = ["stock_code", "회사명"]                  # 식별자
    cat_cols = ["경기국면"]                              # 범주형
    keep_cols = id_cols + cat_cols

    # 3) 수치형만 분리
    num_df = df.drop(columns=keep_cols, errors="ignore")
    # 컬럼별 최소값·왜도 확인
    skew = num_df.skew().abs()
    # 4) 왜도 >1 & 최솟값 >=0 인 컬럼 → 로그 변환
    log_cols = skew[(skew > 1) & (num_df.min() >= 0)].index.tolist()
    if log_cols:
        num_df[log_cols] = np.log1p(num_df[log_cols])
        print(f"  • Log1p 변환: {log_cols}")
    else:
        print("  • Log1p 대상 없음")

    # 5) StandardScaler 적용 (모든 수치형)
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_df)
    num_df[:] = num_scaled
    print(f"  • StandardScaler 적용: {list(num_df.columns)}")

    # 6) 결과 병합
    result = pd.concat([
        df[id_cols],        # 식별자 (클러스터링용은 제외해도 됨)
        num_df,             # 스케일된 수치형
        df[cat_cols]        # 원본 경기국면
    ], axis=1)

    # 7) 저장
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"[저장] 클러스터링용 스케일링 완료 → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
