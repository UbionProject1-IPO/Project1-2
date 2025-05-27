#!/usr/bin/env python3
# ───────────────────────── scale.py ─────────────────────────
"""
CSV에 권장 스케일링 적용 ─ 컬럼 이름 유지 버전
-------------------------------------------------------------
상단 상수만 수정 → 바로 실행
$ python scale.py                 # 상수 사용
$ python scale.py -i raw.csv ...   # CLI 인자로 덮어쓰기
-------------------------------------------------------------
필수: pandas, numpy, scikit-learn>=1.4, joblib
"""

# ─── 0) ★ 경로 상수 (필요 시 수정) ────────────────────────────
INPUT_PATH    = "./data/주가수익률/주가수익률_CSI_noFundmental.csv"         # 입력 CSV
OUTPUT_PATH   = "./data/주가수익률/주가수익률_CSI_noFundmental_scaled.csv"      # 전처리 결과 CSV

# 1) 라이브러리
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    PowerTransformer,
)

def main():
    # 2) 데이터 로드
    df = pd.read_csv(INPUT_PATH)
    print(f"[로드] {INPUT_PATH} → shape={df.shape}")

    # 3) 컬럼 그룹 정의 (필요 시 수정)
    log_cols  = ["1~5일 수익률 표준편차", "6개월 확약", "VIXCLS", "업력", "asset_turnover"]
    pt_cols   = ["kospi200(-20)", "op_margin", "roe", "roa", "net_margin"]
    rb_cols   = ["nasdaq(-20)", "환율변동률(-20)", "putcall(-20)", "oi_ta", "ni_oi_ratio"]
    std_cols  = ["equity_ratio", "debt_asset"]
    pct_cols  = ["우리사주조합", "기관투자자", "일반투자자"]
    # '경기국면'은 여기서 제외 → 그대로 남음
    # (추가로 변화 없이 남길 다른 컬럼이 있다면 여기에 추가하세요)

    # 4) 변환 적용
    # 4-1) 로그 + StandardScaler
    df[log_cols] = StandardScaler().fit_transform(
        np.log1p(df[log_cols])
    )

    # 4-2) Yeo-Johnson
    df[pt_cols] = PowerTransformer(method="yeo-johnson").fit_transform(
        df[pt_cols]
    )

    # 4-3) RobustScaler
    df[rb_cols] = RobustScaler().fit_transform(
        df[rb_cols]
    )

    # 4-4) StandardScaler
    df[std_cols] = StandardScaler().fit_transform(
        df[std_cols]
    )

    # 4-5) MinMaxScaler (0-1)
    df[pct_cols] = MinMaxScaler().fit_transform(
        df[pct_cols]
    )

    # 5) 결과 저장
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[저장] 스케일링 완료 → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
