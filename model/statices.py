# ─── 0) 라이브러리 ────────────────────────────────────────────
import os, numpy as np, pandas as pd

# ─── 1) 핵심 함수 ─────────────────────────────────────────────
def save_basic_stats(
    df: pd.DataFrame,
    csv_path: str,
    percentiles=(0.05, 0.25, 0.5, 0.75, 0.95),
    encoding: str = "utf-8-sig"
) -> pd.DataFrame:
    """
    숫자형 컬럼의 기초통계량을 계산해 CSV로 저장하고 DataFrame을 반환
    """
    num_df = df.select_dtypes(include=[np.number]).copy()

    # 1) describe() 기반 기본 통계
    desc = num_df.describe(percentiles=percentiles).T.rename(columns={"50%": "median"})

    # 2) 추가 지표 계산
    extra = pd.DataFrame(index=num_df.columns)
    extra["var"]  = num_df.var(ddof=1)  # ← 분산 추가
    extra["IQR"]  = num_df.quantile(0.75) - num_df.quantile(0.25)
    extra["MAD"]  = num_df.apply(lambda s: np.median(np.abs(s - s.median())), axis=0)
    extra["CV"]   = desc["std"] / desc["mean"].replace(0, np.nan)
    extra["skew"] = num_df.skew()
    extra["kurt"] = num_df.kurt()
    extra["SEM"]  = desc["std"] / np.sqrt(desc["count"])
    extra["missing_pct"] = (len(df) - desc["count"]) / len(df) * 100

    # IQR-rule 이상치 개수
    def outlier_cnt(s):
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        return ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    extra["outliers"] = num_df.apply(outlier_cnt, axis=0)

    # 3) 통합·정렬·저장
    stats = pd.concat([desc, extra], axis=1)

    # 최종 열 순서 정의
    percentile_cols = [f"{int(p*100)}%" for p in percentiles if p not in (0.5,)]
    final_cols = ["count", "missing_pct", "mean", "median",
                  "std", "var", "SEM", "min", *percentile_cols,
                  "max", "IQR", "MAD", "CV", "skew", "kurt", "outliers"]
    stats = stats[final_cols]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    stats.to_csv(csv_path, encoding=encoding)
    print(f"■ 기초통계량 CSV 저장 → {csv_path}")

    return stats


# ─── 2) 사용 예시 ────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("data/주가수익률/주가수익률_base.csv")
    save_basic_stats(df, "./model/model_result/basic_stats.csv")
