import pandas as pd
import yfinance as yf
import pymannkendall as mk
from pathlib import Path

# ── 1. 데이터 다운로드 ────────────────────────────────────────────
TICKER   = "^KS200"                       # KOSPI 200 지수 기호
START, END = "2015-01-01", "2024-06-27"

df = (
    yf.download(TICKER, start=START, end=END, progress=False)
      .loc[:, ["Close"]]                 # 종가만 사용
      .rename(columns={"Close": "close"})
)

# ── 2. 월별 Mann-Kendall 검정 함수 ───────────────────────────────
def mk_regime(series, alpha: float = 0.05) -> str:
    """pymannkendall 결과를 '상승·하락·무추세'로 변환"""
    if len(series) < 3:             # 관측치 부족
        return "데이터 부족"
    result = mk.original_test(series)
    if result.p <= alpha:
        if result.trend == "increasing":
            return "상승"
        elif result.trend == "decreasing":
            return "하락"
    return "무추세"

# ── 3. 월별 그룹핑 & 국면 판단 ──────────────────────────────────
monthly_regime = (
    df.groupby(df.index.to_period("M"))["close"]
      .apply(mk_regime)
      .reset_index()
)

monthly_regime.columns = ["날짜(월)", "국면"]
monthly_regime["날짜(월)"] = monthly_regime["날짜(월)"].dt.strftime("%Y-%m")

# ── 4. CSV 저장 ─────────────────────────────────────────────────
OUT_FILE = Path("kospi200_mann_kendall_201501_202406.csv")
monthly_regime.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

print(f"✓ CSV 저장 완료 → {OUT_FILE.resolve()}")