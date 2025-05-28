import os, glob, re
import pandas as pd

# ── 경로 설정 ──────────────────────────────────────────────
PATH_SECTION = "data/경기국면"                 # 지표별 CSV 폴더
PATH_PRICE   = "data/주가수익률"              # 주가수익률 폴더
BASE_FILE    = os.path.join(PATH_PRICE, "주가수익률_base.csv")

# ── 주가수익률 기본 테이블 로드 ─────────────────────────────
df_base = pd.read_csv(BASE_FILE)

# ── 파일명에서 지표 코드(ESI, CSI …) 추출용 패턴 ───────────
regex = re.compile(r"경기국면\((.+?)\)\.csv$")

# ── 경기국면 파일 순회 ─────────────────────────────────────
for fpath in glob.glob(os.path.join(PATH_SECTION, "경기국면(*).csv")):
    m = regex.search(os.path.basename(fpath))
    if not m:
        # 예상 패턴이 아니면 건너뜀
        continue
    code = m.group(1)                    # 예: 'ESI', 'CSI' …

    # ── 경기국면 데이터 로드 & 전처리 ──────────────────────
    df_section = (
        pd.read_csv(fpath)
          .loc[:, ["회사명", "경기국면"]]   # 필요한 컬럼만
          .drop_duplicates(subset="회사명") # 중복 제거
    )
    # ── 병합 ──────────────────────────────────────────────
    merged = pd.merge(
        df_base,
        df_section,
        on="회사명",
        how="left",            # 회사명이 없는 행은 NaN 유지
        validate="one_to_one"  # 오류 방지: 중복 키 체크
    )
    # ── 결과 저장 ─────────────────────────────────────────
    out_path = os.path.join(PATH_PRICE, f"주가수익률_{code}.csv")
    merged.to_csv(out_path, index=False)
    print(f"[✓] 저장 완료 → {out_path}")
