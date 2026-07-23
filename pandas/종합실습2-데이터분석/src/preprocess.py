"""
파일명: preprocess.py
설명: 파이프라인 2단계 — 결측치·중복·이상치를 처리해 분석 가능한 데이터셋을 만든다.
기능:
    - _parse_years(): 경력 컬럼의 문자열 표기('Less than 1 year' 등)를 숫자로 변환한다.
    - preprocess(): 관심 컬럼만 추린 뒤 결측치·중복·이상치를 제거한 DataFrame과
      처리 내역(PreprocessResult)을 반환한다.
"""

from __future__ import annotations

import pandas as pd

from .config import COLUMNS_OF_INTEREST, FEATURE_NUMERIC_COLUMNS, TARGET_COLUMN
from .reporting import Reporter
from .results import PreprocessResult


def _parse_years(value: object) -> float | None:
    """경력 컬럼의 문자열 표기를 숫자로 변환한다.

    원본은 'Less than 1 year' / 'More than 50 years' 같은 문자열을 섞어 쓴다.
    """
    text = str(value)
    if text == "Less than 1 year":
        return 0.5
    if text == "More than 50 years":
        return 51.0
    try:
        return float(text)
    except ValueError:
        return None


def preprocess(
    df: pd.DataFrame, reporter: Reporter
) -> tuple[pd.DataFrame, PreprocessResult]:
    """관심 컬럼만 추린 뒤 결측치·중복·이상치를 제거한다."""
    reporter.section("2. 전처리 (결측치 · 중복 · 이상치)")

    df = df[COLUMNS_OF_INTEREST].copy()
    raw_rows = len(df)
    missing_by_column = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())

    reporter.log("🧹 [처리 전 현황]")
    reporter.log(f"- 전체 행: {raw_rows:,}건")
    reporter.log("- 컬럼별 결측치:")
    for column, count in missing_by_column.items():
        reporter.log(f"    · {column}: {count:,}건 ({count / raw_rows:.1%})")
    reporter.log(f"- 완전 중복 행: {duplicate_rows:,}건")

    # 연봉·경력 등 관심 컬럼이 비어 있으면 분석 자체가 불가능하므로 행을 제거한다.
    df = df.dropna().drop_duplicates()

    # 문자열이 섞인 경력 컬럼을 숫자로 정규화하고, 변환 실패분을 제거한다.
    for column in FEATURE_NUMERIC_COLUMNS:
        df[column] = df[column].apply(_parse_years)
    df = df.dropna(subset=FEATURE_NUMERIC_COLUMNS)

    # 상위 1% 연봉은 극단적 이상치(수백만 달러)라 평균·회귀를 크게 왜곡한다.
    outlier_threshold = df[TARGET_COLUMN].quantile(0.99)
    df = df[df[TARGET_COLUMN] <= outlier_threshold].reset_index(drop=True)

    result = PreprocessResult(
        raw_rows=raw_rows,
        clean_rows=len(df),
        missing_by_column=missing_by_column,
        duplicate_rows=duplicate_rows,
        outlier_threshold=float(outlier_threshold),
    )

    reporter.log("\n✅ [처리 결과]")
    reporter.log(
        f"- 결측치·중복 제거 및 이상치(상위 1%, ${result.outlier_threshold:,.0f} 초과) 절단"
    )
    reporter.log(
        f"- {result.raw_rows:,}건 -> {result.clean_rows:,}건 ({result.retention_rate:.1%} 잔존)"
    )

    return df, result
