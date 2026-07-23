"""
파일명: eda.py
설명: 파이프라인 3단계 — 탐색적 데이터 분석(EDA)으로 기술통계와 분포를 산출한다.
기능:
    - run_eda(): 수치형 변수의 기술통계(평균·표준편차·분위수)와 AI 태도별·국가별
      분포를 계산하고 콘솔에 출력한 뒤 EdaResult로 반환한다.
"""

from __future__ import annotations

import pandas as pd

from .config import AI_SENTIMENT_ORDER, NUMERIC_COLUMNS
from .reporting import Reporter
from .results import EdaResult


def run_eda(df: pd.DataFrame, reporter: Reporter) -> EdaResult:
    """수치형 기술통계(평균·표준편차·분위수)와 범주형 분포를 산출한다."""
    reporter.section("3. EDA (기술통계)")

    result = EdaResult(
        describe=df[NUMERIC_COLUMNS].describe().round(2),
        sentiment_counts=df["AISent"].value_counts().reindex(AI_SENTIMENT_ORDER),
        top_countries=df["Country"].value_counts().head(5),
        country_count=int(df["Country"].nunique()),
        total_rows=len(df),
    )

    reporter.log("📉 [수치형 기술통계]")
    reporter.log(result.describe.to_string())

    reporter.log("\n📊 [AI 태도별 응답자 분포]")
    for label, count in result.sentiment_counts.items():
        reporter.log(f"- {label}: {count:,}명 ({count / result.total_rows:.1%})")

    reporter.log(f"\n📊 [상위 5개 국가] (전체 {result.country_count}개국)")
    for country, count in result.top_countries.items():
        reporter.log(f"- {country}: {count:,}명")

    return result
