"""
파일명: stats.py
설명: 파이프라인 5단계 — 상관분석, t-검정, 효과크기(Cohen's d)·신뢰구간을 계산한다.
기능:
    - run_statistics(): 수치형 변수 간 상관계수 행렬을 구하고, AI 우호 그룹과
      그 외 그룹의 연봉 차이를 Welch's t-test로 검정한 뒤 StatsResult로 반환한다.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from .config import FAVORABLE_LABELS, NUMERIC_COLUMNS, TARGET_COLUMN
from .formatting import (
    cohens_d,
    conclude_ttest,
    describe_correlation,
    format_p_expression,
    interpret_cohens_d,
    interpret_p_value,
)
from .reporting import Reporter
from .results import StatsResult


def run_statistics(df: pd.DataFrame, reporter: Reporter) -> StatsResult:
    """수치형 변수 간 상관계수 행렬을 구하고 두 집단의 연봉 차이를 t-test로 검정한다."""
    reporter.section("5. 통계 분석 (상관계수 · t-test)")

    # 귀무가설 H0: 두 그룹의 평균 연봉은 같다.
    # 대립가설 H1: 두 그룹의 평균 연봉은 다르다. (양측 검정)
    is_favorable = df["AISent"].isin(FAVORABLE_LABELS)
    favorable = df.loc[is_favorable, TARGET_COLUMN]
    others = df.loc[~is_favorable, TARGET_COLUMN]

    # 두 그룹의 크기와 분산이 다르므로 Welch's t-test(equal_var=False)를 사용한다.
    t_statistic, p_value = stats.ttest_ind(favorable, others, equal_var=False)
    experience_corr, experience_corr_p = stats.pearsonr(
        df["YearsCodePro"], df[TARGET_COLUMN]
    )

    # [AI-assisted] 변경: 대표본(n≈1.7만)에선 p가 거의 자동 유의 → 효과크기·CI로 '실질적 크기' 판단.
    d = cohens_d(favorable, others)
    n1, n2 = len(favorable), len(others)
    diff = favorable.mean() - others.mean()
    se = (favorable.std() ** 2 / n1 + others.std() ** 2 / n2) ** 0.5
    ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se  # 대표본 정규근사 95% CI

    result = StatsResult(
        correlation_matrix=df[NUMERIC_COLUMNS].corr().round(3),
        experience_corr=float(experience_corr),
        experience_corr_p=float(experience_corr_p),
        favorable_size=len(favorable),
        favorable_mean=float(favorable.mean()),
        others_size=len(others),
        others_mean=float(others.mean()),
        t_statistic=float(t_statistic),
        p_value=float(p_value),
        cohens_d=float(d),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
    )

    reporter.log("🔗 [수치형 변수 간 상관계수 행렬 (Pearson)]")
    reporter.log(result.correlation_matrix.to_string())

    reporter.log("\n[상관분석] 전문 코딩 경력(YearsCodePro) ↔ 연봉")
    reporter.log(
        f"- 상관계수 r = {result.experience_corr:.3f}, "
        f"{format_p_expression(result.experience_corr_p)}"
    )
    reporter.log(f"-> {describe_correlation(result.experience_corr)}")

    reporter.log("\n[T-test] AI 우호 그룹 vs 그 외 그룹의 평균 연봉 차이")
    reporter.log(
        f"- 우호 그룹: {result.favorable_size:,}명, 평균 ${result.favorable_mean:,.0f}"
    )
    reporter.log(
        f"- 그 외 그룹: {result.others_size:,}명, 평균 ${result.others_mean:,.0f}"
    )
    reporter.log(
        f"- t-statistic = {result.t_statistic:.3f}, {format_p_expression(result.p_value)}"
    )
    reporter.log(f"-> [해석] {interpret_p_value(result.p_value)}")
    reporter.log(
        f"- 평균 차이 ${result.mean_gap:,.0f}, 95% CI [${result.ci_low:,.0f}, ${result.ci_high:,.0f}]"
    )
    reporter.log(
        f"- Cohen's d = {result.cohens_d:.3f} -> {interpret_cohens_d(result.cohens_d)}"
    )
    reporter.log(f"💡 {conclude_ttest(result)}")

    return result
