"""
파일명: formatting.py
설명: 공용 포매팅 헬퍼 모듈 — 통계 해석 문구 생성과 마크다운 표 조립을 담당한다.
기능:
    - 통계 단계(콘솔 출력)와 보고서 단계(report.py) 양쪽에서 공유하는 함수들을 모았다.
    - p-value·Cohen's d·상관계수 해석 문구 생성 함수 제공
    - tabulate 등 외부 의존성 없이 마크다운 표를 직접 조립하는 함수 제공
"""

from __future__ import annotations

import pandas as pd

from .config import ALPHA
from .results import StatsResult


# ──────────────────────────────────────────────────────────────────────────────
# 통계 해석 문구
# ──────────────────────────────────────────────────────────────────────────────
def format_p_value(p_value: float) -> str:
    """p-value를 표기한다. 부동소수점 하한 아래로 내려가면 0 대신 부등호로 표시한다."""
    return "< 1e-308" if p_value == 0 else f"{p_value:.4g}"


def format_p_expression(p_value: float) -> str:
    """`p = 0.001` / `p < 1e-308` 형태의 문장 삽입용 표현을 만든다."""
    formatted = format_p_value(p_value)
    # 하한 미만이면 이미 "< 1e-308"이므로 등호를 덧붙이지 않는다.
    return f"p {formatted}" if formatted.startswith("<") else f"p = {formatted}"


def interpret_p_value(p_value: float, alpha: float = ALPHA) -> str:
    """p-value를 유의수준과 비교해 한국어 해석 문장으로 변환한다."""
    verdict = (
        "기각 — 통계적으로 유의미함"
        if p_value < alpha
        else "채택 — 통계적으로 유의미하지 않음"
    )
    return f"귀무가설 {verdict} ({format_p_expression(p_value)}, α = {alpha})"


# [AI-assisted] 초안 생성 후 수정 — 변경: 대표본에서 p-value가 자동 유의해지는 문제 보완용 효과크기 추가
def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """두 그룹 평균차를 합동표준편차로 나눈 표준화 효과크기 (표본 크기에 둔감)."""
    n1, n2 = len(a), len(b)
    s1, s2 = a.std(), b.std()
    pooled_sd = (((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)) ** 0.5
    return (a.mean() - b.mean()) / pooled_sd


def interpret_cohens_d(d: float) -> str:
    """Cohen 관례: |d| 0.2 소 / 0.5 중 / 0.8 대."""
    ad = abs(d)
    if ad < 0.2:
        return "무시할 수준 (0.2 미만)"
    if ad < 0.5:
        return "작은 효과"
    if ad < 0.8:
        return "중간 효과"
    return "큰 효과"


def describe_correlation(corr: float) -> str:
    """상관계수의 강도와 방향을 한 문장으로 설명한다."""
    strength = "뚜렷한" if abs(corr) > 0.3 else "약한"
    direction = "양" if corr > 0 else "음"
    return f"{strength} {direction}의 상관관계입니다."


def conclude_ttest(result: StatsResult) -> str:
    """t-검정 결과를 주제에 맞는 결론 문장으로 바꾼다. 효과크기까지 함께 언급한다."""
    if not result.is_significant:
        return "결론: 두 그룹의 평균 연봉 차이는 우연으로 설명 가능한 수준입니다."
    winner = (
        "AI에 우호적인" if result.t_statistic > 0 else "AI에 비우호적이거나 무관심한"
    )
    return (
        f"결론: {winner} 개발자 그룹의 평균 연봉이 통계적으로 유의미하게 높습니다. "
        f"다만 Cohen's d={result.cohens_d:.3f}({interpret_cohens_d(result.cohens_d)})로, "
        "'유의미 ≠ 실질적으로 큼'에 주의해야 합니다."
    )


# ──────────────────────────────────────────────────────────────────────────────
# 마크다운 렌더링 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """헤더와 행 목록을 GitHub 형식 마크다운 표로 만든다."""
    divider = ["---"] * len(headers)
    body = [headers, divider, *rows]
    return "\n".join("| " + " | ".join(cell for cell in row) + " |" for row in body)


def frame_to_markdown(
    frame: pd.DataFrame, index_header: str, number_format: str = ",.2f"
) -> str:
    """DataFrame을 인덱스까지 포함한 마크다운 표로 변환한다."""
    headers = [index_header, *(str(column) for column in frame.columns)]
    rows = [
        [str(index), *(_format_cell(value, number_format) for value in row)]
        for index, row in zip(frame.index, frame.to_numpy())
    ]
    return markdown_table(headers, rows)


def series_to_markdown(
    series: pd.Series, index_header: str, value_header: str, number_format: str = ",.0f"
) -> str:
    """Series를 2열짜리 마크다운 표로 변환한다."""
    rows = [
        [str(index), _format_cell(value, number_format)]
        for index, value in series.items()
    ]
    return markdown_table([index_header, value_header], rows)


def _format_cell(value: object, number_format: str) -> str:
    """표 셀 하나를 문자열로 만든다. 숫자만 지정된 형식을 적용한다."""
    if isinstance(value, (int, float)) and pd.notna(value):
        return format(value, number_format)
    return str(value)
