"""
파일명: results.py
설명: 파이프라인 각 단계의 결과를 담는 dataclass 모음.
기능:
    - BenchmarkResult, LoadResult, PreprocessResult, EdaResult, StatsResult,
      ModelResult, MultivariateResult 등 단계별 결과 구조체를 정의한다.
    - 보고서 렌더링(report.py)이 콘솔 로그 문자열이 아니라 이 값들을 읽도록 해서,
      출력 형식과 분석 로직을 분리한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ALPHA


@dataclass
class BenchmarkResult:
    """0단계 — Pandas / Polars 작업별 처리 시간 비교 (로딩·필터·집계)."""

    runs: int
    load: tuple[float, float]  # (pandas 초, polars 초)
    filter_op: tuple[float, float]
    group_op: tuple[float, float]
    chart_name: str

    @property
    def rows(self) -> list[tuple[str, float, float]]:
        """(작업명, pandas 초, polars 초) 목록. 표·차트 양쪽에서 재사용한다."""
        return [
            ("로딩(read_csv)", *self.load),
            ("필터(notna)", *self.filter_op),
            ("집계(groupby mean)", *self.group_op),
        ]


@dataclass
class LoadResult:
    """1단계 — Pandas / Polars 로딩 결과 정합성 비교."""

    pandas_seconds: float
    polars_seconds: float
    pandas_shape: tuple[int, int]
    polars_shape: tuple[int, int]
    pandas_nulls: int
    polars_nulls: int

    @property
    def faster_engine(self) -> str:
        return "Polars" if self.polars_seconds < self.pandas_seconds else "Pandas"

    @property
    def speed_ratio(self) -> float:
        slower, faster = sorted(
            (self.pandas_seconds, self.polars_seconds), reverse=True
        )
        return slower / faster

    @property
    def same_shape(self) -> bool:
        return self.pandas_shape == self.polars_shape

    @property
    def same_nulls(self) -> bool:
        return self.pandas_nulls == self.polars_nulls


@dataclass
class PreprocessResult:
    """2단계 — 결측치·중복·이상치 처리 내역."""

    raw_rows: int
    clean_rows: int
    missing_by_column: pd.Series
    duplicate_rows: int
    outlier_threshold: float

    @property
    def retention_rate(self) -> float:
        return self.clean_rows / self.raw_rows


@dataclass
class EdaResult:
    """3단계 — 기술통계 및 범주형 분포."""

    describe: pd.DataFrame
    sentiment_counts: pd.Series
    top_countries: pd.Series
    country_count: int
    total_rows: int


@dataclass
class StatsResult:
    """5단계 — 상관분석 및 t-검정 결과 (효과크기·신뢰구간 포함)."""

    correlation_matrix: pd.DataFrame
    experience_corr: float
    experience_corr_p: float
    favorable_size: int
    favorable_mean: float
    others_size: int
    others_mean: float
    t_statistic: float
    p_value: float
    # [AI-assisted] 변경: 대표본에서 p가 자동 유의해지는 문제를 보완할 효과크기·신뢰구간 필드 추가
    cohens_d: float
    ci_low: float
    ci_high: float

    @property
    def is_significant(self) -> bool:
        return self.p_value < ALPHA

    @property
    def mean_gap(self) -> float:
        """우호 그룹 평균에서 그 외 그룹 평균을 뺀 값 (음수면 비우호 그룹이 높다)."""
        return self.favorable_mean - self.others_mean


@dataclass
class ModelResult:
    """6단계 — ML Pipeline 학습·평가 결과."""

    train_rows: int
    test_rows: int
    r2: float
    rmse: float
    mae: float


@dataclass
class MultivariateResult:
    """6단계(확장) — 교란변수를 한 모델에서 동시 통제한 다변량 회귀 결과."""

    baseline_r2: float
    enriched_r2: float
    n_baseline_features: int
    n_enriched_features: int
    top_coefficients: list[tuple[str, float]]  # (피처명, 계수) Top 10
    ai_coefficients: list[tuple[str, float]]  # AISent 범주별 계수 (내림차순)
    vf_coef: float  # Very favorable 계수
    vu_coef: float  # Very unfavorable 계수

    @property
    def gain(self) -> float:
        return self.enriched_r2 - self.baseline_r2

    @property
    def paradox_persists(self) -> bool:
        """모든 변수를 통제한 뒤에도 회의론자(vu) 계수가 우호론자(vf)보다 높은가."""
        return self.vu_coef > self.vf_coef
