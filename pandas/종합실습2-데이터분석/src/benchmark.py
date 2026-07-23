"""
파일명: benchmark.py
설명: 파이프라인 0단계(부록) — Pandas와 Polars의 순수 처리 속도를 비교한다.
기능:
    - run_benchmark(): 로딩·필터·집계 3개 작업을 각각 여러 번 실행해 두 라이브러리의
      평균 처리 시간을 재고 BenchmarkResult로 반환한다.
    - _save_benchmark_chart(): 작업별 처리 시간을 비교하는 막대차트를 저장한다.

1단계 load_and_compare()가 "결과 정합성"을 보는 것과 목적이 달라(순수 속도) 별도 단계로 둔다.
"""

from __future__ import annotations

import timeit
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns

from .config import (
    BASE_DIR,
    BENCHMARK_CHART_NAME,
    CHART_DIR,
    POLARS_NULL_VALUES,
    TARGET_COLUMN,
)
from .reporting import Reporter
from .results import BenchmarkResult


def run_benchmark(
    data_path: Path, reporter: Reporter, runs: int = 3
) -> BenchmarkResult:
    """동일 작업(로딩·필터·집계)을 두 라이브러리로 각 runs회 실행해 평균 시간을 잰다."""
    reporter.section("0. Pandas vs Polars 성능 벤치마크")

    # 로딩: 파일 → DataFrame (I/O 포함)
    pd_load = (
        timeit.timeit(lambda: pd.read_csv(data_path, low_memory=False), number=runs)
        / runs
    )
    pl_load = (
        timeit.timeit(
            lambda: pl.read_csv(
                str(data_path), null_values=POLARS_NULL_VALUES, ignore_errors=True
            ),
            number=runs,
        )
        / runs
    )

    # 필터·집계는 이미 메모리에 올라온 프레임 대상 (I/O 제외, 순수 연산만 비교)
    df_pd = pd.read_csv(data_path, low_memory=False)
    df_pl = pl.read_csv(
        str(data_path), null_values=POLARS_NULL_VALUES, ignore_errors=True
    )

    pd_filter = (
        timeit.timeit(lambda: df_pd[df_pd[TARGET_COLUMN].notna()], number=runs) / runs
    )
    pl_filter = (
        timeit.timeit(
            lambda: df_pl.filter(pl.col(TARGET_COLUMN).is_not_null()), number=runs
        )
        / runs
    )

    pd_group = (
        timeit.timeit(
            lambda: df_pd.groupby("Country")[TARGET_COLUMN].mean(), number=runs
        )
        / runs
    )
    pl_group = (
        timeit.timeit(
            lambda: df_pl.group_by("Country").agg(pl.col(TARGET_COLUMN).mean()),
            number=runs,
        )
        / runs
    )

    result = BenchmarkResult(
        runs=runs,
        load=(pd_load, pl_load),
        filter_op=(pd_filter, pl_filter),
        group_op=(pd_group, pl_group),
        chart_name=BENCHMARK_CHART_NAME,
    )

    reporter.log(f"📊 [작업별 처리 시간 ({runs}회 평균, 초 — 낮을수록 빠름)]")
    for task, pd_sec, pl_sec in result.rows:
        ratio = pd_sec / pl_sec if pl_sec else float("inf")
        reporter.log(
            f"- {task}: Pandas {pd_sec:.4f} / Polars {pl_sec:.4f} (Polars {ratio:.2f}배)"
        )

    _save_benchmark_chart(result, reporter)
    return result


def _save_benchmark_chart(result: BenchmarkResult, reporter: Reporter) -> None:
    """작업별 pandas vs polars 처리 시간을 나란히 놓은 막대차트를 저장한다."""
    plot_df = pd.DataFrame(
        [
            {"작업": task, "라이브러리": lib, "시간(초)": sec}
            for task, pd_sec, pl_sec in result.rows
            for lib, sec in (("Pandas", pd_sec), ("Polars", pl_sec))
        ]
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_df, x="작업", y="시간(초)", hue="라이브러리", palette="Set2")
    plt.title("Pandas vs Polars 작업별 처리 시간 (낮을수록 빠름)", fontsize=13)
    plt.ylabel("평균 소요 시간 (초)")
    plt.xlabel("작업")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    chart_path = CHART_DIR / result.chart_name
    plt.savefig(chart_path, dpi=150)
    plt.close()
    reporter.log(f"✅ 벤치마크 차트 저장: {chart_path.relative_to(BASE_DIR)}")
