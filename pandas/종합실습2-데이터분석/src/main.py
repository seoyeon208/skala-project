"""
파일명: main.py
설명: 파이프라인 전체를 순서대로 실행하는 진입점 (day2_merged.py 기능별 분리본).
기능:
    - parse_args(): CLI 인자(--data-path)를 파싱한다.
    - main(): 0~8단계(벤치마크 → 로딩 → 전처리 → EDA → 시각화 → 통계 → ML →
      심층분석 → 보고서 생성)를 순서대로 호출하고 report.md를 생성한다.

분석 주제:
    "AI 도구에 대한 태도(AISent)가 개발자의 연봉과 어떤 관계를 갖는가?"

데이터:
    Stack Overflow Developer Survey 2024
    https://github.com/StackExchange/Survey/tree/main/packages/archive/2024

실행:
    python3 -m src.main
    python3 -m src.main --data-path results.csv

구조:
    각 분석 단계는 콘솔에 진행 상황을 출력하는 동시에, 결과를 구조화된 dataclass로
    반환한다. 마지막의 render_report()가 그 결과들을 받아 report.md를 조립하므로,
    보고서 문구를 바꿔도 분석 로직은 건드릴 필요가 없다.

파이프라인 단계 (각 단계는 src/ 아래 동명 모듈):
    0. 벤치마크  — Pandas vs Polars를 로딩·필터·집계 3개 작업으로 비교 (부록)
    1. 로딩      — Pandas / Polars 양쪽으로 읽어 결과 정합성(shape·결측)을 비교
    2. 전처리    — 결측치·중복·이상치 처리
    3. EDA       — 기술통계(평균·표준편차·분위수) 및 분포 확인
    4. 시각화    — Seaborn 정적 차트 + Plotly 인터랙티브 차트
    5. 통계분석  — 상관계수 행렬 + t-test + 효과크기(Cohen's d)·신뢰구간
    6. ML        — sklearn Pipeline(전처리 + 모델) 학습·평가 + 다변량 결정요인 분석
    7. 심층분석  — 국가/경력/직군을 통제한 교차 검증
    8. 자동화    — 위 결과를 report.md로 자동 생성
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from .benchmark import run_benchmark
from .config import CHART_DIR, DEFAULT_DATA_PATH, MODEL_DIR
from .deep_dive import run_deep_dive
from .eda import run_eda
from .loading import load_and_compare
from .model import run_multivariate, train_model
from .preprocess import preprocess
from .report import render_report, write_report
from .reporting import Reporter
from .stats import run_statistics
from .viz import configure_korean_font, create_charts

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    """`--data-path` 등 CLI 인자를 파싱해 반환한다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="분석할 설문 CSV 경로 (기본값: results.csv)",
    )
    return parser.parse_args()


def main() -> None:
    """파이프라인 전체를 순서대로 실행하고 보고서를 생성한다."""
    args = parse_args()
    if not args.data_path.exists():
        raise SystemExit(f"데이터 파일을 찾을 수 없습니다: {args.data_path}")

    configure_korean_font()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    reporter = Reporter()
    benchmark_result = run_benchmark(args.data_path, reporter)
    raw_df, load_result = load_and_compare(args.data_path, reporter)
    df, preprocess_result = preprocess(raw_df, reporter)

    eda_result = run_eda(df, reporter)
    create_charts(df, reporter)
    stats_result = run_statistics(df, reporter)
    _, model_result = train_model(df, reporter)
    multivariate_result = run_multivariate(df, model_result.r2, reporter)
    deep_dive_markdown = run_deep_dive(df, reporter)

    reporter.section("8. 보고서 자동 생성")
    write_report(
        render_report(
            data_path=args.data_path,
            load=load_result,
            prep=preprocess_result,
            eda=eda_result,
            stats_result=stats_result,
            model=model_result,
            multivariate=multivariate_result,
            deep_dive_markdown=deep_dive_markdown,
            benchmark=benchmark_result,
        )
    )


if __name__ == "__main__":
    main()
