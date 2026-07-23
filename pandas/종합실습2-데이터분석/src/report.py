"""
파일명: report.py
설명: 파이프라인 8단계 — 각 단계의 결과 구조체를 받아 report.md 전문(마크다운)을 조립한다.
기능:
    - render_report(): 전체 보고서 문자열을 각 장(章) 렌더링 함수를 호출해 조립한다.
    - _overview_section 등 `_xxx_section()` 함수들: 장별(개요·데이터 준비·EDA·시각화·
      통계·모델·다변량·부록) 마크다운 본문을 생성한다.
    - write_report(): 조립된 보고서 문자열을 report.md 파일로 저장한다.
    - 분석 로직과 분리돼 있어, 여기 문구를 바꿔도 분석 코드는 건드릴 필요가 없다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import (
    ALPHA,
    BASE_DIR,
    BOXPLOT_NAME,
    CATEGORICAL_COLUMNS,
    DEEP_DIVE_CHART_NAME,
    FEATURE_NUMERIC_COLUMNS,
    INTERACTIVE_NAME,
    MODEL_NAME,
    REPORT_FILE,
    TARGET_COLUMN,
)
from .formatting import (
    conclude_ttest,
    describe_correlation,
    format_p_expression,
    format_p_value,
    frame_to_markdown,
    interpret_cohens_d,
    interpret_p_value,
    markdown_table,
)
from .results import (
    BenchmarkResult,
    EdaResult,
    LoadResult,
    ModelResult,
    MultivariateResult,
    PreprocessResult,
    StatsResult,
)


def render_report(
    data_path: Path,
    load: LoadResult,
    prep: PreprocessResult,
    eda: EdaResult,
    stats_result: StatsResult,
    model: ModelResult,
    multivariate: MultivariateResult,
    deep_dive_markdown: str,
    benchmark: BenchmarkResult,
) -> str:
    """각 단계의 결과를 받아 보고서 전문(마크다운)을 조립한다."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    return "\n".join(
        [
            "# Stack Overflow 2024 — AI 태도와 개발자 연봉 분석 보고서",
            "",
            f"> 이 문서는 `python3 -m src.main` 실행 시 자동 생성됩니다. (생성 시각: {generated_at})",
            "",
            _overview_section(data_path, prep, stats_result),
            _data_preparation_section(load, prep),
            _eda_section(eda),
            _visualization_section(),
            _statistics_section(stats_result),
            _model_section(model),
            _multivariate_section(multivariate),
            "## 7. 핵심 인사이트",
            "",
            deep_dive_markdown,
            _benchmark_section(benchmark),
        ]
    )


def _overview_section(
    data_path: Path, prep: PreprocessResult, stats_result: StatsResult
) -> str:
    """1장 — 분석 주제와 가설, 데이터 개요."""
    return "\n".join(
        [
            "## 1. 분석 개요",
            "",
            markdown_table(
                ["항목", "내용"],
                [
                    [
                        "분석 주제",
                        "AI 도구에 대한 태도가 개발자 연봉과 어떤 관계를 갖는가",
                    ],
                    ["데이터", "Stack Overflow Developer Survey 2024"],
                    ["원본 파일", f"`{data_path.name}`"],
                    ["원본 표본", f"{prep.raw_rows:,}건"],
                    ["분석 표본", f"{prep.clean_rows:,}건 (전처리 후)"],
                    ["종속 변수", "`ConvertedCompYearly` (연봉, USD)"],
                    ["핵심 독립 변수", "`AISent` (AI 도구에 대한 태도, 6단계)"],
                ],
            ),
            "",
            "**연구 가설**",
            "",
            "- 귀무가설 H₀: AI 도구에 우호적인 그룹과 그 외 그룹의 평균 연봉은 같다.",
            "- 대립가설 H₁: 두 그룹의 평균 연봉은 다르다. (양측 검정, α = 0.05)",
            "",
            f"→ **검정 결과: {'H₀ 기각' if stats_result.is_significant else 'H₀ 채택'}** "
            "(자세한 내용은 5장 참고)",
            "",
        ]
    )


def _data_preparation_section(load: LoadResult, prep: PreprocessResult) -> str:
    """2장 — 로딩 비교와 전처리 내역."""
    missing_rows = [
        [
            str(column),
            f"{count:,}",
            f"{count / prep.raw_rows:.1%}",
        ]
        for column, count in prep.missing_by_column.items()
    ]

    return "\n".join(
        [
            "## 2. 데이터 준비",
            "",
            "### 2.1 Pandas · Polars 로딩 비교",
            "",
            "동일한 CSV를 두 라이브러리로 각각 읽어 속도와 결과를 대조했습니다.",
            "",
            markdown_table(
                ["항목", "Pandas", "Polars"],
                [
                    [
                        "로딩 시간 (3회 평균)",
                        f"{load.pandas_seconds:.4f}초",
                        f"{load.polars_seconds:.4f}초",
                    ],
                    [
                        "형태 (행 × 열)",
                        f"{load.pandas_shape[0]:,} × {load.pandas_shape[1]}",
                        f"{load.polars_shape[0]:,} × {load.polars_shape[1]}",
                    ],
                    [
                        "관심 컬럼 결측치",
                        f"{load.pandas_nulls:,}",
                        f"{load.polars_nulls:,}",
                    ],
                ],
            ),
            "",
            f"→ **{load.faster_engine}가 약 {load.speed_ratio:.1f}배 빠릅니다.** "
            f"형태 일치: {'✅' if load.same_shape else '❌'} · "
            f"결측치 집계 일치: {'✅' if load.same_nulls else '❌'}",
            "",
            "> 이 설문 CSV는 결측치를 문자열 `NA`로 표기합니다. Pandas는 이를 자동으로 결측 처리하지만",
            "> Polars는 `null_values`를 지정해야 하며, 지정하지 않으면 숫자 컬럼이 문자열로 읽히고",
            "> 결측치가 0건으로 잘못 집계됩니다.",
            "",
            "### 2.2 결측치 · 중복 · 이상치 처리",
            "",
            markdown_table(["컬럼", "결측치", "비율"], missing_rows),
            "",
            "적용한 처리:",
            "",
            "1. **결측치 제거** — 관심 컬럼에 결측이 있는 행 삭제",
            f"2. **중복 제거** — 완전 중복 {prep.duplicate_rows:,}건 삭제",
            "3. **표기 정규화** — `Less than 1 year` → 0.5, `More than 50 years` → 51.0",
            f"4. **이상치 절단** — 연봉 상위 1%(${prep.outlier_threshold:,.0f} 초과) 제외",
            "",
            f"→ **{prep.raw_rows:,}건 → {prep.clean_rows:,}건** "
            f"({prep.retention_rate:.1%} 잔존)",
            "",
        ]
    )


def _eda_section(eda: EdaResult) -> str:
    """3장 — 기술통계와 분포."""
    sentiment_rows = [
        [str(label), f"{count:,}", f"{count / eda.total_rows:.1%}"]
        for label, count in eda.sentiment_counts.items()
    ]
    country_rows = [
        [str(country), f"{count:,}", f"{count / eda.total_rows:.1%}"]
        for country, count in eda.top_countries.items()
    ]

    return "\n".join(
        [
            "## 3. 탐색적 데이터 분석 (EDA)",
            "",
            "### 3.1 기술통계",
            "",
            frame_to_markdown(eda.describe, "통계량"),
            "",
            "### 3.2 AI 태도별 응답자 분포",
            "",
            markdown_table(["AI 태도", "응답자", "비율"], sentiment_rows),
            "",
            "### 3.3 상위 5개 국가",
            "",
            f"전체 {eda.country_count}개국 중 상위 5개국입니다.",
            "",
            markdown_table(["국가", "응답자", "비율"], country_rows),
            "",
        ]
    )


def _visualization_section() -> str:
    """4장 — 생성된 차트를 보고서에 임베드한다."""
    return "\n".join(
        [
            "## 4. 시각화",
            "",
            "### 4.1 AI 도구 인식별 연봉 분포 (Seaborn · 정적)",
            "",
            f"![AI 태도별 연봉 분포](./outputs/charts/{BOXPLOT_NAME})",
            "",
            "### 4.2 주요 국가별 AI 인식에 따른 평균 연봉 (Plotly · 인터랙티브)",
            "",
            f"[▶ 인터랙티브 차트 열기](./outputs/charts/{INTERACTIVE_NAME})",
            "",
            "### 4.3 경력 구간별 AI 인식과 연봉 추이 (Seaborn · 정적)",
            "",
            f"![경력별 AI 인식 추이](./outputs/charts/{DEEP_DIVE_CHART_NAME})",
            "",
        ]
    )


def _statistics_section(result: StatsResult) -> str:
    """5장 — 상관분석과 t-검정 (효과크기·신뢰구간 포함)."""
    return "\n".join(
        [
            "## 5. 통계 분석",
            "",
            "### 5.1 수치형 변수 간 상관계수 (Pearson)",
            "",
            frame_to_markdown(result.correlation_matrix, "변수", number_format=".3f"),
            "",
            f"→ 전문 코딩 경력과 연봉의 상관계수는 **r = {result.experience_corr:.3f}** "
            f"({format_p_expression(result.experience_corr_p)})로, "
            f"{describe_correlation(result.experience_corr)}",
            "",
            "### 5.2 독립표본 t-검정 (Welch)",
            "",
            "두 그룹의 표본 크기와 분산이 다르므로 등분산을 가정하지 않는 Welch's t-test를 사용했습니다.",
            "",
            markdown_table(
                ["그룹", "표본 수", "평균 연봉 (USD)"],
                [
                    [
                        "AI 우호 (Favorable · Very favorable)",
                        f"{result.favorable_size:,}",
                        f"{result.favorable_mean:,.0f}",
                    ],
                    [
                        "그 외",
                        f"{result.others_size:,}",
                        f"{result.others_mean:,.0f}",
                    ],
                ],
            ),
            "",
            markdown_table(
                ["검정 통계량", "값"],
                [
                    ["t-statistic", f"{result.t_statistic:.3f}"],
                    ["p-value", format_p_value(result.p_value)],
                    ["유의수준 α", f"{ALPHA}"],
                    ["평균 차이 (우호 − 그 외)", f"${result.mean_gap:,.0f}"],
                    [
                        "95% 신뢰구간",
                        f"[${result.ci_low:,.0f}, ${result.ci_high:,.0f}]",
                    ],
                    [
                        "Cohen's d (효과크기)",
                        f"{result.cohens_d:.3f} ({interpret_cohens_d(result.cohens_d)})",
                    ],
                ],
            ),
            "",
            f"**p-value 해석**: {interpret_p_value(result.p_value)}",
            "",
            "> 표본이 약 1.7만 건으로 크면 아주 작은 차이도 p-value가 유의하게 나오므로, "
            "**효과크기(Cohen's d)** 로 '차이가 실질적으로도 큰지'를 함께 판단했습니다.",
            "",
            f"**{conclude_ttest(result)}**",
            "",
        ]
    )


def _model_section(model: ModelResult) -> str:
    """6장 — ML Pipeline 구성과 평가."""
    return "\n".join(
        [
            "## 6. 머신러닝 파이프라인",
            "",
            "### 6.1 파이프라인 구성",
            "",
            "전처리와 모델을 하나의 `sklearn.pipeline.Pipeline`으로 묶어, 학습 데이터의 통계가",
            "테스트 데이터로 새지 않도록 했습니다.",
            "",
            markdown_table(
                ["단계", "구성 요소", "대상 컬럼"],
                [
                    [
                        "수치형 전처리",
                        "`StandardScaler`",
                        ", ".join(f"`{column}`" for column in FEATURE_NUMERIC_COLUMNS),
                    ],
                    [
                        "범주형 전처리",
                        '`OneHotEncoder(handle_unknown="ignore")`',
                        ", ".join(f"`{column}`" for column in CATEGORICAL_COLUMNS),
                    ],
                    ["모델", "`Ridge(alpha=1.0)`", f"타깃: `{TARGET_COLUMN}`"],
                ],
            ),
            "",
            f"학습/테스트 분할: **{model.train_rows:,}건 / {model.test_rows:,}건** (8:2, `random_state=42`)",
            "",
            "### 6.2 평가 지표",
            "",
            markdown_table(
                ["지표", "값", "의미"],
                [
                    ["R² Score", f"{model.r2:.4f}", "모델이 설명하는 연봉 분산의 비율"],
                    ["RMSE", f"${model.rmse:,.0f}", "예측 오차의 제곱평균제곱근"],
                    ["MAE", f"${model.mae:,.0f}", "예측 오차의 절댓값 평균"],
                ],
            ),
            "",
            f"저장된 모델: `outputs/models/{MODEL_NAME}`",
            "",
        ]
    )


def _multivariate_section(result: MultivariateResult) -> str:
    """6.3장 — 교란변수를 동시 통제한 다변량 회귀 (day2 이식)."""
    top_rows = [
        [name.replace("num__", "").replace("cat__", ""), f"{val:+,.0f}"]
        for name, val in result.top_coefficients
    ]
    ai_rows = [
        [name.replace("cat__AISent_", ""), f"{val:+,.0f}"]
        for name, val in result.ai_coefficients
    ]
    paradox_note = (
        f"→ 학력·조직·근무형태·연령·경력·국가·직군을 **모두 통제한 뒤에도** "
        f"Very unfavorable 계수({result.vu_coef:+,.0f})가 "
        f"Very favorable({result.vf_coef:+,.0f})보다 높습니다. "
        "**역설은 교란변수로 설명되지 않고 잔존합니다.**"
        if result.paradox_persists
        else f"→ 다변량 통제 후에는 Very favorable 계수({result.vf_coef:+,.0f})가 "
        f"Very unfavorable({result.vu_coef:+,.0f})보다 높거나 비슷합니다. "
        "앞서 보인 역설의 상당 부분은 경력·직군 등 **교란변수로 설명됩니다.**"
    )

    return "\n".join(
        [
            "### 6.3 연봉 결정요인 다변량 분석",
            "",
            "6.1의 기본 모델에 학력(EdLevel)·조직규모(OrgSize)·근무형태(RemoteWork)·연령(Age)을",
            "추가해, 여러 요인을 **한 모델에서 동시에 통제**했을 때 각 요인의 순효과를 봅니다.",
            "",
            markdown_table(
                ["모델", "변수 수", "R²"],
                [
                    [
                        "기본",
                        f"{result.n_baseline_features}",
                        f"{result.baseline_r2:.4f}",
                    ],
                    [
                        "확장",
                        f"{result.n_enriched_features}",
                        f"{result.enriched_r2:.4f}",
                    ],
                ],
            ),
            "",
            f"→ 변수 추가로 설명력 **{result.gain:+.4f}** 변화 "
            f"(연봉 분산의 {result.enriched_r2 * 100:.1f}%를 설명).",
            "",
            "#### 연봉을 가장 크게 좌우하는 요인 Top 10 (계수, USD 근사)",
            "",
            markdown_table(["요인", "계수"], top_rows),
            "",
            '#### AISent 순효과 — "다 통제해도 역설이 남는가"',
            "",
            markdown_table(["AI 태도", "계수"], ai_rows),
            "",
            paradox_note,
            "",
            "> ⚠️ 해석 주의: 수치형(경력)만 표준화했고 범주형은 원-핫이라, 계수는 "
            "'해당 범주일 때 연봉 기여도(USD 근사)'로 읽되 절대값보다 **부호와 상대 크기**에 무게를 둘 것.",
            "",
        ]
    )


def _benchmark_section(result: BenchmarkResult) -> str:
    """부록 — Pandas vs Polars 작업별 성능 벤치마크 (day2 이식)."""
    rows = [
        [
            task,
            f"{pd_sec:.4f}",
            f"{pl_sec:.4f}",
            f"{pd_sec / pl_sec:.2f}x" if pl_sec else "—",
        ]
        for task, pd_sec, pl_sec in result.rows
    ]
    return "\n".join(
        [
            "## 8. 부록 — Pandas vs Polars 성능 벤치마크",
            "",
            f"동일 작업을 두 라이브러리로 각 {result.runs}회 실행한 평균 시간입니다 "
            "(로딩은 I/O 포함, 필터·집계는 메모리 프레임 대상).",
            "",
            markdown_table(["작업", "Pandas(초)", "Polars(초)", "Polars 배속"], rows),
            "",
            f"![Pandas vs Polars 벤치마크](./outputs/charts/{result.chart_name})",
            "",
        ]
    )


def write_report(report_markdown: str) -> None:
    """조립된 보고서를 report.md로 저장한다."""
    REPORT_FILE.write_text(report_markdown, encoding="utf-8")
    print(f"✅ 보고서 생성 완료: {REPORT_FILE.relative_to(BASE_DIR)}")
