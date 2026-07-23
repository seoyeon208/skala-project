"""
파일명: config.py
설명: 파이프라인 전역 상수 모듈 — 경로, 산출물 이름, 분석 대상 컬럼, 유의수준을 정의한다.
기능:
    - 산출물 디렉터리(BASE_DIR, CHART_DIR, MODEL_DIR 등) 경로 상수 제공
    - 분석에 사용할 컬럼 목록(수치형·범주형) 정의
    - AI 태도 순서, 유의수준(ALPHA) 등 분석 전역에서 재사용하는 값 정의
    - 다른 src 모듈을 import하지 않아 순환 참조가 발생하지 않는 최하위 모듈
"""

from __future__ import annotations

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 경로 및 상수
#   이 파일은 src/ 안에 있으므로, 산출물 기준 디렉터리(end2end/)는 parent.parent다.
#   (원본 day2_merged.py는 end2end/ 바로 아래에 있어 parent 한 번이면 됐다.)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "results.csv"
REPORT_FILE = BASE_DIR / "report.md"
CHART_DIR = BASE_DIR / "outputs" / "charts"
MODEL_DIR = BASE_DIR / "outputs" / "models"

# 보고서에서 참조할 산출물 파일명 (report.md 기준 상대경로 조합에 사용)
BOXPLOT_NAME = "ai_sentiment_income_boxplot.png"
INTERACTIVE_NAME = "country_ai_sentiment_income.html"
DEEP_DIVE_CHART_NAME = "experience_ai_sentiment_income.png"
BENCHMARK_CHART_NAME = "pandas_polars_benchmark.png"
MODEL_NAME = "ai_income_pipeline.joblib"

# 분석에 사용할 컬럼. 원본은 114개 컬럼이라 주제와 관련된 것만 추린다.
# WorkExp(총 경력)는 결측률 54.7%로, 넣으면 표본이 29% 줄어드는 데 비해 YearsCodePro와
# 정보가 겹쳐 제외했다. (실측: 공통 5컬럼 17,315건 → WorkExp 추가 시 12,318건)
TARGET_COLUMN = "ConvertedCompYearly"
NUMERIC_COLUMNS = [TARGET_COLUMN, "YearsCodePro", "YearsCode"]
FEATURE_NUMERIC_COLUMNS = NUMERIC_COLUMNS[1:]  # 타깃을 제외한 수치형 피처 목록
CATEGORICAL_COLUMNS = ["AISent", "Country", "DevType"]

# [AI-assisted] 변경: day2 다변량 분석용 범주형 컬럼(학력·조직규모·근무형태·연령) 추가.
#                기본 ML은 CATEGORICAL_COLUMNS만, 다변량 확장 모델은 여기까지 함께 쓴다.
EXTRA_CATEGORICAL_COLUMNS = ["EdLevel", "OrgSize", "RemoteWork", "Age"]
COLUMNS_OF_INTEREST = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + EXTRA_CATEGORICAL_COLUMNS

# AI 태도는 순서가 있는 범주형이므로 차트에서 이 순서로 고정한다.
AI_SENTIMENT_ORDER = [
    "Very unfavorable",
    "Unfavorable",
    "Indifferent",
    "Unsure",
    "Favorable",
    "Very favorable",
]
FAVORABLE_LABELS = ["Favorable", "Very favorable"]

ALPHA = 0.05  # 통계 검정 유의수준

# 이 설문 CSV는 결측치를 문자열 "NA"로 표기한다. Pandas는 기본으로 NaN 처리하지만
# Polars는 문자열로 읽어 숫자 컬럼까지 String이 되므로, 명시적으로 지정해 맞춘다.
POLARS_NULL_VALUES = ["NA"]
