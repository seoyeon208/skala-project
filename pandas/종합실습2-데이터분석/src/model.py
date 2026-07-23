"""
파일명: model.py
설명: 파이프라인 6단계 — 회귀 모델 학습·평가와 다변량 연봉 결정요인 분석을 담당한다.
기능:
    - train_model(): 전처리(표준화·원-핫 인코딩)와 Ridge 회귀를 하나의 sklearn
      Pipeline으로 묶어 학습·평가하고 모델 파일을 저장한다.
    - run_multivariate(): 학력·조직규모·근무형태·연령까지 통제한 확장 모델로
      연봉에 영향을 미치는 요인의 순효과를 분석한다.
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    BASE_DIR,
    CATEGORICAL_COLUMNS,
    EXTRA_CATEGORICAL_COLUMNS,
    FEATURE_NUMERIC_COLUMNS,
    MODEL_DIR,
    MODEL_NAME,
    TARGET_COLUMN,
)
from .reporting import Reporter
from .results import ModelResult, MultivariateResult


def train_model(df: pd.DataFrame, reporter: Reporter) -> tuple[Pipeline, ModelResult]:
    """전처리와 회귀 모델을 하나의 sklearn Pipeline으로 묶어 학습·평가·저장한다."""
    reporter.section("6. ML Pipeline (Ridge Regression)")

    features = df[FEATURE_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    target = df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # 수치형은 표준화, 범주형은 원-핫 인코딩. 학습 데이터에 없던 범주는 무시한다.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    # 전처리를 Pipeline 안에 넣어야 학습 데이터의 통계가 테스트로 새지 않는다.
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", Ridge(alpha=1.0))]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    result = ModelResult(
        train_rows=len(x_train),
        test_rows=len(x_test),
        r2=float(r2_score(y_test, predictions)),
        rmse=float(mean_squared_error(y_test, predictions) ** 0.5),
        mae=float(mean_absolute_error(y_test, predictions)),
    )

    reporter.log(f"- 학습 {result.train_rows:,}건 / 테스트 {result.test_rows:,}건")
    reporter.log("\n[모델 평가 지표]")
    reporter.log(f"- R² Score (설명력): {result.r2:.4f}")
    reporter.log(f"- RMSE (평균 제곱근 오차): ${result.rmse:,.0f}")
    reporter.log(f"- MAE  (평균 절대 오차): ${result.mae:,.0f}")

    model_path = MODEL_DIR / MODEL_NAME
    joblib.dump(pipeline, model_path)
    reporter.log(f"✅ 모델 저장(joblib): {model_path.relative_to(BASE_DIR)}")

    return pipeline, result


# [AI-assisted] 초안 생성 후 수정 — 변경: day2 딥다이브의 '변수 하나씩 통제'를 다변량 회귀로 통합.
def run_multivariate(
    df: pd.DataFrame, baseline_r2: float, reporter: Reporter
) -> MultivariateResult:
    """학력·조직규모·근무형태·연령까지 한 모델에서 동시 통제한 Ridge 회귀로 연봉 결정요인을 본다."""
    reporter.section("6-2. 연봉 결정요인 다변량 분석")

    num_features = FEATURE_NUMERIC_COLUMNS
    cat_features = CATEGORICAL_COLUMNS + EXTRA_CATEGORICAL_COLUMNS
    X = df[num_features + cat_features]
    y = df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", Ridge(alpha=1.0))]
    )
    pipeline.fit(x_train, y_train)
    enriched_r2 = float(r2_score(y_test, pipeline.predict(x_test)))

    # 전처리 후 피처명 ↔ Ridge 계수 매핑
    feat_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipeline.named_steps["regressor"].coef_
    coef_s = pd.Series(coefs, index=feat_names)

    # 연봉을 가장 크게 좌우하는 요인 Top 10 (계수 절대값 기준)
    top = coef_s.reindex(coef_s.abs().sort_values(ascending=False).index).head(10)
    ai_coefs = coef_s[[n for n in coef_s.index if "AISent" in n]].sort_values(
        ascending=False
    )
    vf = float(coef_s.get("cat__AISent_Very favorable", 0.0))
    vu = float(coef_s.get("cat__AISent_Very unfavorable", 0.0))

    # 학습에 실제로 쓰인 피처 개수(원-핫 전개 전, 원본 컬럼 기준)
    n_baseline = len(FEATURE_NUMERIC_COLUMNS) + len(CATEGORICAL_COLUMNS)
    n_enriched = len(num_features) + len(cat_features)

    result = MultivariateResult(
        baseline_r2=baseline_r2,
        enriched_r2=enriched_r2,
        n_baseline_features=n_baseline,
        n_enriched_features=n_enriched,
        top_coefficients=[(str(name), float(val)) for name, val in top.items()],
        ai_coefficients=[(str(name), float(val)) for name, val in ai_coefs.items()],
        vf_coef=vf,
        vu_coef=vu,
    )

    reporter.log(
        f"[다변량] 기존 {n_baseline}변수 R²={baseline_r2:.4f} "
        f"→ 확장 {n_enriched}변수 R²={enriched_r2:.4f} ({result.gain:+.4f})"
    )
    reporter.log(
        f" - AISent 순효과: Very favorable({vf:+,.0f}) vs Very unfavorable({vu:+,.0f})"
    )
    reporter.log(
        f" - 역설 지속 여부(통제 후에도 회의론자 계수가 높은가): {result.paradox_persists}"
    )

    return result
