"""
파일명: viz.py
설명: 파이프라인 4단계 — Seaborn 정적 차트와 Plotly 인터랙티브 차트를 생성한다.
기능:
    - configure_korean_font(): 차트의 한글 라벨이 깨지지 않도록 시스템 한글 폰트를 지정한다.
      (원본은 import 시점에 폰트를 잡았지만, 여기서는 함수로 두고 main()이 명시적으로 호출한다.)
    - create_charts(): AI 태도별 연봉 분포 박스플롯(정적)과 국가별 평균 연봉 비교
      막대그래프(인터랙티브)를 만들어 파일로 저장한다.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import font_manager

from .config import (
    AI_SENTIMENT_ORDER,
    BASE_DIR,
    BOXPLOT_NAME,
    CHART_DIR,
    INTERACTIVE_NAME,
    TARGET_COLUMN,
)
from .reporting import Reporter


def configure_korean_font() -> None:
    """차트의 한글 라벨이 깨지지 않도록 시스템에 설치된 한글 폰트를 지정한다."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in (
        "AppleGothic",
        "Apple SD Gothic Neo",
        "NanumGothic",
        "Malgun Gothic",
    ):
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = (
        False  # 마이너스 기호가 네모로 표시되는 것 방지
    )


def create_charts(df: pd.DataFrame, reporter: Reporter) -> None:
    """정적 차트 1개(Seaborn)와 인터랙티브 차트 1개(Plotly)를 저장한다."""
    reporter.section("4. 시각화")

    # [정적] AI 태도별 연봉 분포 — 그룹 비교용 박스플롯
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="AISent",
        y=TARGET_COLUMN,
        hue="AISent",
        order=AI_SENTIMENT_ORDER,
        hue_order=AI_SENTIMENT_ORDER,
        palette="coolwarm",
        legend=False,
        showfliers=False,  # 이미 상위 1%를 절단했고, 박스 형태를 보기 쉽게 한다
    )
    plt.title("AI 도구 인식(AISent)에 따른 연봉 분포", fontsize=14)
    plt.xlabel("AI 도구 사용에 대한 태도 (AI Sentiment)")
    plt.ylabel("연봉 (USD)")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    boxplot_path = CHART_DIR / BOXPLOT_NAME
    plt.savefig(boxplot_path, dpi=150)
    plt.close()
    reporter.log(f"✅ 정적 차트(Seaborn) 저장: {boxplot_path.relative_to(BASE_DIR)}")

    # [인터랙티브] 주요 10개국의 AI 태도별 평균 연봉 비교
    top_countries = df["Country"].value_counts().nlargest(10).index
    grouped = (
        df[df["Country"].isin(top_countries)]
        .groupby(["Country", "AISent"], observed=True)[TARGET_COLUMN]
        .mean()
        .reset_index()
    )
    figure = px.bar(
        grouped,
        x="Country",
        y=TARGET_COLUMN,
        color="AISent",
        barmode="group",
        category_orders={"AISent": AI_SENTIMENT_ORDER},
        title="주요 국가별 AI 도구 인식에 따른 평균 연봉 비교",
        labels={
            TARGET_COLUMN: "평균 연봉 (USD)",
            "Country": "국가",
            "AISent": "AI 인식",
        },
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    figure.update_layout(title_font_size=16, xaxis_tickangle=-20)
    interactive_path = CHART_DIR / INTERACTIVE_NAME
    figure.write_html(interactive_path)
    reporter.log(
        f"✅ 인터랙티브 차트(Plotly) 저장: {interactive_path.relative_to(BASE_DIR)}"
    )
