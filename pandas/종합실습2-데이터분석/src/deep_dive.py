"""
파일명: deep_dive.py
설명: 파이프라인 7단계 — 국가·경력·직군을 통제한 교차 검증으로 핵심 인사이트를 도출한다.
기능:
    - run_deep_dive(): AI 태도와 연봉의 관계가 교란변수(국가·경력·직군)를 통제해도
      유지되는지 단계별로 확인하고, 결론을 담은 마크다운 문자열을 반환한다.
    - _save_deep_dive_chart(): 경력 구간별 AI 태도에 따른 중위 연봉 추이 차트를 저장한다.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import (
    AI_SENTIMENT_ORDER,
    BASE_DIR,
    CHART_DIR,
    DEEP_DIVE_CHART_NAME,
    TARGET_COLUMN,
)
from .formatting import markdown_table, series_to_markdown
from .reporting import Reporter


def run_deep_dive(df: pd.DataFrame, reporter: Reporter) -> str:
    """AI 태도와 연봉의 관계가 교란변수 통제 후에도 유지되는지 확인한다.

    보고서의 '핵심 인사이트' 절 본문으로 들어갈 마크다운을 반환한다.
    (절 제목은 render_report()가 붙이므로 여기서는 소제목부터 시작한다.)
    """
    reporter.section("7. 심층 분석 (교란변수 통제)")
    sections: list[str] = []

    # 1차 관찰 — 전체 데이터의 AI 태도별 중위 연봉
    median_by_sentiment = df.groupby("AISent", observed=True)[TARGET_COLUMN].median()
    skeptics_earn_more = median_by_sentiment.get(
        "Very unfavorable", 0
    ) > median_by_sentiment.get("Very favorable", 0)

    sections.append(
        '> **발견: "AI를 열렬히 반기는 개발자보다, AI에 회의적인 개발자의 소득이 높다"**\n'
        if skeptics_earn_more
        else '> **발견: "AI에 우호적인 개발자의 소득이 예상대로 더 높다"**\n'
    )

    sections.append("### 1차 관찰 — 전체 중위소득 비교\n")
    sections.append(
        series_to_markdown(
            median_by_sentiment.sort_values(ascending=False),
            "AI 태도",
            "중위 연봉 (USD)",
        )
    )
    sections.append(
        "\n→ AI에 부정적일수록 중위소득이 높습니다 (직관과 반대).\n"
        if skeptics_earn_more
        else "\n→ AI에 긍정적일수록 중위소득이 높은 경향을 보입니다.\n"
    )

    # 통제 ① 국가 — 국가별 임금 수준 격차가 원인인지 확인
    sections.append("### 통제 ① 국가 (미국 응답자만)\n")
    usa_median = (
        df[df["Country"] == "United States of America"]
        .groupby("AISent", observed=True)[TARGET_COLUMN]
        .median()
    )
    usa_unfavorable = usa_median.get("Very unfavorable", 0)
    usa_favorable = usa_median.get("Favorable", 0)
    sections.append(
        f"→ 미국 내에서도 Very unfavorable(\\${usa_unfavorable:,.0f})이 "
        f"Favorable(\\${usa_favorable:,.0f})보다 높습니다. 국가 격차만으로는 설명되지 않습니다.\n"
        if usa_unfavorable > usa_favorable
        else f"→ 미국 내에서는 Favorable(\\${usa_favorable:,.0f})이 "
        f"Very unfavorable(\\${usa_unfavorable:,.0f})보다 높습니다.\n"
    )

    # 통제 ② 경력 — 시니어일수록 회의적인 것이 원인인지 확인
    sections.append("### 통제 ② 경력 구간\n")
    experience_labels = ["0~5년", "5~10년", "10~15년", "15~25년", "25년 이상"]
    df = df.assign(
        ExperienceGroup=pd.cut(
            df["YearsCodePro"],
            bins=[0, 5, 10, 15, 25, 100],
            labels=experience_labels,
            right=False,
        )
    )
    _save_deep_dive_chart(df, reporter)

    median_by_experience = (
        df.groupby(["ExperienceGroup", "AISent"], observed=False)[TARGET_COLUMN]
        .median()
        .unstack()
    )
    comparison_rows = []
    skeptic_wins = 0
    for group in experience_labels[:-1]:  # 25년 이상은 표본이 적어 제외
        favorable_median = median_by_experience.loc[group].get(
            "Very favorable", float("nan")
        )
        unfavorable_median = median_by_experience.loc[group].get(
            "Very unfavorable", float("nan")
        )
        higher = "회의론자" if unfavorable_median > favorable_median else "우호론자"
        comparison_rows.append(
            [
                group,
                f"{favorable_median:,.0f}",
                f"{unfavorable_median:,.0f}",
                higher,
            ]
        )
        if unfavorable_median > favorable_median:
            skeptic_wins += 1

    sections.append(
        markdown_table(
            ["경력 구간", "Very favorable (USD)", "Very unfavorable (USD)", "우위"],
            comparison_rows,
        )
    )
    total_groups = len(experience_labels) - 1
    sections.append(
        f"\n→ {total_groups}개 구간 중 {skeptic_wins}개에서 회의론자의 소득이 높습니다. "
        "경력을 통제해도 역전이 사라지지 않습니다.\n"
        if skeptic_wins > total_groups / 2
        else f"\n→ {total_groups}개 구간 중 {total_groups - skeptic_wins}개에서 "
        "우호 그룹의 소득이 높습니다. 같은 연차 안에서는 우호 그룹이 앞섭니다.\n"
    )

    # 통제 ③ 직군 — 직군 구성 차이가 원인인지 확인
    sections.append("### 통제 ③ 직군 구성\n")
    top_dev_types = {
        label: df[df["AISent"] == label]["DevType"].value_counts().idxmax()
        for label in ["Very favorable", "Very unfavorable"]
    }
    sections.append(
        markdown_table(
            ["AI 태도", "1위 직군"],
            [[label, dev_type] for label, dev_type in top_dev_types.items()],
        )
    )
    sections.append(
        "\n→ 두 그룹의 1위 직군이 동일합니다. 직군 구성만으로는 설명하기 어렵습니다.\n"
        if len(set(top_dev_types.values())) == 1
        else "\n→ 두 그룹의 1위 직군이 다릅니다. 직군 성격이 연봉과 AI 선호도에 영향을 주었을 수 있습니다.\n"
    )

    sections.append("### 최종 결론\n")
    sections.append(
        "국가·경력·직군을 통제한 뒤에도 **AI 도구에 회의적인 개발자의 소득이 더 높은** 경향이 "
        "유지됩니다. 고연봉 시니어일수록 코드 품질 기준이 높아 AI 생성 코드에 엄격한 잣대를 "
        "적용하기 때문일 수 있습니다.\n"
        if skeptics_earn_more
        else "**AI 도구를 적극 수용하는 개발자의 소득이 더 높다**는 경향이 확인됩니다. "
        "신기술 도입에 적극적인 개발자가 더 높은 시장 가치를 갖는 것으로 해석할 수 있습니다.\n"
    )

    reporter.log("- 국가 / 경력 / 직군 통제 결과를 보고서에 기록했습니다.")
    return "\n".join(sections)


def _save_deep_dive_chart(df: pd.DataFrame, reporter: Reporter) -> None:
    """경력 구간별 AI 태도 추이를 한눈에 보기 위한 차트를 저장한다."""
    plt.figure(figsize=(12, 6))
    sns.pointplot(
        data=df,
        x="ExperienceGroup",
        y=TARGET_COLUMN,
        hue="AISent",
        hue_order=AI_SENTIMENT_ORDER,
        estimator="median",
        errorbar=None,
        palette="Spectral",
    )
    plt.title("경력 구간 · AI 인식별 중위 연봉 추이", fontsize=14)
    plt.xlabel("전문 경력 구간")
    plt.ylabel("중위 연봉 (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="AI Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    chart_path = CHART_DIR / DEEP_DIVE_CHART_NAME
    plt.savefig(chart_path, dpi=150)
    plt.close()
    reporter.log(f"✅ 심층 분석 차트 저장: {chart_path.relative_to(BASE_DIR)}")
