# Stack Overflow 2024 — AI 태도와 개발자 연봉 분석 보고서

> 이 문서는 `python3 -m src.main` 실행 시 자동 생성됩니다. (생성 시각: 2026-07-21 18:14)

## 1. 분석 개요

| 항목 | 내용 |
| --- | --- |
| 분석 주제 | AI 도구에 대한 태도가 개발자 연봉과 어떤 관계를 갖는가 |
| 데이터 | Stack Overflow Developer Survey 2024 |
| 원본 파일 | `results.csv` |
| 원본 표본 | 65,437건 |
| 분석 표본 | 17,535건 (전처리 후) |
| 종속 변수 | `ConvertedCompYearly` (연봉, USD) |
| 핵심 독립 변수 | `AISent` (AI 도구에 대한 태도, 6단계) |

**연구 가설**

- 귀무가설 H₀: AI 도구에 우호적인 그룹과 그 외 그룹의 평균 연봉은 같다.
- 대립가설 H₁: 두 그룹의 평균 연봉은 다르다. (양측 검정, α = 0.05)

→ **검정 결과: H₀ 기각** (자세한 내용은 5장 참고)

## 2. 데이터 준비

### 2.1 Pandas · Polars 로딩 비교

동일한 CSV를 두 라이브러리로 각각 읽어 속도와 결과를 대조했습니다.

| 항목 | Pandas | Polars |
| --- | --- | --- |
| 로딩 시간 (3회 평균) | 0.9469초 | 0.0351초 |
| 형태 (행 × 열) | 65,437 × 114 | 65,437 × 114 |
| 관심 컬럼 결측치 | 126,701 | 126,701 |

→ **Polars가 약 27.0배 빠릅니다.** 형태 일치: ✅ · 결측치 집계 일치: ✅

> 이 설문 CSV는 결측치를 문자열 `NA`로 표기합니다. Pandas는 이를 자동으로 결측 처리하지만
> Polars는 `null_values`를 지정해야 하며, 지정하지 않으면 숫자 컬럼이 문자열로 읽히고
> 결측치가 0건으로 잘못 집계됩니다.

### 2.2 결측치 · 중복 · 이상치 처리

| 컬럼 | 결측치 | 비율 |
| --- | --- | --- |
| ConvertedCompYearly | 42,002 | 64.2% |
| YearsCodePro | 13,827 | 21.1% |
| YearsCode | 5,568 | 8.5% |
| AISent | 19,564 | 29.9% |
| Country | 6,507 | 9.9% |
| DevType | 5,992 | 9.2% |
| EdLevel | 4,653 | 7.1% |
| OrgSize | 17,957 | 27.4% |
| RemoteWork | 10,631 | 16.2% |
| Age | 0 | 0.0% |

적용한 처리:

1. **결측치 제거** — 관심 컬럼에 결측이 있는 행 삭제
2. **중복 제거** — 완전 중복 6,063건 삭제
3. **표기 정규화** — `Less than 1 year` → 0.5, `More than 50 years` → 51.0
4. **이상치 절단** — 연봉 상위 1%($396,114 초과) 제외

→ **65,437건 → 17,535건** (26.8% 잔존)

## 3. 탐색적 데이터 분석 (EDA)

### 3.1 기술통계

| 통계량 | ConvertedCompYearly | YearsCodePro | YearsCode |
| --- | --- | --- | --- |
| count | 17,535.00 | 17,535.00 | 17,535.00 |
| mean | 75,138.71 | 9.70 | 14.27 |
| std | 62,327.60 | 8.28 | 9.51 |
| min | 1.00 | 0.50 | 0.50 |
| 25% | 29,057.00 | 4.00 | 7.00 |
| 50% | 62,000.00 | 7.00 | 12.00 |
| 75% | 104,476.00 | 13.00 | 19.00 |
| max | 386,662.00 | 51.00 | 51.00 |

### 3.2 AI 태도별 응답자 분포

| AI 태도 | 응답자 | 비율 |
| --- | --- | --- |
| Very unfavorable | 223 | 1.3% |
| Unfavorable | 927 | 5.3% |
| Indifferent | 3,246 | 18.5% |
| Unsure | 422 | 2.4% |
| Favorable | 8,573 | 48.9% |
| Very favorable | 4,144 | 23.6% |

### 3.3 상위 5개 국가

전체 163개국 중 상위 5개국입니다.

| 국가 | 응답자 | 비율 |
| --- | --- | --- |
| United States of America | 3,123 | 17.8% |
| Germany | 1,434 | 8.2% |
| Ukraine | 1,269 | 7.2% |
| India | 919 | 5.2% |
| United Kingdom of Great Britain and Northern Ireland | 905 | 5.2% |

## 4. 시각화

### 4.1 AI 도구 인식별 연봉 분포 (Seaborn · 정적)

![AI 태도별 연봉 분포](./outputs/charts/ai_sentiment_income_boxplot.png)

### 4.2 주요 국가별 AI 인식에 따른 평균 연봉 (Plotly · 인터랙티브)

[▶ 인터랙티브 차트 열기](./outputs/charts/country_ai_sentiment_income.html)

### 4.3 경력 구간별 AI 인식과 연봉 추이 (Seaborn · 정적)

![경력별 AI 인식 추이](./outputs/charts/experience_ai_sentiment_income.png)

## 5. 통계 분석

### 5.1 수치형 변수 간 상관계수 (Pearson)

| 변수 | ConvertedCompYearly | YearsCodePro | YearsCode |
| --- | --- | --- | --- |
| ConvertedCompYearly | 1.000 | 0.414 | 0.417 |
| YearsCodePro | 0.414 | 1.000 | 0.920 |
| YearsCode | 0.417 | 0.920 | 1.000 |

→ 전문 코딩 경력과 연봉의 상관계수는 **r = 0.414** (p < 1e-308)로, 뚜렷한 양의 상관관계입니다.

### 5.2 독립표본 t-검정 (Welch)

두 그룹의 표본 크기와 분산이 다르므로 등분산을 가정하지 않는 Welch's t-test를 사용했습니다.

| 그룹 | 표본 수 | 평균 연봉 (USD) |
| --- | --- | --- |
| AI 우호 (Favorable · Very favorable) | 12,717 | 73,187 |
| 그 외 | 4,818 | 80,290 |

| 검정 통계량 | 값 |
| --- | --- |
| t-statistic | -6.716 |
| p-value | 1.979e-11 |
| 유의수준 α | 0.05 |
| 평균 차이 (우호 − 그 외) | $-7,103 |
| 95% 신뢰구간 | [$-9,176, $-5,030] |
| Cohen's d (효과크기) | -0.114 (무시할 수준 (0.2 미만)) |

**p-value 해석**: 귀무가설 기각 — 통계적으로 유의미함 (p = 1.979e-11, α = 0.05)

> 표본이 약 1.7만 건으로 크면 아주 작은 차이도 p-value가 유의하게 나오므로, **효과크기(Cohen's d)** 로 '차이가 실질적으로도 큰지'를 함께 판단했습니다.

**결론: AI에 비우호적이거나 무관심한 개발자 그룹의 평균 연봉이 통계적으로 유의미하게 높습니다. 다만 Cohen's d=-0.114(무시할 수준 (0.2 미만))로, '유의미 ≠ 실질적으로 큼'에 주의해야 합니다.**

## 6. 머신러닝 파이프라인

### 6.1 파이프라인 구성

전처리와 모델을 하나의 `sklearn.pipeline.Pipeline`으로 묶어, 학습 데이터의 통계가
테스트 데이터로 새지 않도록 했습니다.

| 단계 | 구성 요소 | 대상 컬럼 |
| --- | --- | --- |
| 수치형 전처리 | `StandardScaler` | `YearsCodePro`, `YearsCode` |
| 범주형 전처리 | `OneHotEncoder(handle_unknown="ignore")` | `AISent`, `Country`, `DevType` |
| 모델 | `Ridge(alpha=1.0)` | 타깃: `ConvertedCompYearly` |

학습/테스트 분할: **14,028건 / 3,507건** (8:2, `random_state=42`)

### 6.2 평가 지표

| 지표 | 값 | 의미 |
| --- | --- | --- |
| R² Score | 0.5561 | 모델이 설명하는 연봉 분산의 비율 |
| RMSE | $42,439 | 예측 오차의 제곱평균제곱근 |
| MAE | $28,783 | 예측 오차의 절댓값 평균 |

저장된 모델: `outputs/models/ai_income_pipeline.joblib`

### 6.3 연봉 결정요인 다변량 분석

6.1의 기본 모델에 학력(EdLevel)·조직규모(OrgSize)·근무형태(RemoteWork)·연령(Age)을
추가해, 여러 요인을 **한 모델에서 동시에 통제**했을 때 각 요인의 순효과를 봅니다.

| 모델 | 변수 수 | R² |
| --- | --- | --- |
| 기본 | 5 | 0.5561 |
| 확장 | 9 | 0.5918 |

→ 변수 추가로 설명력 **+0.0357** 변화 (연봉 분산의 59.2%를 설명).

#### 연봉을 가장 크게 좌우하는 요인 Top 10 (계수, USD 근사)

| 요인 | 계수 |
| --- | --- |
| Country_United States of America | +90,272 |
| Country_Switzerland | +69,175 |
| Country_Andorra | +61,000 |
| Country_Singapore | +60,026 |
| Country_Israel | +56,887 |
| Age_65 years or older | -45,302 |
| Country_Antigua and Barbuda | +44,066 |
| Country_Australia | +43,291 |
| Country_Denmark | +43,062 |
| Country_Canada | +42,899 |

#### AISent 순효과 — "다 통제해도 역설이 남는가"

| AI 태도 | 계수 |
| --- | --- |
| Very unfavorable | +1,900 |
| Very favorable | +1,688 |
| Favorable | -128 |
| Indifferent | -205 |
| Unfavorable | -1,411 |
| Unsure | -1,845 |

→ 학력·조직·근무형태·연령·경력·국가·직군을 **모두 통제한 뒤에도** Very unfavorable 계수(+1,900)가 Very favorable(+1,688)보다 높습니다. **역설은 교란변수로 설명되지 않고 잔존합니다.**

> ⚠️ 해석 주의: 수치형(경력)만 표준화했고 범주형은 원-핫이라, 계수는 '해당 범주일 때 연봉 기여도(USD 근사)'로 읽되 절대값보다 **부호와 상대 크기**에 무게를 둘 것.

## 7. 핵심 인사이트

> **발견: "AI를 열렬히 반기는 개발자보다, AI에 회의적인 개발자의 소득이 높다"**

### 1차 관찰 — 전체 중위소득 비교

| AI 태도 | 중위 연봉 (USD) |
| --- | --- |
| Very unfavorable | 76,433 |
| Unfavorable | 70,063 |
| Indifferent | 66,954 |
| Favorable | 60,000 |
| Unsure | 60,000 |
| Very favorable | 59,157 |

→ AI에 부정적일수록 중위소득이 높습니다 (직관과 반대).

### 통제 ① 국가 (미국 응답자만)

→ 미국 내에서도 Very unfavorable(\$153,500)이 Favorable(\$140,000)보다 높습니다. 국가 격차만으로는 설명되지 않습니다.

### 통제 ② 경력 구간

| 경력 구간 | Very favorable (USD) | Very unfavorable (USD) | 우위 |
| --- | --- | --- | --- |
| 0~5년 | 26,985 | 38,154 | 회의론자 |
| 5~10년 | 61,866 | 82,102 | 회의론자 |
| 10~15년 | 76,800 | 83,220 | 회의론자 |
| 15~25년 | 91,295 | 105,353 | 회의론자 |

→ 4개 구간 중 4개에서 회의론자의 소득이 높습니다. 경력을 통제해도 역전이 사라지지 않습니다.

### 통제 ③ 직군 구성

| AI 태도 | 1위 직군 |
| --- | --- |
| Very favorable | Developer, full-stack |
| Very unfavorable | Developer, full-stack |

→ 두 그룹의 1위 직군이 동일합니다. 직군 구성만으로는 설명하기 어렵습니다.

### 최종 결론

국가·경력·직군을 통제한 뒤에도 **AI 도구에 회의적인 개발자의 소득이 더 높은** 경향이 유지됩니다. 고연봉 시니어일수록 코드 품질 기준이 높아 AI 생성 코드에 엄격한 잣대를 적용하기 때문일 수 있습니다.

## 8. 부록 — Pandas vs Polars 성능 벤치마크

동일 작업을 두 라이브러리로 각 3회 실행한 평균 시간입니다 (로딩은 I/O 포함, 필터·집계는 메모리 프레임 대상).

| 작업 | Pandas(초) | Polars(초) | Polars 배속 |
| --- | --- | --- | --- |
| 로딩(read_csv) | 1.0608 | 0.0488 | 21.75x |
| 필터(notna) | 0.0106 | 0.0074 | 1.43x |
| 집계(groupby mean) | 0.0015 | 0.0098 | 0.15x |

![Pandas vs Polars 벤치마크](./outputs/charts/pandas_polars_benchmark.png)
