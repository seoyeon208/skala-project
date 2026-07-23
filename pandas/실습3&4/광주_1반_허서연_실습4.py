"""
작성자: 허서연

프로그램 전체 설명:
    Pandas와 다양한 라이브러리(Seaborn, SciPy, Scikit-learn, Plotly)를 활용하여 데이터를 다각도로 분석합니다.
    실습 3의 산출물과 연계하여 이상치가 제거된 데이터를 시각화 및 통계 검정에 활용하며,
    원본 데이터는 머신러닝 파이프라인 학습에 사용합니다.

변경내역(머리말):
    - 2026-07-21: 문제 1~4 작성 (2x2 서브플롯 시각화, t-test 및 카이제곱 검정, sklearn Pipeline 및 joblib 저장, Plotly 인터랙티브 차트 저장)
    - 2026-07-21: hue 다차원 인코딩, coolwarm 히트맵, grid, 귀무/대립 가설, 왜도/첨도/정규성 검정, Ridge, facet_col
    - 2026-07-21: 통계 검정 최종 해석 요약(Summary) 추가

문제 1: EDA 시각화 4종 (2x2 서브플롯)
문제 2: 통계 검정 — t-test + 카이제곱
문제 3: sklearn Pipeline 구성 + 저장
문제 4: Plotly 인터랙티브 차트 저장
"""

import warnings
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import joblib
import plotly.express as px

warnings.filterwarnings("ignore", category=FutureWarning)

# 폰트 설치 방식으로 한글 깨짐 해결 (pip install koreanize-matplotlib 활용)
import koreanize_matplotlib

# 실습 3 모듈 연계 (IQR 이상치 제거 함수 가져오기) — 실습 3, 실습 4 두 파일이 같은 폴더에 있어야 합니다.
try:
    from 광주_1반_허서연_실습3 import process_eda_and_outliers
except ImportError:
    raise ImportError(
        "[오류] '광주_1반_허서연.py' 파일을 찾을 수 없습니다.\n"
        "이 파일과 '광주_1반_허서연.py'가 동일한 폴더에 있는지 확인해주세요."
    )

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR


# ──────────────────────────────────────────────────────────────────────────────
# 공통 헬퍼 함수: p-value 해석 중복 제거
# ──────────────────────────────────────────────────────────────────────────────
def interpret_p_value(p_value, alpha=0.05):
    """
    함수/기능 설명:
    p-value와 유의수준(alpha)을 받아 통계적 유의미성 여부를 문자열로 반환합니다.
    t-test / 카이제곱 두 곳에서 재사용하여 중복 로직을 제거합니다.
    """
    if p_value < alpha:
        return f"귀무가설 기각 — 유의미함 (p={p_value:.4f} < α={alpha})"
    else:
        return f"귀무가설 채택 — 유의미하지 않음 (p={p_value:.4f} ≥ α={alpha})"


def run_eda_subplots(df_clean):
    """
    함수/기능 설명:
    (문제 1) EDA 시각화 4종 (2x2 서브플롯)
    실습 3에서 IQR 이상치가 제거된 데이터를 입력받아 4종의 차트를 한 Figure에 구성합니다.
    - 1: 매출액 분포 히스토그램 + KDE
    - 2: 지역·카테고리별 매출 박스플롯 + hue
    - 3: 월별 총매출 바차트
    - 4: 수치형 변수 상관관계 히트맵
    저장: dpi=150 고화질 저장 (수업 자료 반영)
    """
    print("\n===== 문제 1: EDA 시각화 4종 (2x2 서브플롯) =====")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sales Data EDA (이상치 제거본)', fontsize=16, fontweight='bold')

        # 1. 히스토그램 + KDE: 매출액 분포를 시각화하여 왜도와 이상치 감지를 돕습니다.
        sns.histplot(data=df_clean, x='amount', kde=True, bins=40, color='steelblue', ax=axes[0, 0])
        axes[0, 0].set_title('매출액 분포 (히스토그램 + KDE)')
        axes[0, 0].set_xlabel('매출액 (원)')
        axes[0, 0].set_ylabel('빈도')
        axes[0, 0].grid(alpha=0.3)

        # 2. 박스플롯 — 지역별 매출액 + hue='category' 다차원 인코딩 (수업 자료 반영)
        sns.boxplot(data=df_clean, x='region', y='amount', hue='category', ax=axes[0, 1], palette='Set2')
        axes[0, 1].set_title('지역·카테고리별 매출액 (박스플롯 + hue)')
        axes[0, 1].set_xlabel('지역명')
        axes[0, 1].set_ylabel('결제 금액 (원)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend(title='카테고리', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

        # 3. 월별 총매출 라인차트: 집계된 시계열 패턴을 시각화합니다.
        df_copy = df_clean.copy()
        df_copy['month'] = df_copy['order_date'].dt.to_period('M').astype(str)
        monthly_sales = df_copy.groupby('month')['amount'].sum().reset_index()

        axes[1, 0].plot(monthly_sales['month'], monthly_sales['amount'],
                        color='teal', marker='o', linewidth=2)
        axes[1, 0].set_title('월별 총매출 (라인차트)')
        axes[1, 0].set_xlabel('주문 월 (연-월)')
        axes[1, 0].set_ylabel('월별 총 매출액 (원)')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. 수치형 상관관계 히트맵 (cmap='coolwarm')
        numeric_df = df_clean[['quantity', 'unit_price', 'customer_age', 'amount']]
        corr = numeric_df.corr()

        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1, 1], fmt='.2f', linewidths=0.5)
        axes[1, 1].set_title('수치형 변수 상관 히트맵 (coolwarm)')
        axes[1, 1].set_xlabel('수치형 변수')
        axes[1, 1].set_ylabel('수치형 변수')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        output_path = OUTPUT_DIR / 'eda_subplots.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"시각화 '{output_path.name}' 저장 성공")

    except Exception as e:
        print(f"시각화 도중 예기치 않은 오류가 발생했습니다: {e}")


def run_statistical_tests(df_clean):
    """
    함수/기능 설명:
    (문제 2) 통계 검정 (t-test + 카이제곱)
    수업 자료 반영: 귀무/대립 가설 명시 + 왜도/첨도 + Shapiro 정규성 검정 추가
    - 서울 vs 부산 평균 매출 차이를 독립표본 t-test로 검정합니다.
    - 지역과 카테고리 간의 연관성(독립성)을 카이제곱 검정으로 확인합니다.
    - interpret_p_value() 헬퍼 함수를 재사용하여 중복 로직을 제거합니다.
    """
    print("\n===== 문제 2: 통계 검정 (t-test + 카이제곱) =====")
    try:
        # [사전 분석] amount 컬럼의 분포 특성 파악
        amount = df_clean['amount']
        print("[사전 분석] amount 컬럼 기술통계")
        print(f" - 평균: {amount.mean():,.0f}원  |  중앙값: {amount.median():,.0f}원")
        print(f" - 왜도(skewness): {amount.skew():.4f}  |  첨도(kurtosis): {amount.kurt():.4f}")

        # Shapiro-Wilk 정규성 검정 — 샘플 5000개로 속도 확보
        _, p_normal = stats.shapiro(amount.sample(5000, random_state=42))
        print(f" - 정규분포 검정(Shapiro) p값: {p_normal:.4f} → {'정규분포 아님' if p_normal < 0.05 else '정규분포'}")
        print("-" * 40)

        # [1] T-Test: 서울 vs 부산 평균 매출 차이
        # 귀무가설(H0): 서울과 부산의 평균 매출액은 같다
        # 대립가설(H1): 서울과 부산의 평균 매출액은 다르다  (수업 자료 반영: 가설 명시)
        print("[T-Test] 서울 vs 부산 매출 차이 검정")
        print(" - H0 (귀무가설): 서울과 부산의 평균 매출액은 같다")
        print(" - H1 (대립가설): 서울과 부산의 평균 매출액은 다르다")

        # 각 지역 데이터만 필터링하여 분리
        seoul_amount = df_clean[df_clean['region'] == '서울']['amount']
        busan_amount = df_clean[df_clean['region'] == '부산']['amount']

        # 맥락 이해를 위해 표본 크기와 평균값도 함께 출력
        print(f" - 서울: 표본수={len(seoul_amount):,}건, 평균={seoul_amount.mean():,.0f}원")
        print(f" - 부산: 표본수={len(busan_amount):,}건, 평균={busan_amount.mean():,.0f}원")

        # Welch's t-test 수행 (등분산 가정 안함: equal_var=False)
        t_stat, p_val_t = stats.ttest_ind(seoul_amount, busan_amount, equal_var=False)
        print(f" - t={t_stat:.3f}, p={p_val_t:.3f}")

        # 공통 헬퍼 함수로 p-value 해석 (중복 로직 제거)
        print(f" -> [해석] {interpret_p_value(p_val_t)}")
        print("-" * 40)

        # [2] 카이제곱 검정: 지역 × 카테고리 독립성
        # 귀무가설(H0): 지역과 카테고리는 독립이다 (연관성이 없다)
        # 대립가설(H1): 지역과 카테고리는 독립이 아니다 (연관성이 있다)
        print("[Chi-Square] 지역과 카테고리 독립성 검정")
        print(" - H0 (귀무가설): 지역과 카테고리는 서로 독립이다")
        print(" - H1 (대립가설): 지역과 카테고리 사이에 연관성이 있다")

        # 지역(행)과 카테고리(열)의 빈도수로 교차표(contingency table) 생성
        contingency_table = pd.crosstab(df_clean['region'], df_clean['category'])

        # 카이제곱 검정으로 두 범주형 변수 간 독립성 확인
        chi2, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)
        print(f" - 카이제곱={chi2:.3f}, p={p_val_chi2:.3f}, 자유도(dof)={dof}")

        # 공통 헬퍼 함수로 p-value 해석 (중복 로직 제거)
        print(f" -> [해석] {interpret_p_value(p_val_chi2)}")

        # 비즈니스 관점 최종 해석 요약
        print("\n[최종 통계 해석 요약]")
        print("1. [T-Test 분석]: 서울과 부산의 건당 결제 금액 규모에는 " +
              ("차이가 있으므로, 지역별 맞춤형 프로모션 전략이 필요합니다." if p_val_t < 0.05
               else "유의미한 차이가 없으므로, 동일한 가격 전략을 적용해도 무방합니다."))
        print("2. [Chi-Square 분석]: 지역과 카테고리 간에는 " +
              ("유의미한 연관성이 있으므로, 지역 특화 상품 배치가 효과적입니다." if p_val_chi2 < 0.05
               else "유의미한 연관성이 없으므로, 전국 단위 카테고리 마케팅이 유리합니다."))

    except KeyError as e:
        print(f"오류: 통계 검정에 필요한 컬럼이 없습니다: {e}")
    except Exception as e:
        print(f"통계 검정 도중 예기치 않은 오류 발생: {e}")


def run_sklearn_pipeline(file_path):
    """
    함수/기능 설명:
    (문제 3) sklearn Pipeline 구성 + 저장
    수업 자료 반영: LinearRegression 대신 Ridge(alpha=1.0) 사용
    - 전처리(ColumnTransformer)와 Ridge 회귀 모델을 Pipeline으로 통합합니다.
    - fit → predict → score 수행 후 joblib으로 저장 및 재로딩 검증합니다.
    """
    print("\n===== 문제 3: sklearn Pipeline 구성 + 저장 =====")
    try:
        # 원본 데이터 로드 및 결측치 제거 (Clean Data 준비)
        df = pd.read_csv(file_path)
        df = df.dropna().copy()

        # 수치형/범주형 피처 리스트 분리 정의
        numeric_features = ['quantity', 'unit_price', 'customer_age']
        categorical_features = ['region', 'category']

        # 위에서 정의한 리스트를 합쳐 X 구성 (중복 방지)
        X = df[numeric_features + categorical_features]
        y = df['amount']

        # 학습(80%)과 테스트(20%) 데이터 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 수치형: 표준화(StandardScaler), 범주형: 원핫인코딩(OneHotEncoder) 전처리기 구성
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

        # 전처리 단계와 Ridge 회귀 모델을 하나의 파이프라인으로 연결 (수업 자료 반영: Ridge 사용)
        # Ridge는 L2 정규화를 적용하여 과적합을 방지하는 선형 회귀 모델입니다.
        pipeline = Pipeline(steps=[
            ('prep', preprocessor),
            ('reg', Ridge(alpha=1.0))
        ])

        # 파이프라인 학습(fit) 및 테스트 평가(score)
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        print(f"파이프라인 모델 평가 점수 (R² Score): {score:.3f}")

        # 학습 완료된 파이프라인을 .pkl 파일로 저장(Serialize)
        model_filename = OUTPUT_DIR / 'sales_pipeline_model.pkl'
        joblib.dump(pipeline, model_filename)
        print(f"-> '{model_filename.name}' 파일 모델 저장 완료.")

        # 저장된 파일을 재로딩(Deserialize)하여 정상 작동 검증
        loaded = joblib.load(model_filename)
        print(f"-> 재로딩된 모델 R² 검증: {loaded.score(X_test, y_test):.3f}")

    except Exception as e:
        print(f"파이프라인 구축 및 모델 저장 도중 오류 발생: {e}")


def run_plotly_interactive(df_clean):
    """
    함수/기능 설명:
    (문제 4) Plotly 인터랙티브 차트 저장
    - 지역·카테고리별 매출 막대차트, facet_col='region'으로 지역별 분할
    """
    print("\n===== 문제 4: Plotly 인터랙티브 차트 저장 =====")
    try:
        # 지역·카테고리별 매출 막대차트 — facet_col='region'으로 지역별 분할 서브플롯 
        agg_df = df_clean.groupby(['region', 'category'])['amount'].sum().reset_index()

        fig_bar = px.bar(
            agg_df,
            x='category',
            y='amount',
            color='category',
            facet_col='region',          # 지역별 분할 서브플롯
            title='지역·카테고리별 총매출액',
            labels={'amount': '총매출액 (원)', 'category': '카테고리', 'region': '지역'}
        )
        
        # '지역=' 접두어 제거
        fig_bar.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        # x축의 눈금(카테고리 이름)과 기본 x축 제목을 모두 제거
        fig_bar.update_xaxes(showticklabels=False, title_text='')

        # update_layout으로 제목·폰트 커스터마이징 및 우측 범례 표시  
        fig_bar.update_layout(
            title_font_size=16,
            showlegend=True,             # 우측에 카테고리 인덱스(범례) 표시
            legend_title_text='카테고리'
        )

        # 중앙 하단에 x축 제목('카테고리') 하나만 추가
        fig_bar.add_annotation(
            x=0.5, y=-0.1, xref='paper', yref='paper',
            text='카테고리', showarrow=False, font=dict(size=14)
        )
        
        bar_filename = OUTPUT_DIR / 'plotly_sales_bar.html'
        fig_bar.write_html(bar_filename)
        print(f"-> 지역·카테고리 막대차트 : '{bar_filename.name}' 저장 완료.")

    except ImportError:
        print("오류: plotly 모듈을 찾을 수 없습니다. (설치 권장: pip install plotly)")
    except Exception as e:
        print(f"Plotly 차트 생성 및 저장 도중 오류 발생: {e}")


if __name__ == "__main__":
    TARGET_FILE = BASE_DIR / 'sales_100k.csv'

    # 실습 3 모듈 연계: 이상치 제거된 데이터프레임 수신
    print("\n[연계 작업] 실습 3 모듈에서 데이터 로드 및 이상치 필터링 수행...")
    df_clean, lower_bound, upper_bound = process_eda_and_outliers(TARGET_FILE)

    if df_clean is not None:
        # 문제 1: 2x2 서브플롯 EDA (이상치 제거 데이터)
        run_eda_subplots(df_clean)

        # 문제 2: 통계 검정 — 귀무/대립 가설 + 왜도/첨도/정규성 + t-test + 카이제곱 (이상치 제거 데이터)
        run_statistical_tests(df_clean)

        # 문제 3: Ridge 파이프라인 + joblib 저장 (원본 데이터)
        run_sklearn_pipeline(TARGET_FILE)

        # 문제 4: Plotly 인터랙티브 차트 저장 (이상치 제거 데이터)
        run_plotly_interactive(df_clean)
