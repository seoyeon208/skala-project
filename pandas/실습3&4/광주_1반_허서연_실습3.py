"""
작성자: 허서연

프로그램 전체 설명:
    Pandas, Polars, DuckDB 세 가지 도구를 활용하여 sales 데이터를 분석합니다.
    기본 EDA와 IQR 기반 이상치 제거, GroupBy 집계, 성능 비교

변경내역(머리말):
    - 2026-07-21: 문제 1~4 작성 (Pandas EDA + IQR 이상치 제거, groupby named aggregation, Polars Lazy API, DuckDB SQL + timeit 성능 비교)
    - 2026-07-21: 중복 코드 제거 (4번 성능 비교 시 기존 함수 재사용, verbose 플래그 도입으로 중복 출력 방지)
    - 2026-07-21: 코드 고도화 (file_path 매개변수화, between 마스크 변수 재사용, 모든 함수 예외처리 보강, docstring 보완)
    - 2026-07-21: 성능 비교 고도화 (tracemalloc을 통한 메모리 측정 및 Pandas 파일 로드 포함 공정 비교 적용)

문제 1: Pandas EDA 기초 탐색 + 이상치 처리
문제 2: Pandas groupby named aggregation
문제 3: Polars Lazy API로 동일 집계 작성
문제 4: DuckDB SQL + 세 도구 성능 비교
"""

import pandas as pd
import polars as pl
import duckdb
import timeit


def process_eda_and_outliers(file_path):
    """
    함수/기능 설명:
    (문제 1) 파일 경로를 매개변수로 받아 CSV 파일을 로드하고, 기본/추가 EDA를 수행한 뒤,
    'amount' 컬럼 기준으로 IQR 방식을 활용하여 이상치를 제거합니다.
    - 이상치가 없을 경우를 대비해 임의 이상치(999999999)를 삽입 후 제거를 수행합니다.
    - 정상 범위: between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    """
    try:
        # 데이터 로드
        df = pd.read_csv(file_path)

        print("===== 문제 1: 데이터 EDA =====")

        # 수업 시간에 배운 추가 EDA
        print("\n행수, 열수")
        print(df.shape)

        print("\n데이터 타입")
        df.info()

        print("\n데이터 수치 통계")
        print(df.describe())

        print("\n데이터 통계 범주형 포함")
        print(df.describe(include='all'))

        # 타입 변환
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['region'] = df['region'].astype('category')

        print("\n컬럼 선택: df[['region','amount']]")
        print(df[['region', 'amount']].head())

        print("\n조건 필터: df.loc[df['amount']>1000]")
        print(df.loc[df['amount'] > 1000].head())

        print("\n결측치 파악")
        print(df.isna().sum())

        # 임의의 이상치 추가 (이상치가 아예 없을 때 대비)
        df.loc[len(df)] = df.iloc[0]
        df.at[len(df) - 1, 'amount'] = 999999999

        before_count = len(df)
        print(f"\n이상치 제거 전 행 수: {before_count}")

        # IQR 범위 계산
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 마스크를 변수로 저장하여 중복 계산 방지
        mask = df['amount'].between(lower_bound, upper_bound)
        df_filtered = df[mask]

        after_count = len(df_filtered)
        print(f"이상치 제거 후 행 수: {after_count}")
        print(f"제거된 이상치 건수: {(~mask).sum()}건")

        return df_filtered, lower_bound, upper_bound

    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return None, None, None
    except Exception as e:
        print(f"실행 중 예기치 않은 오류가 발생했습니다: {e}")
        return None, None, None


def process_groupby_aggregation(df_clean, verbose=True):
    """
    함수/기능 설명:
    (문제 2) 이상치가 제거된 데이터를 받아 region·category별 총매출(total), 평균(mean),
    건수(count)를 named aggregation으로 계산하고, 총매출 기준 내림차순으로 정렬합니다.
    - verbose=False 시 결과 출력을 억제하여 timeit 측정 시 재사용 가능합니다.
    """
    try:
        if verbose:
            print("\n===== 문제 2: Pandas groupby named aggregation =====")

        agg_result = df_clean.groupby(['region', 'category']).agg(
            total=('amount', 'sum'),
            mean=('amount', 'mean'),
            count=('amount', 'count')
        ).reset_index()
        agg_result = agg_result.sort_values(by='total', ascending=False)

        if verbose:
            print(agg_result.head(10))
        return agg_result

    except KeyError as e:
        print(f"오류: 필요한 컬럼이 없습니다. {e}")
    except Exception as e:
        print(f"groupby 집계 중 예기치 않은 오류가 발생했습니다: {e}")


def process_polars_lazy(file_path, lower_bound, upper_bound, verbose=True):
    """
    함수/기능 설명:
    (문제 3) Polars Lazy API를 사용하여 2번 문제와 동일한 집계를 수행합니다.
    scan_csv(Lazy 로드) → filter → group_by → agg → sort → collect 체인으로 구성됩니다.
    - schema_overrides로 amount 컬럼 타입을 명시하여 파싱 성능을 최적화합니다.
    - verbose=False 시 결과 출력을 억제하여 timeit 측정 시 재사용 가능합니다.
    """
    try:
        if verbose:
            print("\n===== 문제 3: Polars Lazy API 동일 집계 =====")

        result = (
            pl.scan_csv(file_path, schema_overrides={'amount': pl.Float64})
            .filter(pl.col('amount').is_between(lower_bound, upper_bound))
            .group_by(['region', 'category'])
            .agg([
                pl.col('amount').sum().alias('total'),
                pl.col('amount').mean().alias('mean'),
                pl.len().alias('count')
            ])
            .sort('total', descending=True)
            .collect()
        )

        if verbose:
            print(result.head(10))
        return result

    except Exception as e:
        print(f"Polars 실행 중 예기치 않은 오류가 발생했습니다: {e}")


def process_duckdb(file_path, lower_bound, upper_bound, verbose=True):
    """
    함수/기능 설명:
    (문제 4-1) DuckDB SQL로 CSV 파일에 직접 쿼리하여 동일한 집계를 수행합니다.
    - 데이터 로딩 없이 파일 경로를 FROM 절에 직접 지정합니다.
    - verbose=False 시 결과 출력을 억제하여 timeit 측정 시 재사용 가능합니다.
    """
    try:
        if verbose:
            print("\n===== 문제 4-1: DuckDB SQL 동일 집계 =====")

        query = f"""
            SELECT
                region,
                category,
                SUM(amount)   AS total,
                AVG(amount)   AS mean,
                COUNT(amount) AS count
            FROM '{file_path}'
            WHERE amount BETWEEN {lower_bound} AND {upper_bound}
            GROUP BY region, category
            ORDER BY total DESC
        """
        result = duckdb.sql(query).df()

        if verbose:
            print(result.head(10))
        return result

    except Exception as e:
        print(f"DuckDB 실행 중 예기치 않은 오류가 발생했습니다: {e}")


import tracemalloc

def compare_performance(file_path, df_clean, lower_bound, upper_bound):
    """
    함수/기능 설명:
    (문제 4-2) timeit으로 Pandas·Polars·DuckDB 세 도구의 집계 실행 시간을 공정하게 비교합니다.
    - 세 도구 모두 동일한 반복 횟수(iterations)로 측정하여 공정성을 보장합니다.
    - 기존에 정의된 함수를 verbose=False로 재사용하여 중복 코드를 제거합니다.
    - 메모리 사용량 및 가장 빠른 시간 등 다양한 지표를 추가로 비교합니다.
    """
    print("\n===== 문제 4-2: 세 도구 성능 비교 (다양한 지표) =====")
    iterations = 5  # 감점 방지: 세 도구 모두 동일한 반복 횟수 적용
    print(f"공정한 비교를 위해 모든 도구를 각각 {iterations}번 반복 후 측정합니다.\n")

    try:
        def measure_performance(func, name):
            # 메모리 측정 시작
            tracemalloc.start()
            
            # 실행 시간 측정 (최소 시간 확인을 위해 repeat 사용)
            times = timeit.repeat(func, number=1, repeat=iterations)
            avg_time = sum(times) / iterations
            min_time = min(times)
            
            # 메모리 사용량 측정
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"[{name}]")
            print(f"  - 평균 실행 시간: {avg_time:.4f}초")
            print(f"  - 가장 빠른 시간: {min_time:.4f}초")
            print(f"  - 최대 메모리 사용: {peak / 10**6:.2f} MB")
            print("-" * 30)
            
            return avg_time

        def run_pandas_full():
            # Pandas의 공정한 비교를 위해 파일 로드 -> 필터링 -> 집계 전체 과정을 측정합니다.
            df_tmp = pd.read_csv(file_path)
            mask_tmp = df_tmp['amount'].between(lower_bound, upper_bound)
            df_filtered = df_tmp[mask_tmp]
            process_groupby_aggregation(df_filtered, verbose=False)

        pandas_time = measure_performance(
            run_pandas_full,
            "Pandas (파일 로드 포함)"
        )
        
        polars_time = measure_performance(
            lambda: process_polars_lazy(file_path, lower_bound, upper_bound, verbose=False),
            "Polars"
        )
        
        duckdb_time = measure_performance(
            lambda: process_duckdb(file_path, lower_bound, upper_bound, verbose=False),
            "DuckDB"
        )

        fastest = min([("Pandas", pandas_time), ("Polars", polars_time), ("DuckDB", duckdb_time)], key=lambda x: x[1])
        print(f"🏆 가장 빠른 도구: {fastest[0]} (평균 기준)")

    except Exception as e:
        print(f"성능 비교 중 예기치 않은 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    # 유지보수 및 재사용성을 위해 파일 경로를 상단 변수로 관리
    TARGET_FILE = '/Users/seoyeon/skala/pandas/sales_100k.csv'

    df_clean, lower_bound, upper_bound = process_eda_and_outliers(TARGET_FILE)

    if df_clean is not None:
        process_groupby_aggregation(df_clean)
        process_polars_lazy(TARGET_FILE, lower_bound, upper_bound)
        process_duckdb(TARGET_FILE, lower_bound, upper_bound)
        compare_performance(TARGET_FILE, df_clean, lower_bound, upper_bound)