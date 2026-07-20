"""
검증된 데이터를 CSV와 Parquet 두 형식으로 저장하고
읽기/쓰기 성능을 측정·비교합니다.
"""

import asyncio
import time

import pandas as pd

from async_pipeline import fetch_all
from schemas import parse_country, parse_ip, parse_weather


def build_dataframes(
    weather_data: dict,
    country_data: dict,
    ip_data: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    API 원본 데이터를 스키마로 검증한 뒤 DataFrame으로 변환합니다.
    타입 오류 발생 시 ValidationError를 그대로 전파합니다.
    """
    # 날씨 DataFrame
    weather_records = parse_weather(weather_data)
    df_weather = pd.DataFrame([r.model_dump() for r in weather_records])

    # 국가 DataFrame
    country_info = parse_country(country_data)
    df_country = pd.DataFrame([country_info.model_dump()])

    # IP DataFrame
    ip_info = parse_ip(ip_data)
    df_ip = pd.DataFrame([ip_info.model_dump()])

    return df_weather, df_country, df_ip


def measure_write_csv(df: pd.DataFrame, path: str) -> float:
    """CSV 쓰기 시간(초)을 반환합니다."""
    start = time.perf_counter()
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return time.perf_counter() - start


def measure_read_csv(path: str) -> tuple[pd.DataFrame, float]:
    """CSV 읽기 시간(초)과 DataFrame을 반환합니다."""
    start = time.perf_counter()
    df = pd.read_csv(path)
    elapsed = time.perf_counter() - start
    return df, elapsed


def measure_write_parquet(df: pd.DataFrame, path: str) -> float:
    """Parquet 쓰기 시간(초)을 반환합니다."""
    start = time.perf_counter()
    df.to_parquet(path, index=False)
    return time.perf_counter() - start


def measure_read_parquet(path: str) -> tuple[pd.DataFrame, float]:
    """Parquet 읽기 시간(초)과 DataFrame을 반환합니다."""
    start = time.perf_counter()
    df = pd.read_parquet(path)
    elapsed = time.perf_counter() - start
    return df, elapsed


def save_and_compare(df: pd.DataFrame, label: str) -> None:
    """하나의 DataFrame을 CSV·Parquet로 저장하고 성능을 비교 출력합니다."""
    csv_path = f"{label}.csv"
    parquet_path = f"{label}.parquet"

    # ── 쓰기 ─────────────────────────────────────────────
    t_write_csv = measure_write_csv(df, csv_path)
    t_write_parquet = measure_write_parquet(df, parquet_path)

    # ── 읽기 ─────────────────────────────────────────────
    _, t_read_csv = measure_read_csv(csv_path)
    _, t_read_parquet = measure_read_parquet(parquet_path)

    # ── 결과 출력 ────────────────────────────────────────
    print(f"\n[{label}] 성능 비교 (행 수: {len(df)})")
    print(f"  → 파일 저장 완료 ({csv_path}, {parquet_path})")
    print(f"  쓰기: CSV {t_write_csv:.4f}s | Parquet {t_write_parquet:.4f}s")
    print(f"  읽기: CSV {t_read_csv:.4f}s | Parquet {t_read_parquet:.4f}s")


if __name__ == "__main__":
    weather_data, country_data, ip_data = asyncio.run(fetch_all())

    df_weather, df_country, df_ip = build_dataframes(weather_data, country_data, ip_data)
    
    save_and_compare(df_weather, "weather")
    save_and_compare(df_country, "country")
    save_and_compare(df_ip, "ip_info")
