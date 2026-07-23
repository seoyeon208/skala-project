"""
검증된 데이터를 CSV와 Parquet 두 형식으로 저장하고
읽기/쓰기 성능을 측정·비교합니다.
멀티프로세싱, 데코레이터, 예외 처리 패턴 적용
"""

import asyncio
import csv
import json
import logging
import math
import multiprocessing as mp
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from src.async_pipeline import fetch_all
from src.schemas import parse_country, parse_ip

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PARALLEL_THRESHOLD = 1_000
P = ParamSpec("P")
R = TypeVar("R")


class DataStorageError(RuntimeError):
    """CSV 또는 Parquet 파일 입출력에 실패했을 때 발생합니다."""


def run_timed(
    func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> tuple[R, float]:
    """함수를 한 번 실행하고 결과와 경과 시간을 함께 반환합니다."""
    started = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - started


def process_weather_chunk(
    chunk: list[tuple[int, str, Any, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """날씨 데이터 청크를 처리하여 정상 데이터와 오류를 분리 반환 (멀티프로세싱 용도)"""
    from src.schemas import WeatherRecord

    valid = []
    errors = []
    for row_index, t, temp, precip in chunk:
        try:
            record = WeatherRecord(
                time=t, temperature_2m=temp, precipitation_probability=precip
            )
            valid.append(record.model_dump())
        except ValidationError as e:
            errors.append(
                {
                    "source": "weather",
                    "row": row_index,
                    "time": t,
                    "error": str(e),
                }
            )
    return valid, errors


def _process_weather_rows(
    rows: list[tuple[int, str, Any, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """작은 입력은 직접 처리하고 큰 입력만 여러 프로세스로 나눕니다."""
    if not rows:
        return [], []
    if len(rows) < PARALLEL_THRESHOLD:
        return process_weather_chunk(rows)

    worker_count = min(mp.cpu_count() or 1, len(rows))
    chunk_size = math.ceil(len(rows) / worker_count)
    chunks = [
        rows[start : start + chunk_size] for start in range(0, len(rows), chunk_size)
    ]

    valid: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for chunk_valid, chunk_errors in executor.map(process_weather_chunk, chunks):
            valid.extend(chunk_valid)
            errors.extend(chunk_errors)

    if len(valid) + len(errors) != len(rows):
        raise RuntimeError("날씨 데이터 처리 중 일부 행이 누락되었습니다.")
    return valid, errors


def build_records(
    weather_data: dict[str, Any],
    country_data: dict[str, Any],
    ip_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    API 원본 데이터를 스키마로 검증한 뒤 딕셔너리 리스트로 변환합니다.
    """
    valid_weather: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    # 1. 날씨 레코드 (병렬 처리 시뮬레이션)
    hourly = weather_data.get("hourly", {})
    if isinstance(hourly, dict):
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        precips = hourly.get("precipitation_probability", [])
    else:
        times = temps = precips = None

    if not all(isinstance(values, list) for values in (times, temps, precips)):
        errors.append(
            {"source": "weather", "error": "hourly 필드는 모두 배열이어야 합니다."}
        )
    elif len({len(times), len(temps), len(precips)}) != 1:
        errors.append(
            {
                "source": "weather",
                "error": "날씨 배열 길이가 서로 다릅니다.",
                "lengths": {
                    "time": len(times),
                    "temperature_2m": len(temps),
                    "precipitation_probability": len(precips),
                },
            }
        )
    else:
        raw_weather_rows = [
            (index, time_value, temperature, precipitation)
            for index, (time_value, temperature, precipitation) in enumerate(
                zip(times, temps, precips, strict=True)
            )
        ]
        valid_weather, weather_errors = _process_weather_rows(raw_weather_rows)
        errors.extend(weather_errors)

    # 국가 레코드
    try:
        country_info = parse_country(country_data)
        dicts_country = [country_info.model_dump()]
    except (KeyError, TypeError, ValueError, ValidationError) as e:
        logger.error(f"국가 데이터 파싱 에러: {e}")
        errors.append({"source": "country", "error": str(e)})
        dicts_country = []

    # IP 레코드
    try:
        ip_info = parse_ip(ip_data)
        dicts_ip = [ip_info.model_dump()]
    except (KeyError, TypeError, ValueError, ValidationError) as e:
        logger.error(f"IP 데이터 파싱 에러: {e}")
        errors.append({"source": "ip", "error": str(e)})
        dicts_ip = []

    error_path = DATA_DIR / "errors.json"
    if errors:
        error_path.write_text(
            json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.warning(f"{len(errors)}건의 검증 실패 발생 -> {error_path} 저장")
    else:
        error_path.unlink(missing_ok=True)

    return valid_weather, dicts_country, dicts_ip


def measure_write_csv(records: list[dict[str, Any]], path: Path) -> None:
    """CSV 쓰기"""
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def measure_read_csv(path: Path) -> list[dict[str, Any]]:
    """CSV 읽기"""
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def measure_write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    """Parquet 쓰기"""
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def measure_read_parquet(path: Path) -> list[dict[str, Any]]:
    """Parquet 읽기"""
    table = pq.read_table(path)
    return cast(list[dict[str, Any]], table.to_pylist())


def save_and_compare(
    records: list[dict[str, Any]], label: str
) -> dict[str, float | int]:
    """CSV·Parquet 입출력 시간과 파일 크기를 측정해 반환합니다."""
    if not records:
        logger.warning(f"[{label}] 저장할 데이터가 없습니다.")
        return {}

    csv_path = DATA_DIR / f"{label}.csv"
    parquet_path = DATA_DIR / f"{label}.parquet"

    logger.info(f"\n[{label}] 파일 저장 성능 측정 (행 수: {len(records)})")

    try:
        _, csv_write_seconds = run_timed(measure_write_csv, records, csv_path)
        _, parquet_write_seconds = run_timed(
            measure_write_parquet, records, parquet_path
        )
        _, csv_read_seconds = run_timed(measure_read_csv, csv_path)
        _, parquet_read_seconds = run_timed(measure_read_parquet, parquet_path)
    except Exception as e:
        raise DataStorageError(f"[{label}] 파일 입출력 에러") from e

    metrics: dict[str, float | int] = {
        "row_count": len(records),
        "csv_write_seconds": csv_write_seconds,
        "parquet_write_seconds": parquet_write_seconds,
        "csv_read_seconds": csv_read_seconds,
        "parquet_read_seconds": parquet_read_seconds,
        "csv_size_bytes": csv_path.stat().st_size,
        "parquet_size_bytes": parquet_path.stat().st_size,
    }
    logger.info(f"[{label}] 비교 결과: {metrics}")
    return metrics


if __name__ == "__main__":  # pragma: no cover
    weather_data, country_data, ip_data = asyncio.run(fetch_all())
    dicts_weather, dicts_country, dicts_ip = build_records(
        weather_data, country_data, ip_data
    )

    save_and_compare(dicts_weather, "weather")
    save_and_compare(dicts_country, "country")
    save_and_compare(dicts_ip, "ip_info")
