import json
from pathlib import Path
from typing import Any

import pytest

from src.save_compare import (
    DataStorageError,
    _process_weather_rows,
    build_records,
    measure_read_csv,
    measure_read_parquet,
    measure_write_csv,
    measure_write_parquet,
    process_weather_chunk,
    save_and_compare,
)


def test_process_weather_chunk():
    """날씨 청크 처리 테스트 (성공 및 에러 분리)"""
    chunk = [
        (10, "2024-07-01T00:00", 25.0, 30),  # 정상
        (11, "2024-07-01T01:00", 100.0, 30),  # 기온 범위 초과 에러
    ]
    valid, errors = process_weather_chunk(chunk)

    assert len(valid) == 1
    assert valid[0]["temperature_2m"] == 25.0

    assert len(errors) == 1
    assert errors[0]["row"] == 11
    assert "validation error" in errors[0]["error"].lower()


def test_build_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """API 응답 딕셔너리 리스트 변환 및 에러 로깅 테스트"""
    # DATA_DIR을 임시 폴더로 변경하여 errors.json이 임시 폴더에 저장되도록 함
    monkeypatch.setattr("src.save_compare.DATA_DIR", tmp_path)

    weather_data = {
        "hourly": {
            "time": ["2024-07-01T00:00", "2024-07-01T01:00"],
            "temperature_2m": [25.0, 150.0],  # 두 번째는 에러
            "precipitation_probability": [30, 20],
        }
    }
    country_data = {
        "alpha3Code": "KOR",
        "name": "Korea",
        "capital": "Seoul",
        "region": "Asia",
        "population": 50000000,
    }
    ip_data = {
        "query": "8.8.8.8",
        "city": "City",
        "regionName": "Region",
        "country": "Country",
        "lat": 30.0,
        "lon": 30.0,
        "isp": "ISP",
    }

    dicts_weather, dicts_country, dicts_ip = build_records(
        weather_data, country_data, ip_data
    )

    assert len(dicts_weather) == 1
    assert len(dicts_country) == 1
    assert len(dicts_ip) == 1

    # errors.json이 생성되었는지 확인
    error_file = tmp_path / "errors.json"
    assert error_file.exists()

    with error_file.open("r", encoding="utf-8") as f:
        errors = json.load(f)
    assert len(errors) == 1
    assert errors[0]["time"] == "2024-07-01T01:00"


def test_build_records_preserves_all_72_weather_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """CPU 수로 나누어떨어지지 않아도 마지막 행까지 모두 처리해야 합니다."""

    class InlineExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, chunks):
            return map(function, chunks)

    monkeypatch.setattr("src.save_compare.DATA_DIR", tmp_path)
    monkeypatch.setattr("src.save_compare.PARALLEL_THRESHOLD", 1)
    monkeypatch.setattr("src.save_compare.mp.cpu_count", lambda: 10)
    monkeypatch.setattr("src.save_compare.ProcessPoolExecutor", InlineExecutor)

    times = [
        f"2024-07-{1 + index // 24:02d}T{index % 24:02d}:00" for index in range(72)
    ]
    weather_data = {
        "hourly": {
            "time": times,
            "temperature_2m": [20.0] * 72,
            "precipitation_probability": [10] * 72,
        }
    }
    country_data = {
        "alpha3Code": "KOR",
        "name": "Korea",
        "capital": "Seoul",
        "region": "Asia",
        "population": 50_000_000,
    }
    ip_data = {
        "status": "success",
        "query": "8.8.8.8",
        "city": "City",
        "regionName": "Region",
        "country": "Country",
        "lat": 30.0,
        "lon": 30.0,
        "isp": "ISP",
    }

    weather, _, _ = build_records(weather_data, country_data, ip_data)

    assert len(weather) == 72
    assert weather[-1]["time"] == times[-1]


def test_process_weather_rows_detects_dropped_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    """병렬 처리 결과의 행 수가 입력보다 작으면 즉시 실패해야 합니다."""

    class DroppingExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, chunks):
            return [([], [])]

    monkeypatch.setattr("src.save_compare.PARALLEL_THRESHOLD", 1)
    monkeypatch.setattr("src.save_compare.mp.cpu_count", lambda: 2)
    monkeypatch.setattr("src.save_compare.ProcessPoolExecutor", DroppingExecutor)
    rows = [
        (0, "2024-07-01T00:00", 20.0, 10),
        (1, "2024-07-01T01:00", 21.0, 20),
    ]

    with pytest.raises(RuntimeError, match="일부 행이 누락"):
        _process_weather_rows(rows)


def test_build_records_rejects_non_mapping_hourly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """hourly가 객체가 아니면 타입 오류를 errors.json에 기록해야 합니다."""
    monkeypatch.setattr("src.save_compare.DATA_DIR", tmp_path)

    weather, _, _ = build_records({"hourly": []}, {}, {})
    errors = json.loads((tmp_path / "errors.json").read_text(encoding="utf-8"))

    assert weather == []
    weather_error = next(error for error in errors if error["source"] == "weather")
    assert "모두 배열이어야" in weather_error["error"]


def test_build_records_reports_mismatched_weather_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """길이가 다른 시계열 배열은 errors.json에 명확히 기록해야 합니다."""
    monkeypatch.setattr("src.save_compare.DATA_DIR", tmp_path)
    weather_data = {
        "hourly": {
            "time": ["2024-07-01T00:00", "2024-07-01T01:00"],
            "temperature_2m": [20.0],
            "precipitation_probability": [10, 20],
        }
    }

    weather, _, _ = build_records(weather_data, {}, {})
    errors = json.loads((tmp_path / "errors.json").read_text(encoding="utf-8"))

    assert weather == []
    assert errors[0]["source"] == "weather"
    assert errors[0]["lengths"]["time"] == 2


def test_build_records_invalid_country_ip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """국가 및 IP 데이터 검증 실패 시 빈 리스트 반환 테스트"""
    monkeypatch.setattr("src.save_compare.DATA_DIR", tmp_path)

    # 필수 필드 누락 및 잘못된 값으로 ValidationError 유도
    invalid_country = {
        "alpha3Code": "K",  # 3글자가 아니므로 에러
        "name": "Korea",
        "capital": "Seoul",
        "region": "Asia",
        "population": -1,  # 0보다 작으므로 에러
    }
    invalid_ip = {
        "query": "invalid_ip",  # IP 형식이 아니므로 에러
        "city": "City",
        "regionName": "Region",
        "country": "Country",
        "lat": 30.0,
        "lon": 30.0,
        "isp": "ISP",
    }
    w, c, i = build_records({}, invalid_country, invalid_ip)

    assert len(w) == 0
    assert len(c) == 0
    assert len(i) == 0

    errors = json.loads((tmp_path / "errors.json").read_text(encoding="utf-8"))
    assert {error["source"] for error in errors} == {"country", "ip"}


def test_measure_csv_parquet(tmp_path: Path):
    """CSV 및 Parquet 파일 읽기/쓰기 통합 테스트"""
    records: list[dict[str, Any]] = [
        {"col1": "val1", "col2": 1},
        {"col1": "val2", "col2": 2},
    ]
    csv_path = tmp_path / "test.csv"
    parquet_path = tmp_path / "test.parquet"

    # ── 쓰기 테스트 ──
    measure_write_csv(records, csv_path)
    assert csv_path.exists()

    measure_write_parquet(records, parquet_path)
    assert parquet_path.exists()

    # ── 읽기 테스트 ──
    csv_read = measure_read_csv(csv_path)
    # csv 모듈은 문자열로 반환하므로 타입 변환 없이 단순 비교
    assert len(csv_read) == 2
    assert csv_read[0]["col1"] == "val1"

    parquet_read = measure_read_parquet(parquet_path)
    assert len(parquet_read) == 2
    assert parquet_read[0]["col2"] == 1


def test_save_and_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """save_and_compare 전체 흐름 테스트"""
    monkeypatch.setattr("src.save_compare.DATA_DIR", tmp_path)

    records = [{"id": 1, "value": "test"}]

    metrics = save_and_compare(records, "test_label")

    assert (tmp_path / "test_label.csv").exists()
    assert (tmp_path / "test_label.parquet").exists()
    assert metrics["row_count"] == 1
    assert metrics["csv_size_bytes"] > 0
    assert metrics["parquet_size_bytes"] > 0


def test_save_and_compare_empty():
    """데이터가 없을 때 조기 반환 테스트"""
    # 에러가 발생하지 않고 조용히 리턴되어야 함
    assert save_and_compare([], "empty_test") == {}


def test_save_and_compare_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """파일 I/O 에러 시 DataStorageError 발생 테스트"""
    monkeypatch.setattr("src.save_compare.DATA_DIR", tmp_path)

    # 유효하지 않은 경로를 주입하여 에러 유도
    def mock_write_csv(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr("src.save_compare.measure_write_csv", mock_write_csv)

    records = [{"id": 1}]
    with pytest.raises(DataStorageError) as exc_info:
        save_and_compare(records, "error_test")

    assert "파일 입출력 에러" in str(exc_info.value)
