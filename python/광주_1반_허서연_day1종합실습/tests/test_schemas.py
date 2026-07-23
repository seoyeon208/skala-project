"""
pytest 테스트 모음
스키마 검증 로직이 올바르게 동작하는지 확인합니다.
"""

import pytest
from pydantic import ValidationError

from src.schemas import (
    CountryInfo,
    IpInfo,
    WeatherRecord,
    parse_country,
    parse_ip,
    parse_weather,
)

# ── WeatherRecord 테스트 ──────────────────────────────────────────────────────


def test_weather_record_valid():
    """정상 데이터는 WeatherRecord 생성에 성공해야 합니다."""
    record = WeatherRecord(
        time="2024-07-01T00:00",
        temperature_2m=25.3,
        precipitation_probability=40,
    )
    assert record.temperature_2m == 25.3
    assert record.precipitation_probability == 40


def test_weather_record_temperature_out_of_range():
    """기온이 범위를 벗어나면 ValidationError를 발생시켜야 합니다."""
    with pytest.raises(ValidationError):
        WeatherRecord(
            time="2024-07-01T00:00",
            temperature_2m=100.0,  # 60°C 초과
            precipitation_probability=10,
        )


def test_weather_record_rejects_wrong_numeric_type():
    """숫자 문자열을 자동 변환하지 않고 타입 오류로 처리해야 합니다."""
    with pytest.raises(ValidationError):
        WeatherRecord(
            time="2024-07-01T00:00",
            temperature_2m="25.3",  # type: ignore[arg-type]
            precipitation_probability=10,
        )


def test_weather_record_precip_out_of_range():
    """강수확률이 100을 초과하면 ValidationError를 발생시켜야 합니다."""
    with pytest.raises(ValidationError):
        WeatherRecord(
            time="2024-07-01T00:00",
            temperature_2m=20.0,
            precipitation_probability=150,  # 100% 초과
        )


def test_weather_record_none_precip_rejected():
    """결측값은 강수확률 0%와 의미가 다르므로 검증에 실패해야 합니다."""
    with pytest.raises(ValidationError):
        WeatherRecord(
            time="2024-07-01T00:00",
            temperature_2m=20.0,
            precipitation_probability=None,  # type: ignore[arg-type]
        )


def test_weather_record_invalid_time():
    """시간 필드가 ISO 8601 형식이 아니면 검증에 실패해야 합니다."""
    with pytest.raises(ValidationError):
        WeatherRecord(
            time="not-a-datetime",
            temperature_2m=20.0,
            precipitation_probability=10,
        )


def test_parse_weather_returns_list():
    """parse_weather는 WeatherRecord 리스트를 반환해야 합니다."""
    raw = {
        "hourly": {
            "time": ["2024-07-01T00:00", "2024-07-01T01:00"],
            "temperature_2m": [22.0, 21.5],
            "precipitation_probability": [30, 20],
        }
    }
    records = parse_weather(raw)
    assert len(records) == 2
    assert all(isinstance(r, WeatherRecord) for r in records)


def test_parse_weather_rejects_mismatched_array_lengths():
    """배열 길이가 다를 때 zip으로 데이터를 조용히 버리면 안 됩니다."""
    raw = {
        "hourly": {
            "time": ["2024-07-01T00:00", "2024-07-01T01:00"],
            "temperature_2m": [22.0],
            "precipitation_probability": [30, 20],
        }
    }

    with pytest.raises(ValueError, match="길이"):
        parse_weather(raw)


# ── CountryInfo 테스트 ───────────────────────────────────────────────────────


def test_country_info_valid():
    """정상 데이터는 CountryInfo 생성에 성공해야 합니다."""
    info = CountryInfo(
        cca3="KOR",
        official_name="대한민국",
        capital="서울",
        region="Asia",
        population=51_000_000,
    )
    assert info.cca3 == "KOR"


def test_country_info_invalid_cca3():
    """cca3가 3글자가 아니면 ValidationError를 발생시켜야 합니다."""
    with pytest.raises(ValidationError):
        CountryInfo(
            cca3="KR",  # 2글자
            official_name="대한민국",
            capital="서울",
            region="Asia",
            population=51_000_000,
        )


def test_country_info_rejects_non_alpha_cca3():
    """국가 코드는 영문 대문자 세 글자여야 합니다."""
    with pytest.raises(ValidationError):
        CountryInfo(
            cca3="12x",
            official_name="대한민국",
            capital="서울",
            region="Asia",
            population=51_000_000,
        )


def test_country_info_population_not_positive():
    """인구가 0 이하면 ValidationError를 발생시켜야 합니다."""
    with pytest.raises(ValidationError):
        CountryInfo(
            cca3="KOR",
            official_name="대한민국",
            capital="서울",
            region="Asia",
            population=0,  # gt=0 위반
        )


def test_parse_country():
    """parse_country가 올바른 CountryInfo를 반환해야 합니다."""
    raw = {
        "alpha3Code": "KOR",
        "name": "Korea (Republic of)",
        "capital": "Seoul",
        "region": "Asia",
        "population": 51_744_876,
    }
    info = parse_country(raw)
    assert info.cca3 == "KOR"
    assert info.capital == "Seoul"


# ── IpInfo 테스트 ────────────────────────────────────────────────────────────


def test_ip_info_valid():
    """정상 데이터는 IpInfo 생성에 성공해야 합니다."""
    info = IpInfo(
        ip="8.8.8.8",
        city="Mountain View",
        region="California",
        country="United States",
        lat=37.386,
        lon=-122.0838,
        isp="Google LLC",
    )
    assert info.ip == "8.8.8.8"


def test_ip_info_invalid_ip():
    """IPv4 형식이 아니면 ValidationError를 발생시켜야 합니다."""
    with pytest.raises(ValidationError):
        IpInfo(
            ip="999",  # 점이 3개 없음
            city="Test",
            region="Test",
            country="Test",
            lat=0.0,
            lon=0.0,
            isp="Test",
        )


def test_ip_info_rejects_out_of_range_octet():
    """점이 세 개여도 각 옥텟이 IPv4 범위를 벗어나면 실패해야 합니다."""
    with pytest.raises(ValidationError):
        IpInfo(
            ip="999.999.999.999",
            city="Test",
            region="Test",
            country="Test",
            lat=0.0,
            lon=0.0,
            isp="Test",
        )


def test_ip_info_lat_out_of_range():
    """위도가 범위를 벗어나면 ValidationError를 발생시켜야 합니다."""
    with pytest.raises(ValidationError):
        IpInfo(
            ip="8.8.8.8",
            city="Test",
            region="Test",
            country="Test",
            lat=200.0,  # 90 초과
            lon=0.0,
            isp="Test",
        )


def test_parse_ip():
    """parse_ip가 올바른 IpInfo를 반환해야 합니다."""
    raw = {
        "query": "8.8.8.8",
        "city": "Mountain View",
        "regionName": "California",
        "country": "United States",
        "lat": 37.386,
        "lon": -122.0838,
        "isp": "Google LLC",
    }
    info = parse_ip(raw)
    assert info.city == "Mountain View"
    assert info.isp == "Google LLC"


def test_parse_ip_rejects_failed_api_response():
    """ip-api가 실패 상태를 반환하면 위치 데이터로 처리하지 않습니다."""
    with pytest.raises(ValueError, match="요청 실패"):
        parse_ip({"status": "fail", "message": "invalid query"})
