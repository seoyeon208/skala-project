"""
Pydantic v2 스키마 정의 및 검증
각 API 응답에서 필요한 필드를 추출하여 타입·범위를 검증합니다.
"""

from datetime import datetime
from ipaddress import IPv4Address
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """API 응답의 잘못된 타입을 자동 변환하지 않는 공통 모델입니다."""

    model_config = ConfigDict(strict=True, str_strip_whitespace=True)


# ── 1. 날씨 데이터 스키마 ─────────────────────────────────────────────────────
class WeatherRecord(StrictModel):
    """한 시간대의 기온·강수확률 레코드"""

    time: str
    temperature_2m: float = Field(description="기온 (°C)", ge=-60, le=60)
    precipitation_probability: int = Field(description="강수확률 (%)", ge=0, le=100)

    @field_validator("time")
    @classmethod
    def must_be_iso_datetime(cls, value: str) -> str:
        """시간 값이 ISO 8601 형식인지 확인합니다."""
        try:
            datetime.fromisoformat(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"유효한 ISO 8601 시간이 아닙니다: {value!r}") from e
        return value


def parse_weather(raw: dict[str, Any]) -> list[WeatherRecord]:
    """Open-Meteo 응답에서 WeatherRecord 리스트를 만듭니다."""
    hourly = raw["hourly"]
    times = hourly["time"]
    temperatures = hourly["temperature_2m"]
    precipitation = hourly["precipitation_probability"]
    lengths = {len(times), len(temperatures), len(precipitation)}
    if len(lengths) != 1:
        raise ValueError(
            "날씨 배열 길이가 서로 다릅니다: "
            f"time={len(times)}, temperature_2m={len(temperatures)}, "
            f"precipitation_probability={len(precipitation)}"
        )

    records = []
    for time, temp, precip in zip(times, temperatures, precipitation, strict=True):
        records.append(
            WeatherRecord(
                time=time,
                temperature_2m=temp,
                precipitation_probability=precip,
            )
        )
    return records


# ── 2. 국가 데이터 스키마 ────────────────────────────────────────────────────
class CountryInfo(StrictModel):
    """한국 국가 기본 정보"""

    cca3: str = Field(description="3자리 국가 코드", pattern=r"^[A-Z]{3}$")
    official_name: str = Field(description="국가명", min_length=1)
    capital: str = Field(description="수도", min_length=1)
    region: str = Field(description="지역", min_length=1)
    population: int = Field(description="인구", gt=0)


def parse_country(raw: dict[str, Any]) -> CountryInfo:
    """
    countries.dev 응답에서 CountryInfo를 만듭니다.
    실제 응답 구조: name(str), alpha3Code, capital(str), region, population
    """
    return CountryInfo(
        cca3=raw["alpha3Code"],
        official_name=raw["name"],
        capital=raw.get("capital") or "N/A",
        region=raw["region"],
        population=raw["population"],
    )


# ── 3. IP 데이터 스키마 ──────────────────────────────────────────────────────
class IpInfo(StrictModel):
    """IP 기반 지역 정보"""

    ip: str = Field(description="조회된 IP 주소")
    city: str = Field(description="도시")
    region: str = Field(description="지역/주")
    country: str = Field(description="국가명")
    lat: float = Field(description="위도", ge=-90, le=90)
    lon: float = Field(description="경도", ge=-180, le=180)
    isp: str = Field(description="ISP 제공자")

    @field_validator("ip")
    @classmethod
    def must_be_ipv4(cls, value: str) -> str:
        try:
            IPv4Address(value)
        except ValueError as e:
            raise ValueError(f"유효한 IPv4 주소가 아닙니다: {value!r}") from e
        return value


def parse_ip(raw: dict[str, Any]) -> IpInfo:
    """ip-api 응답에서 IpInfo를 만듭니다."""
    if raw.get("status") not in (None, "success"):
        raise ValueError(f"ip-api 요청 실패: {raw.get('message', '알 수 없는 오류')}")
    return IpInfo(
        ip=raw["query"],
        city=raw["city"],
        region=raw["regionName"],
        country=raw["country"],
        lat=raw["lat"],
        lon=raw["lon"],
        isp=raw["isp"],
    )
