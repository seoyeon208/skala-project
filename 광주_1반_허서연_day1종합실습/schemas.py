"""
Pydantic v2 스키마 정의 및 검증
각 API 응답에서 필요한 필드를 추출하여 타입·범위를 검증합니다.
"""

from pydantic import BaseModel, Field, field_validator


# ── 1. 날씨 데이터 스키마 ─────────────────────────────────────────────────────
class WeatherRecord(BaseModel):
    """한 시간대의 기온·강수확률 레코드"""

    time: str
    temperature_2m: float = Field(description="기온 (°C)", ge=-60, le=60)
    precipitation_probability: int = Field(description="강수확률 (%)", ge=0, le=100)

    @field_validator("precipitation_probability", mode="before")
    @classmethod
    def coerce_precip(cls, v):
        """None 값은 0으로 대체합니다."""
        return 0 if v is None else v


def parse_weather(raw: dict) -> list[WeatherRecord]:
    """Open-Meteo 응답에서 WeatherRecord 리스트를 만듭니다."""
    hourly = raw["hourly"]
    records = []
    for time, temp, precip in zip(
        hourly["time"],
        hourly["temperature_2m"],
        hourly["precipitation_probability"],
    ):
        records.append(
            WeatherRecord(
                time=time,
                temperature_2m=temp,
                precipitation_probability=precip,
            )
        )
    return records


# ── 2. 국가 데이터 스키마 ────────────────────────────────────────────────────
class CountryInfo(BaseModel):
    """한국 국가 기본 정보"""

    cca3: str = Field(description="3자리 국가 코드")
    official_name: str = Field(description="국가명")
    capital: str = Field(description="수도")
    region: str = Field(description="지역")
    population: int = Field(description="인구", gt=0)

    @field_validator("cca3")
    @classmethod
    def must_be_three_chars(cls, v: str) -> str:
        if len(v) != 3:
            raise ValueError(f"cca3는 3글자여야 합니다: {v!r}")
        return v


def parse_country(raw: dict) -> CountryInfo:
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
class IpInfo(BaseModel):
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
    def must_have_dots(cls, v: str) -> str:
        if v.count(".") != 3:
            raise ValueError(f"유효한 IPv4 주소가 아닙니다: {v!r}")
        return v


def parse_ip(raw: dict) -> IpInfo:
    """ip-api 응답에서 IpInfo를 만듭니다."""
    return IpInfo(
        ip=raw["query"],
        city=raw["city"],
        region=raw["regionName"],
        country=raw["country"],
        lat=raw["lat"],
        lon=raw["lon"],
        isp=raw["isp"],
    )
