"""
비동기 API 수집 파이프라인
asyncio + httpx를 사용하여 3개 API를 동시에 호출합니다.
"""

import asyncio
import json

import httpx

# ── API URL 정의 ──────────────────────────────────────────────────────────────
WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=37.5665&longitude=126.9780"
    "&hourly=temperature_2m,precipitation_probability"
    "&forecast_days=3&timezone=Asia/Seoul"
)
COUNTRY_URL = "https://countries.dev/alpha/KOR"
IP_URL = "http://ip-api.com/json/8.8.8.8"


async def fetch(client: httpx.AsyncClient, url: str) -> dict:
    """단일 URL을 비동기로 GET 요청하여 JSON을 반환합니다."""
    response = await client.get(url, timeout=10.0)
    response.raise_for_status()
    return response.json()


async def fetch_all() -> tuple[dict, dict, dict]:
    """
    3개 API를 asyncio.gather()로 동시에 수집합니다.
    반환값: (weather_data, country_data, ip_data)
    """
    async with httpx.AsyncClient() as client:
        weather_data, country_data, ip_data = await asyncio.gather(
            fetch(client, WEATHER_URL),
            fetch(client, COUNTRY_URL),
            fetch(client, IP_URL),
        )
    return weather_data, country_data, ip_data


if __name__ == "__main__":
    print("=== 비동기 API 수집 시작 ===\n")

    weather, country, ip_info = asyncio.run(fetch_all())

    print("[날씨] 서울 기온 첫 3개:", weather["hourly"]["temperature_2m"][:3])
    print("[날씨] 강수확률 첫 3개:", weather["hourly"]["precipitation_probability"][:3])
    print(f"[국가] 한국 공식 명칭: {country['name']} ({country['alpha3Code']})")
    print(f"[IP]  지역 정보: {ip_info['city']}, {ip_info['country']}")

    # 원본 JSON 저장 (스키마 검증·저장 단계에서 재사용)
    with open("async_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"weather": weather, "country": country, "ip": ip_info},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("\nasync_results.json 저장 완료")
