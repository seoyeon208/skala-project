"""
비동기 API 수집 파이프라인
asyncio + httpx를 사용하여 3개 API를 동시에 호출합니다.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import httpx
import tomllib

# 프로젝트 루트 (src/ 의 상위 폴더)
_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── 설정 파일 로드 ────────────────────────────────────────────────────────────
with (_PROJECT_ROOT / "config.toml").open("rb") as config_file:
    config = tomllib.load(config_file)

WEATHER_URL = config["api"]["weather_url"]
COUNTRY_URL = config["api"]["country_url"]
IP_URL = config["api"]["ip_url"]
API_TIMEOUT = config["api"]["timeout"]


def _is_retryable(exc: Exception) -> bool:
    """일시적인 통신 장애나 서버 오류인지 판단합니다."""
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 429 or status_code >= 500
    return isinstance(exc, httpx.TransportError)


def retry(retries: int = 3, delay: float = 1.0) -> Callable:
    """일시적인 API 장애만 지수 백오프로 재시도하는 데코레이터입니다."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (httpx.TransportError, httpx.HTTPStatusError) as e:
                    logger.warning(
                        f"[{func.__name__}] 시도 {attempt}/{retries} 실패: {e}"
                    )
                    if not _is_retryable(e) or attempt == retries:
                        logger.error(f"[{func.__name__}] 최종 실패.")
                        raise
                    await asyncio.sleep(delay * (2 ** (attempt - 1)))

        return wrapper

    return decorator


@retry(retries=3, delay=1.0)
async def fetch(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    """단일 URL을 비동기로 GET 요청하여 JSON을 반환합니다."""
    try:
        response = await client.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"JSON 최상위 값이 객체가 아닙니다: {url}")
        return data
    except httpx.HTTPError as e:
        logger.error(f"HTTP 통신 에러 발생: {url} ({e})")
        raise
    finally:
        logger.info(f"요청 종료: {url}")


async def fetch_all() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    3개 API를 asyncio.gather()로 동시에 수집합니다.
    return_exceptions=True: 일부 API가 실패해도 나머지 수집을 계속합니다.
    반환값: (weather_data, country_data, ip_data)
    """
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            fetch(client, WEATHER_URL),
            fetch(client, COUNTRY_URL),
            fetch(client, IP_URL),
            return_exceptions=True,
        )

    # 각 결과에서 예외 발생 여부 확인
    labels = ["weather", "country", "ip"]
    validated: list[dict[str, Any]] = []
    for label, result in zip(labels, results):
        if isinstance(result, BaseException):
            raise RuntimeError(f"[{label}] API 수집 실패: {result}") from result
        validated.append(result)

    weather_data, country_data, ip_data = validated
    return weather_data, country_data, ip_data


if __name__ == "__main__":  # pragma: no cover
    print("=== 비동기 API 수집 시작 ===\n")

    try:
        weather, country, ip_info = asyncio.run(fetch_all())
    except Exception as e:
        logger.error(f"파이프라인 실행 중 치명적인 오류 발생: {e}")
        exit(1)

    print("[날씨] 서울 기온 첫 3개:", weather["hourly"]["temperature_2m"][:3])
    print("[날씨] 강수확률 첫 3개:", weather["hourly"]["precipitation_probability"][:3])
    print(f"[국가] 한국 공식 명칭: {country['name']} ({country['alpha3Code']})")
    print(f"[IP]  지역 정보: {ip_info['city']}, {ip_info['country']}")

    # 원본 JSON 저장 (스키마 검증·저장 단계에서 재사용)
    _DATA_DIR.mkdir(exist_ok=True)
    with (_DATA_DIR / "async_results.json").open("w", encoding="utf-8") as output_file:
        json.dump(
            {"weather": weather, "country": country, "ip": ip_info},
            output_file,
            ensure_ascii=False,
            indent=2,
        )
    print("\ndata/async_results.json 저장 완료")
