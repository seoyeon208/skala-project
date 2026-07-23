import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.async_pipeline import fetch, fetch_all


@pytest.mark.asyncio
async def test_fetch_success():
    """정상적인 API 호출 성공 테스트"""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.json.return_value = {"key": "value"}
    mock_response.raise_for_status.return_value = None
    mock_client.get.return_value = mock_response

    result = await fetch(mock_client, "http://test.url")
    assert result == {"key": "value"}
    mock_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_rejects_non_object_json():
    """API 응답의 최상위 JSON이 객체가 아니면 실패해야 합니다."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.json.return_value = ["unexpected", "array"]
    mock_response.raise_for_status.return_value = None
    mock_client.get.return_value = mock_response

    with pytest.raises(ValueError, match="최상위 값이 객체가 아닙니다"):
        await fetch(mock_client, "http://test.url")

    assert mock_client.get.call_count == 1


@pytest.mark.asyncio
async def test_fetch_http_error():
    """일시적인 네트워크 에러는 세 번까지 재시도해야 합니다."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.side_effect = httpx.ConnectError("연결 실패")

    with patch("src.async_pipeline.asyncio.sleep", new_callable=AsyncMock) as sleep:
        with pytest.raises(httpx.ConnectError):
            await fetch(mock_client, "http://test.url")

    assert mock_client.get.call_count == 3
    assert [call.args[0] for call in sleep.await_args_list] == [1.0, 2.0]


@pytest.mark.asyncio
async def test_fetch_does_not_retry_non_retryable_4xx():
    """잘못된 요청처럼 재시도로 해결되지 않는 4xx는 즉시 실패해야 합니다."""
    request = httpx.Request("GET", "http://test.url")
    response = httpx.Response(404, request=request)
    error = httpx.HTTPStatusError("Not Found", request=request, response=response)
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.side_effect = error

    with patch("src.async_pipeline.asyncio.sleep", new_callable=AsyncMock) as sleep:
        with pytest.raises(httpx.HTTPStatusError):
            await fetch(mock_client, "http://test.url")

    assert mock_client.get.call_count == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
@patch("src.async_pipeline.WEATHER_URL", "http://weather")
@patch("src.async_pipeline.COUNTRY_URL", "http://country")
@patch("src.async_pipeline.IP_URL", "http://ip")
@patch("src.async_pipeline.fetch")
async def test_fetch_all_success(mock_fetch):
    """fetch_all 3개 API 동시 수집 성공 테스트"""
    mock_fetch.side_effect = [
        {"weather": "data"},
        {"country": "data"},
        {"ip": "data"},
    ]

    w, c, i = await fetch_all()
    assert w == {"weather": "data"}
    assert c == {"country": "data"}
    assert i == {"ip": "data"}
    assert mock_fetch.call_count == 3


@pytest.mark.asyncio
@patch("src.async_pipeline.WEATHER_URL", "http://weather")
@patch("src.async_pipeline.COUNTRY_URL", "http://country")
@patch("src.async_pipeline.IP_URL", "http://ip")
async def test_fetch_all_starts_requests_concurrently():
    """세 요청이 모두 시작되기 전에 어느 하나도 완료되지 않아야 합니다."""
    all_started = asyncio.Event()
    started = 0

    async def synchronized_fetch(client, url):
        nonlocal started
        started += 1
        if started == 3:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=1)
        return {"url": url}

    with patch("src.async_pipeline.fetch", side_effect=synchronized_fetch):
        weather, country, ip_info = await fetch_all()

    assert started == 3
    assert weather == {"url": "http://weather"}
    assert country == {"url": "http://country"}
    assert ip_info == {"url": "http://ip"}


@pytest.mark.asyncio
@patch("src.async_pipeline.WEATHER_URL", "http://weather")
@patch("src.async_pipeline.COUNTRY_URL", "http://country")
@patch("src.async_pipeline.IP_URL", "http://ip")
@patch("src.async_pipeline.fetch")
async def test_fetch_all_failure(mock_fetch):
    """fetch_all 중 일부 API 실패 시 런타임 에러 발생 테스트"""
    # 첫 번째는 성공, 두 번째는 에러
    mock_fetch.side_effect = [
        {"weather": "data"},
        Exception("API Failed"),
        {"ip": "data"},
    ]

    with pytest.raises(RuntimeError) as exc_info:
        await fetch_all()

    assert "API 수집 실패" in str(exc_info.value)


@pytest.mark.asyncio
async def test_fetch_general_error():
    """프로그래밍 오류처럼 일시적이지 않은 예외는 재시도하지 않습니다."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.side_effect = ValueError("Unknown Error")

    with patch("src.async_pipeline.asyncio.sleep", new_callable=AsyncMock) as sleep:
        with pytest.raises(ValueError, match="Unknown Error"):
            await fetch(mock_client, "http://test.url")

    assert mock_client.get.call_count == 1
    sleep.assert_not_awaited()
