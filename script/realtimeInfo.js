import { getCoordinates, getWeather } from './weatherAPI.js';

document.addEventListener("DOMContentLoaded", () => {
  const citySelect = document.getElementById("city-select");
  const weatherBox = document.getElementById("weather-box");

  if (!citySelect || !weatherBox) return;

  citySelect.addEventListener("change", async () => {
    const selectedOption = citySelect.options[citySelect.selectedIndex];
    const cityValue = selectedOption.value;
    const cityText = selectedOption.text;

    weatherBox.innerHTML =
      `<p class='weather-city'>${cityText}</p>` +
      `<p class='weather-loading'>로딩 중... ⏳</p>`;

    try {
      const { lat, lon } = await getCoordinates(cityValue);
      const latStr = lat.toFixed(4);
      const lonStr = lon.toFixed(4);

      const { temperature, humidity } = await getWeather(lat, lon);

      const tempHtml = (temperature !== null && temperature !== undefined)
        ? `<p class='weather-temp'>🌡️ 온도: <strong>${temperature}°C</strong></p>` +
        `<p class='weather-temp'>💧 습도: <strong>${humidity}%</strong></p>`
        : `<p class='weather-error'>날씨 데이터를 불러올 수 없습니다.</p>`;

      weatherBox.innerHTML =
        `<p class='weather-city'>${cityText}</p>` +
        `<p class='weather-coord'>📍 위도: ${latStr}</p>` +
        `<p class='weather-coord'>📍 경도: ${lonStr}</p>` +
        tempHtml;

    } catch (error) {
      weatherBox.innerHTML =
        `<p class='weather-city'>${cityText}</p>` +
        `<p class='weather-error'>${error.message}</p>`;
    }
  });

  citySelect.dispatchEvent(new Event("change"));
});
