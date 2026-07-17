export async function getCoordinates(city) {
  const geocodingUrl = `https://geocoding-api.open-meteo.com/v1/search?name=${city}&count=1&language=en&format=json`;

  try {
    const response = await fetch(geocodingUrl);
    const geoData = await response.json();

    return {
      lat: geoData.results[0].latitude,
      lon: geoData.results[0].longitude
    };
  } catch (error) {
    throw new Error("도시 정보를 찾을 수 없습니다.");
  }
}

export async function getWeather(lat, lon) {
  const weatherUrl = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m`;

  try {
    const response = await fetch(weatherUrl);
    const weatherData = await response.json();

    return {
      temperature: weatherData.current.temperature_2m,
      humidity: weatherData.current.relative_humidity_2m
    };
  } catch (error) {
    throw new Error("날씨 정보를 불러올 수 없습니다.");
  }
}
