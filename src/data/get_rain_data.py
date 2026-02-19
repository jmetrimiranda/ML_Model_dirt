"""
Obtém dados de precipitação horária da API Open-Meteo para a região da Praia de Ubu/Além.

Salva em data/raw/chuva_2025.csv com colunas: datetime, precipitacao_mm

API: https://open-meteo.com/ (gratuita, sem autenticação)
"""

from pathlib import Path

import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

RAW_DIR = Path("data/raw")

# Coordenadas do ponto do filtro (Praia de Ubu/Além)
PRAIA_LAT = -20.7956661
PRAIA_LON = -40.5816175


def fetch_rain_data(
    lat: float = PRAIA_LAT,
    lon: float = PRAIA_LON,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """Busca precipitação horária da Open-Meteo Archive API."""

    # Setup com cache (evita requisições repetidas)
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["precipitation", "rain"],
        "timezone": "America/Sao_Paulo",
    }

    responses = om.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )
    response = responses[0]

    print(f"[CHUVA] Coordenadas: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"[CHUVA] Timezone: {response.Timezone()}")

    hourly = response.Hourly()
    hourly_precipitation = hourly.Variables(0).ValuesAsNumpy()
    hourly_rain = hourly.Variables(1).ValuesAsNumpy()

    df = pd.DataFrame({
        "datetime": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
        "precipitacao_mm": hourly_precipitation,
        "chuva_mm": hourly_rain,
    })

    # Converter para timezone local e remover tz info para compatibilidade
    df["datetime"] = df["datetime"].dt.tz_convert("America/Sao_Paulo").dt.tz_localize(None)

    print(f"[CHUVA] Shape: {df.shape}")
    print(f"[CHUVA] Range: {df['datetime'].min()} a {df['datetime'].max()}")
    print(f"[CHUVA] Total precipitação: {df['precipitacao_mm'].sum():.1f} mm")
    print(f"[CHUVA] Dias com chuva (>0.1mm): {(df.groupby(df['datetime'].dt.date)['precipitacao_mm'].sum() > 0.1).sum()}")

    return df


def main():
    print("=" * 60)
    print("Download de Dados de Chuva — Open-Meteo API")
    print("=" * 60)

    df = fetch_rain_data()

    out_path = RAW_DIR / "chuva_2025.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[OK] Salvo em: {out_path}")

    return df


if __name__ == "__main__":
    main()