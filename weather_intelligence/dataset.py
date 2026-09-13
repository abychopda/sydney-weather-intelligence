"""
Raw data acquisition.

Downloads historical weather observations for Sydney from the Open-Meteo
API and stores them in ``data/raw``. The raw data is deliberately excluded
from version control, so this module is the reproducible route to
regenerating it:

    uv run python -m weather_intelligence.dataset
"""

from pathlib import Path

from loguru import logger
from mlkit_aby.helpers import weather
import typer

from weather_intelligence.config import (
    RAW_DATA_END,
    RAW_DATA_START,
    RAW_WEATHER_FILE,
)

app = typer.Typer()


@app.command()
def main(
    output_path: Path = RAW_WEATHER_FILE,
    start_date: str = RAW_DATA_START,
    end_date: str = RAW_DATA_END,
    overwrite: bool = False,
):
    """
    Download hourly weather observations and save them to ``data/raw``.

    The download is skipped if the output file already exists, unless
    ``--overwrite`` is passed.
    """
    if output_path.exists() and not overwrite:
        logger.info(
            f"{output_path.name} already exists; skipping download. "
            "Pass --overwrite to download it again."
        )
        return

    logger.info(
        f"Downloading Sydney weather observations from {start_date} to {end_date}..."
    )

    hourly = weather.fetch_hourly_weather(
        start_date=start_date,
        end_date=end_date,
        timeout=300,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_path, index=False)

    logger.success(
        f"Saved {len(hourly):,} hourly observations to {output_path}."
    )


if __name__ == "__main__":
    app()
