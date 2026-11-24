"""Download curated bacterial genome-scale metabolic models from BiGG."""

import logging
import re
from pathlib import Path
from urllib.parse import urlparse

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BIGG_MODEL_URL = "http://bigg.ucsd.edu/static/models/{model_id}.json"

# Clean, well-curated bacterial GEMs for the 6-week sprint
DEFAULT_MODELS: list[str] = [
    "iML1515",  # Escherichia coli K-12 MG1655
    "iYO844",  # Bacillus subtilis
    "iJN1463",  # Pseudomonas putida KT2440
    "STM_v1_0",  # Salmonella enterica
    "iAF1260",  # Escherichia coli K-12
    "iSB619",  # Staphylococcus aureus
    "iNJ661",  # Mycobacterium tuberculosis
    "iIT341",  # Helicobacter pylori
]

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_model_id(model_id: str) -> str:
    """Validate BiGG model identifier against path traversal characters."""
    if not _MODEL_ID_RE.match(model_id):
        raise ValueError(f"Invalid BiGG model_id: {model_id!r}")
    return model_id


def download_bigg_models(
    model_ids: list[str] = DEFAULT_MODELS,
    output_dir: Path = Path("data/raw/bigg"),
) -> list[Path]:
    """Download BiGG JSON models to output_dir, skipping already cached files."""
    resolved_dir = output_dir.resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for model_id in model_ids:
        safe_id = _validate_model_id(model_id)
        dest = (resolved_dir / f"{safe_id}.json").resolve()
        if not str(dest).startswith(str(resolved_dir) + "/"):
            raise ValueError(f"Destination {dest} escapes output_dir {resolved_dir}")

        if dest.exists():
            logger.info("Skipping %s (already exists at %s)", safe_id, dest)
            saved_paths.append(dest)
            continue

        url = BIGG_MODEL_URL.format(model_id=safe_id)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname != "bigg.ucsd.edu":
            raise ValueError(f"Disallowed download URL: {url}")

        logger.info("Downloading %s from %s", safe_id, url)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
        saved_paths.append(dest)

    return saved_paths


if __name__ == "__main__":
    download_bigg_models()
