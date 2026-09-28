import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# core_ml/config/config.yaml, resolved from this file rather than from the
# current working directory, so `python -m src.train` works whether it is
# launched from core_ml/, from the repository root, or from a container.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Loads configuration from a YAML file.

    Args:
        config_path: Path to the configuration file. Defaults to core_ml's own
            config/config.yaml.

    Returns:
        Dict[str, Any]: Dictionary containing configuration parameters.
    """
    try:
        with open(config_path, encoding="utf-8") as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        raise
