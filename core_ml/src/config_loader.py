import logging
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Dictionary containing configuration parameters.
    """
    try:
        with open(config_path) as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        raise
