from pathlib import Path

import pytest

from src.config_loader import load_config


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("data:\n  raw_data_path: 'data/raw.csv'\nmodel:\n  random_state: 42\n")

    config = load_config(str(config_file))

    assert config["data"]["raw_data_path"] == "data/raw.csv"
    assert config["model"]["random_state"] == 42


def test_load_config_raises_on_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent/config.yaml")
