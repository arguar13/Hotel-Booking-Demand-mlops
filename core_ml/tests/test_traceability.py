from pathlib import Path

from src.traceability import get_container_image_tag, get_dvc_data_hash, get_git_commit_hash


def test_get_git_commit_hash_prefers_ci_env_var(monkeypatch) -> None:
    monkeypatch.setenv("CI_COMMIT_SHA", "abc123")
    assert get_git_commit_hash() == "abc123"


def test_get_git_commit_hash_falls_back_to_git(monkeypatch) -> None:
    monkeypatch.delenv("CI_COMMIT_SHA", raising=False)
    commit_hash = get_git_commit_hash()
    assert commit_hash != "unknown"
    assert len(commit_hash) == 40


def test_get_container_image_tag_defaults_to_local_dev(monkeypatch) -> None:
    monkeypatch.delenv("CI_COMMIT_SHA", raising=False)
    monkeypatch.delenv("IMAGE_TAG", raising=False)
    assert get_container_image_tag() == "local-dev"


def test_get_container_image_tag_reads_ci_commit_sha(monkeypatch) -> None:
    monkeypatch.setenv("CI_COMMIT_SHA", "deadbeef")
    assert get_container_image_tag() == "deadbeef"


def test_get_dvc_data_hash_reads_dvc_add_sidecar(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.csv.dvc").write_text("outs:\n- md5: deadbeefcafe\n  path: data.csv\n")

    assert get_dvc_data_hash("data.csv") == "deadbeefcafe"


def test_get_dvc_data_hash_reads_pipeline_lock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dvc.lock").write_text(
        "schema: '2.0'\n"
        "stages:\n"
        "  clean_data:\n"
        "    outs:\n"
        "    - path: data/processed.csv\n"
        "      md5: feedface1234\n"
    )

    assert get_dvc_data_hash("data/processed.csv") == "feedface1234"


def test_get_dvc_data_hash_returns_unknown_when_untracked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert get_dvc_data_hash("nonexistent.csv") == "unknown"
