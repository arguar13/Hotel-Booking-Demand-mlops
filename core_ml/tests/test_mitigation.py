"""mitigation.py against a fake Postgres connection - no real database needed.

The fake reproduces exactly the three query shapes mitigation.py issues
(CREATE TABLE, INSERT, SELECT ... WHERE triggered_at >= %s) against an
in-memory list, rather than mocking return values call-by-call: that keeps
these tests honest about *what* mitigation.py is asking the database, not
just about the sequence of calls it happens to make today.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.monitoring import mitigation


class _FakeCursor:
    def __init__(self, events: list[tuple]) -> None:
        self._events = events
        self._result: list[tuple] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, sql: str, params: tuple | None = None) -> None:
        statement = sql.strip()
        if statement.startswith("CREATE TABLE") or "CREATE INDEX" in statement:
            return
        if statement.startswith("INSERT INTO"):
            assert params is not None
            self._events.append(params)
            return
        if statement.startswith("SELECT"):
            assert params is not None
            (since,) = params
            rows = [
                (triggered_at, outcome) for _, triggered_at, _, _, _, _, outcome in self._events
            ]
            self._result = sorted(
                (row for row in rows if row[0] >= since), key=lambda r: r[0], reverse=True
            )
            return
        raise AssertionError(f"unexpected SQL in fake cursor: {sql}")

    def fetchall(self) -> list[tuple]:
        return self._result


class _FakeConnection:
    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.events)

    def commit(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeResponse:
    def __init__(self, pipeline_id: int = 4242) -> None:
        self._pipeline_id = pipeline_id

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"id": self._pipeline_id}


def _config(**overrides: object) -> dict:
    mitigation_config = {
        "enabled": True,
        "cooldown_hours": 6,
        "max_auto_retrains": 3,
        "max_auto_retrains_window_days": 7,
        "gitlab_api_url_env": "GITLAB_API_URL",
        "gitlab_project_path_env": "GITLAB_PROJECT_PATH",
        "gitlab_ref_env": "GITLAB_REF",
        "gitlab_token_env": "GITLAB_TRIGGER_TOKEN",
        "trigger_timeout_seconds": 10,
        **overrides,
    }
    return {"mitigation": mitigation_config}


@pytest.fixture(autouse=True)
def _gitlab_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITLAB_API_URL", "https://gitlab.example.com/api/v4")
    monkeypatch.setenv("GITLAB_PROJECT_PATH", "group/hotel-mlops")
    monkeypatch.setenv("GITLAB_REF", "main")
    monkeypatch.setenv("GITLAB_TRIGGER_TOKEN", "trigger-token")
    # trigger_retrain() builds `dsn or dsn_from_env()` before psycopg2.connect
    # is ever called, so dsn_from_env()'s own env guard has to be satisfied
    # even though the fake_conn fixture below intercepts the actual connect().
    monkeypatch.setenv("DB_HOST", "db.example.internal")
    monkeypatch.setenv("POSTGRES_PASSWORD", "unused-in-tests")


@pytest.fixture
def fake_conn(monkeypatch: pytest.MonkeyPatch) -> _FakeConnection:
    conn = _FakeConnection()
    monkeypatch.setattr(mitigation.psycopg2, "connect", lambda dsn: conn)
    return conn


def test_disabled_never_touches_the_database_or_gitlab(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("psycopg2.connect must not be called when mitigation is disabled")

    monkeypatch.setattr(mitigation.psycopg2, "connect", _fail)
    outcome = mitigation.trigger_retrain(
        source="batch", reason="concept drift", model_version="7", config=_config(enabled=False)
    )
    assert outcome.triggered is False
    assert "disabled" in outcome.reason.lower() or "false" in outcome.reason.lower()


def test_first_alert_triggers_and_records_the_event(
    fake_conn: _FakeConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mitigation.requests, "post", lambda *a, **k: _FakeResponse(pipeline_id=101))

    outcome = mitigation.trigger_retrain(
        source="batch", reason="concept drift ALERT", model_version="7", config=_config()
    )

    assert outcome.triggered is True
    assert outcome.pipeline_id == "101"
    assert len(fake_conn.events) == 1
    assert fake_conn.events[0][6] == "triggered"  # outcome column


def test_cooldown_blocks_a_second_trigger_within_the_window(
    fake_conn: _FakeConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mitigation.requests, "post", lambda *a, **k: _FakeResponse())
    config = _config(cooldown_hours=6)

    first = mitigation.trigger_retrain(
        source="batch", reason="r1", model_version="7", config=config
    )
    second = mitigation.trigger_retrain(
        source="stream", reason="r2", model_version="7", config=config
    )

    assert first.triggered is True
    assert second.triggered is False
    assert "cooldown" in second.reason.lower()
    # The blocked attempt must not itself count as a triggered event.
    assert len(fake_conn.events) == 1


def test_a_trigger_older_than_the_cooldown_is_allowed_again(
    fake_conn: _FakeConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mitigation.requests, "post", lambda *a, **k: _FakeResponse())
    stale = datetime.now(UTC) - timedelta(hours=7)
    fake_conn.events.append(("evt-1", stale, "batch", "old alert", "6", "99", "triggered"))

    outcome = mitigation.trigger_retrain(
        source="batch", reason="new alert", model_version="7", config=_config(cooldown_hours=6)
    )

    assert outcome.triggered is True
    assert len(fake_conn.events) == 2


def test_ceiling_refuses_further_automatic_retrains(
    fake_conn: _FakeConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mitigation.requests, "post", lambda *a, **k: _FakeResponse())
    now = datetime.now(UTC)
    # Three prior triggers, each outside the (short, 0h) cooldown but inside
    # the 7-day ceiling window, so only the ceiling check is exercised.
    for i in range(3):
        fake_conn.events.append(
            (f"evt-{i}", now - timedelta(days=i + 1), "batch", "old", "6", str(i), "triggered")
        )

    outcome = mitigation.trigger_retrain(
        source="batch",
        reason="yet another alert",
        model_version="7",
        config=_config(cooldown_hours=0, max_auto_retrains=3, max_auto_retrains_window_days=7),
    )

    assert outcome.triggered is False
    assert "refusing" in outcome.reason.lower()
    # The ceiling hit is itself recorded, for the next call's ceiling check.
    assert fake_conn.events[-1][6] == "ceiling_reached"


def test_missing_gitlab_config_is_caught_and_recorded_as_failed(
    fake_conn: _FakeConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITLAB_TRIGGER_TOKEN", raising=False)

    def _unexpected_post(*args: object, **kwargs: object) -> None:
        raise AssertionError("requests.post must not be called without a token")

    monkeypatch.setattr(mitigation.requests, "post", _unexpected_post)

    outcome = mitigation.trigger_retrain(
        source="batch", reason="concept drift", model_version="7", config=_config()
    )

    assert outcome.triggered is False
    assert fake_conn.events[-1][6] == "failed"


def test_gitlab_http_failure_is_caught_and_recorded_as_failed(
    fake_conn: _FakeConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*args: object, **kwargs: object) -> None:
        raise mitigation.requests.exceptions.ConnectionError("broker unreachable")

    monkeypatch.setattr(mitigation.requests, "post", _raise)

    outcome = mitigation.trigger_retrain(
        source="stream", reason="prediction drift", model_version="7", config=_config()
    )

    assert outcome.triggered is False
    assert "failed" in outcome.reason.lower() or "unreachable" in outcome.reason.lower()
    assert fake_conn.events[-1][6] == "failed"
