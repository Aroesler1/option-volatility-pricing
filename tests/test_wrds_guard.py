"""Tests for the WRDS connection guard.

WRDS is behind Duo. A script that reconnects, or retries after a refusal, turns
one declined push into a burst of them, and that is what locks an account out.
These tests hold the guard to three promises: nothing connects without an
explicit per-run acknowledgement, one attempt per process whether or not it
succeeded, and a refusal is never retried.

No test opens a socket. `psycopg2.connect` is replaced throughout, and the
counter that records the attempt is reset between tests, so a failure here can
never become a real connection attempt.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fetch_wrds_features as fwf  # noqa: E402
from fetch_wrds_features import DUO_ENV, WRDSGuardError, connect  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_guard(monkeypatch):
    """Reset the attempt counter and make a real connection impossible."""
    monkeypatch.setattr(fwf, "_connection_attempts", 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("a test tried to open a real WRDS connection")

    monkeypatch.setattr(fwf.psycopg2, "connect", forbidden)
    monkeypatch.delenv(DUO_ENV, raising=False)
    monkeypatch.setenv("WRDS_USERNAME", "someone")


class _Boom(Exception):
    pass


def _stub(monkeypatch, calls, exc=None):
    def fake(*args, **kwargs):
        calls.append(kwargs)
        if exc is not None:
            raise exc
        return object()

    monkeypatch.setattr(fwf.psycopg2, "connect", fake)


def test_nothing_connects_without_the_duo_acknowledgement(monkeypatch):
    calls = []
    _stub(monkeypatch, calls)
    with pytest.raises(WRDSGuardError) as err:
        connect()
    assert DUO_ENV in str(err.value)
    assert calls == []


def test_the_acknowledgement_must_be_exactly_one(monkeypatch):
    calls = []
    _stub(monkeypatch, calls)
    for value in ("0", "true", "yes", "", "1 "):
        monkeypatch.setenv(DUO_ENV, value)
        with pytest.raises(WRDSGuardError):
            connect()
    assert calls == []


def test_a_missing_username_is_refused_before_any_socket_is_opened(monkeypatch):
    calls = []
    _stub(monkeypatch, calls)
    monkeypatch.setenv(DUO_ENV, "1")
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    monkeypatch.delenv("PGUSER", raising=False)
    with pytest.raises(WRDSGuardError, match="WRDS_USERNAME"):
        connect()
    assert calls == []


def test_one_successful_connection_per_invocation(monkeypatch):
    calls = []
    _stub(monkeypatch, calls)
    monkeypatch.setenv(DUO_ENV, "1")
    assert connect() is not None
    with pytest.raises(WRDSGuardError, match="already attempted"):
        connect()
    assert len(calls) == 1


def test_a_failed_attempt_still_consumes_the_invocation(monkeypatch):
    """The counter increments BEFORE the attempt, so a refusal cannot be retried."""
    calls = []
    _stub(monkeypatch, calls, exc=fwf.psycopg2.OperationalError("auth failed"))
    monkeypatch.setenv(DUO_ENV, "1")
    with pytest.raises(WRDSGuardError, match="NOT be retried"):
        connect()
    with pytest.raises(WRDSGuardError, match="already attempted"):
        connect()
    assert len(calls) == 1


def test_an_authentication_failure_is_reraised_not_swallowed(monkeypatch):
    calls = []
    original = fwf.psycopg2.OperationalError("password authentication failed")
    _stub(monkeypatch, calls, exc=original)
    monkeypatch.setenv(DUO_ENV, "1")
    with pytest.raises(WRDSGuardError) as err:
        connect()
    assert err.value.__cause__ is original
    assert "password authentication failed" in str(err.value)


def test_an_explicit_username_argument_still_needs_the_acknowledgement(monkeypatch):
    calls = []
    _stub(monkeypatch, calls)
    with pytest.raises(WRDSGuardError):
        connect(username="someone")
    assert calls == []


def test_the_option_chain_fetcher_uses_the_same_guard():
    import fetch_option_chain

    assert fetch_option_chain.connect is connect


def test_no_module_opens_a_connection_at_import_time():
    """Importing a fetcher must never contact WRDS, so tests can import freely.

    Reloading the modules to prove it would reset the very counter being
    checked, so this reads the counter that the whole suite has been importing
    against: if any import had connected, it would not be zero here.
    """
    import fetch_option_chain  # noqa: F401

    assert fwf._connection_attempts == 0
