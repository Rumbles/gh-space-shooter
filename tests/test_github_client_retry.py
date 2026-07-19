"""Tests for GitHubClient: chunked contribution fetch + transient-failure retry."""

from datetime import date, timedelta

import httpx
import pytest

from gh_space_shooter.constants import NUM_WEEKS
from gh_space_shooter.github_client import (
    DAYS_PER_WEEK,
    MAX_ATTEMPTS,
    MAX_QUERY_WEEKS,
    GitHubAPIError,
    GitHubClient,
)

API_URL = GitHubClient.GITHUB_API_URL


def _response(status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", API_URL),
    )


def _total_payload(total: int = 8565) -> dict:
    return {
        "data": {
            "user": {
                "contributionsCollection": {
                    "contributionCalendar": {"totalContributions": total}
                }
            }
        }
    }


def _days_payload(from_iso: str, to_iso: str) -> dict:
    """A day-window response echoing count=1 for every date in [from, to]."""
    start = date.fromisoformat(from_iso[:10])
    end = date.fromisoformat(to_iso[:10])
    days = []
    cursor = start
    while cursor <= end:
        days.append(
            {
                "date": cursor.isoformat(),
                "contributionCount": 1,
                "contributionLevel": "FIRST_QUARTILE",
            }
        )
        cursor += timedelta(days=1)
    return {
        "data": {
            "user": {
                "contributionsCollection": {
                    "contributionCalendar": {
                        "weeks": [{"contributionDays": days}]
                    }
                }
            }
        }
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> GitHubClient:
    """A GitHubClient with sleep patched out so retries don't wait."""
    monkeypatch.setattr(
        "gh_space_shooter.github_client.time.sleep", lambda _seconds: None
    )
    return GitHubClient(token="fake-token")


# --- End-to-end: chunked fetch + grid reconstruction ------------------------


def test_get_contribution_graph_chunks_and_builds_grid(
    client: GitHubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full year is fetched in small windows and stitched into 52x7."""
    calls: list[dict] = []

    def fake_post(*_args, json=None, **_kwargs):
        calls.append(json)
        query = json["query"]
        if "totalContributions" in query:
            return _response(200, _total_payload())
        variables = json["variables"]
        return _response(200, _days_payload(variables["from"], variables["to"]))

    monkeypatch.setattr(client.client, "post", fake_post)

    result = client.get_contribution_graph("Rumbles")

    assert result["total_contributions"] == 8565
    assert len(result["weeks"]) == NUM_WEEKS
    assert all(len(week["days"]) == DAYS_PER_WEEK for week in result["weeks"])

    day_calls = [c for c in calls if "totalContributions" not in c["query"]]
    # Should have chunked into multiple windows, not one big year query.
    assert len(day_calls) > 1
    max_span_days = MAX_QUERY_WEEKS * DAYS_PER_WEEK
    for c in day_calls:
        span = date.fromisoformat(c["variables"]["to"][:10]) - date.fromisoformat(
            c["variables"]["from"][:10]
        )
        assert span.days < max_span_days

    # Every past day was populated (count 1); future days of the current week
    # default to 0. So at least one day is populated and none exceed 1.
    counts = [day["count"] for week in result["weeks"] for day in week["days"]]
    assert any(count == 1 for count in counts)
    assert set(counts) <= {0, 1}


def test_user_not_found_raises(
    client: GitHubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing user (null) surfaces a clear error."""

    def fake_post(*_args, json=None, **_kwargs):
        return _response(200, {"data": {"user": None}})

    monkeypatch.setattr(client.client, "post", fake_post)

    with pytest.raises(GitHubAPIError, match="not found"):
        client.get_contribution_graph("nope")


# --- Retry behaviour of the low-level POST ----------------------------------


def test_retries_on_502_then_succeeds(
    client: GitHubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 502 Bad Gateway is retried and the subsequent success is returned."""
    responses = [_response(502, {}), _response(200, {"data": {"ok": True}})]
    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        response = responses[calls["n"]]
        calls["n"] += 1
        return response

    monkeypatch.setattr(client.client, "post", fake_post)

    result = client._post_graphql("query", {})

    assert calls["n"] == 2
    assert result == {"data": {"ok": True}}


def test_retries_on_transient_rate_limit_then_succeeds(
    client: GitHubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient rate-limit GraphQL error is retried, then succeeds."""
    responses = [
        _response(200, {"errors": [{"message": "API rate limit exceeded"}]}),
        _response(200, {"data": {"ok": True}}),
    ]
    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        response = responses[calls["n"]]
        calls["n"] += 1
        return response

    monkeypatch.setattr(client.client, "post", fake_post)

    result = client._post_graphql("query", {})

    assert calls["n"] == 2
    assert result == {"data": {"ok": True}}


def test_resource_limit_error_is_not_retried(
    client: GitHubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resource-limit error is deterministic, so it must fail fast."""
    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        calls["n"] += 1
        return _response(
            200,
            {"errors": [{"message": "Resource limits for this query exceeded."}]},
        )

    monkeypatch.setattr(client.client, "post", fake_post)

    with pytest.raises(GitHubAPIError, match="Resource limits"):
        client._post_graphql("query", {})

    assert calls["n"] == 1


def test_retries_on_secondary_rate_limit_403(
    client: GitHubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 403 secondary-rate-limit (with Retry-After) is retried, not raised."""
    limited = httpx.Response(
        403,
        headers={"retry-after": "1"},
        json={"message": "You have exceeded a secondary rate limit"},
        request=httpx.Request("POST", API_URL),
    )
    responses = [limited, _response(200, {"data": {"ok": True}})]
    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        response = responses[calls["n"]]
        calls["n"] += 1
        return response

    monkeypatch.setattr(client.client, "post", fake_post)

    result = client._post_graphql("query", {})

    assert calls["n"] == 2
    assert result == {"data": {"ok": True}}


def test_plain_403_is_not_retried(
    client: GitHubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine auth 403 (no rate-limit signal) fails fast."""
    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        calls["n"] += 1
        return httpx.Response(
            403,
            json={"message": "Bad credentials"},
            request=httpx.Request("POST", API_URL),
        )

    monkeypatch.setattr(client.client, "post", fake_post)

    with pytest.raises(GitHubAPIError):
        client._post_graphql("query", {})

    assert calls["n"] == 1


def test_non_transient_graphql_error_is_not_retried(
    client: GitHubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine query error fails immediately without burning retries."""
    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        calls["n"] += 1
        return _response(
            200, {"errors": [{"message": "Field 'nope' doesn't exist"}]}
        )

    monkeypatch.setattr(client.client, "post", fake_post)

    with pytest.raises(GitHubAPIError, match="nope"):
        client._post_graphql("query", {})

    assert calls["n"] == 1


def test_gives_up_after_max_attempts(
    client: GitHubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistent 502s exhaust the retry budget and then raise."""
    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        calls["n"] += 1
        return _response(502, {})

    monkeypatch.setattr(client.client, "post", fake_post)

    with pytest.raises(GitHubAPIError):
        client._post_graphql("query", {})

    assert calls["n"] == MAX_ATTEMPTS
