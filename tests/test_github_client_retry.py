"""Tests for GitHubClient transient-failure retry with exponential backoff."""

import httpx
import pytest

from gh_space_shooter.github_client import (
    MAX_ATTEMPTS,
    GitHubAPIError,
    GitHubClient,
)

API_URL = GitHubClient.GITHUB_API_URL


def _success_payload() -> dict:
    """A minimal well-formed contribution-graph response body."""
    return {
        "data": {
            "user": {
                "contributionsCollection": {
                    "contributionCalendar": {
                        "totalContributions": 1,
                        "weeks": [
                            {
                                "contributionDays": [
                                    {
                                        "date": "2026-07-19",
                                        "contributionCount": 1,
                                        "contributionLevel": "FIRST_QUARTILE",
                                    }
                                ]
                            }
                        ],
                    }
                }
            }
        }
    }


def _response(status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", API_URL),
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> GitHubClient:
    """A GitHubClient with sleep patched out so retries don't wait."""
    monkeypatch.setattr(
        "gh_space_shooter.github_client.time.sleep", lambda _seconds: None
    )
    return GitHubClient(token="fake-token")


def test_retries_on_502_then_succeeds(
    client: GitHubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 502 Bad Gateway is retried and the subsequent success is returned."""
    responses = [_response(502, {}), _response(200, _success_payload())]
    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        response = responses[calls["n"]]
        calls["n"] += 1
        return response

    monkeypatch.setattr(client.client, "post", fake_post)

    result = client.get_contribution_graph("Rumbles")

    assert calls["n"] == 2
    assert result["total_contributions"] == 1


def test_retries_on_transient_graphql_error_then_succeeds(
    client: GitHubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 200 carrying "Resource limits ... exceeded" is retried, then succeeds."""
    responses = [
        _response(
            200,
            {"errors": [{"message": "Resource limits for this query exceeded."}]},
        ),
        _response(200, _success_payload()),
    ]
    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        response = responses[calls["n"]]
        calls["n"] += 1
        return response

    monkeypatch.setattr(client.client, "post", fake_post)

    result = client.get_contribution_graph("Rumbles")

    assert calls["n"] == 2
    assert result["total_contributions"] == 1


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
        client.get_contribution_graph("Rumbles")

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
        client.get_contribution_graph("Rumbles")

    assert calls["n"] == MAX_ATTEMPTS
