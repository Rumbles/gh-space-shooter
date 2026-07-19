"""GitHub API client for fetching contribution graph data."""

import time
from datetime import datetime
from typing import TypedDict

import httpx
from dotenv import load_dotenv

from .constants import NUM_WEEKS

# Load environment variables from .env file
load_dotenv()

# Retry settings for transient GitHub API failures.
# GitHub's GraphQL endpoint intermittently returns 5xx responses (e.g. 502 Bad
# Gateway) or 200 responses carrying a transient error such as "Resource limits
# for this query exceeded". A single such response should not fail the run, so
# we retry with exponential backoff.
MAX_ATTEMPTS = 4
INITIAL_BACKOFF_SECONDS = 2.0
BACKOFF_MULTIPLIER = 2.0
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Substrings identifying transient GraphQL errors returned with HTTP 200.
RETRYABLE_GRAPHQL_SIGNALS = (
    "resource limit",
    "rate limit",
    "timeout",
    "timed out",
    "something went wrong while executing your query",
)


class ContributionDay(TypedDict):
    """Represents a single day's contribution data."""

    date: str
    count: int
    level: int  # 0-4 intensity level


class ContributionWeek(TypedDict):
    """Represents a week of contribution data."""

    days: list[ContributionDay]


class ContributionData(TypedDict):
    """Complete contribution graph data."""

    username: str
    total_contributions: int
    weeks: list[ContributionWeek]


class GitHubAPIError(Exception):
    """Raised when GitHub API request fails."""

    pass


class GitHubClient:
    """Client for interacting with GitHub's GraphQL API."""

    GITHUB_API_URL = "https://api.github.com/graphql"
    GET_CONTRIBUTION_GRAPH_QUERY = """
        query($username: String!) {
            user(login: $username) {
            contributionsCollection {
                contributionCalendar {
                totalContributions
                weeks {
                    contributionDays {
                    date
                    contributionCount
                    contributionLevel
                    }
                }
                }
            }
            }
        }
    """

    def __init__(self, token: str):
        """
        Initialize GitHub client.

        Args:
            token: GitHub personal access token (required).
        """
        self.token = token
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close HTTP client."""
        self.close()

    def close(self):
        self.client.close()

    def get_contribution_graph(self, username: str) -> ContributionData:
        """
        Fetch contribution graph for a GitHub user (last 52 weeks).

        Args:
            username: GitHub username to fetch data for

        Returns:
            ContributionData with user's contribution information

        Raises:
            GitHubAPIError: If the API request fails
        """

        data = self._post_query_with_retry(username)

        # Check if user exists
        if not data.get("data", {}).get("user"):
            raise GitHubAPIError(f"User '{username}' not found")

        # Extract contribution data
        calendar = data["data"]["user"]["contributionsCollection"][
            "contributionCalendar"
        ]

        # Parse weeks and days
        weeks: list[ContributionWeek] = []
        for week_data in calendar["weeks"]:
            days: list[ContributionDay] = []
            for day_data in week_data["contributionDays"]:
                days.append(
                    {
                        "date": day_data["date"],
                        "count": day_data["contributionCount"],
                        "level": self._contribution_level_to_int(
                            day_data["contributionLevel"]
                        ),
                    }
                )
            weeks.append({"days": days})

        # Always return exactly NUM_WEEKS (truncate if more)
        weeks = weeks[-NUM_WEEKS:] if len(weeks) > NUM_WEEKS else weeks

        return {
            "username": username,
            "total_contributions": calendar["totalContributions"],
            "weeks": weeks,
        }

    def _post_query_with_retry(self, username: str) -> dict:
        """POST the contribution query, retrying transient failures.

        Retries on 5xx/429 responses, network/transport errors, and transient
        GraphQL errors (e.g. "Resource limits for this query exceeded"), backing
        off exponentially between attempts. Non-transient failures (a genuine
        query error, an unknown user) are raised immediately.

        Returns:
            The parsed JSON response body once a non-error response is received.

        Raises:
            GitHubAPIError: If every attempt fails, or on a non-transient error.
        """
        last_error: GitHubAPIError | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self.client.post(
                    self.GITHUB_API_URL,
                    json={
                        "query": self.GET_CONTRIBUTION_GRAPH_QUERY,
                        "variables": {"username": username},
                    },
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                last_error = GitHubAPIError(
                    f"Failed to fetch data from GitHub API: {e}"
                )
                if e.response.status_code not in RETRYABLE_STATUS_CODES:
                    raise last_error from e
            except httpx.HTTPError as e:
                # Transport-level error (timeout, connection reset, ...): retry.
                last_error = GitHubAPIError(
                    f"Failed to fetch data from GitHub API: {e}"
                )
            else:
                errors = data.get("errors")
                if not errors:
                    return data
                error_messages = [
                    error.get("message", str(error)) for error in errors
                ]
                last_error = GitHubAPIError(", ".join(error_messages))
                if not self._is_retryable_graphql_error(error_messages):
                    raise last_error

            if attempt < MAX_ATTEMPTS:
                delay = INITIAL_BACKOFF_SECONDS * BACKOFF_MULTIPLIER ** (attempt - 1)
                print(
                    f"GitHub API request failed "
                    f"(attempt {attempt}/{MAX_ATTEMPTS}), "
                    f"retrying in {delay:.0f}s: {last_error}"
                )
                time.sleep(delay)

        # All attempts exhausted; surface the most recent error.
        assert last_error is not None
        raise last_error

    @staticmethod
    def _is_retryable_graphql_error(messages: list[str]) -> bool:
        """Return True if any GraphQL error message looks transient."""
        combined = " ".join(messages).lower()
        return any(signal in combined for signal in RETRYABLE_GRAPHQL_SIGNALS)

    LEVEL_MAP = {
        "NONE": 0,
        "FIRST_QUARTILE": 1,
        "SECOND_QUARTILE": 2,
        "THIRD_QUARTILE": 3,
        "FOURTH_QUARTILE": 4,
    }

    def _contribution_level_to_int(self, level: str) -> int:
        return self.LEVEL_MAP.get(level, 0)
