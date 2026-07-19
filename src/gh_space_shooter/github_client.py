"""GitHub API client for fetching contribution graph data."""

import time
from datetime import date, datetime, timedelta, timezone
from typing import TypedDict

import httpx
from dotenv import load_dotenv

from .constants import NUM_WEEKS

# Load environment variables from .env file
load_dotenv()

# Retry settings for transient GitHub API failures.
# GitHub's GraphQL endpoint intermittently returns 5xx responses (e.g. 502 Bad
# Gateway) or transient rate-limit errors. A single such response should not
# fail the run, so we retry with exponential backoff.
MAX_ATTEMPTS = 4
INITIAL_BACKOFF_SECONDS = 2.0
BACKOFF_MULTIPLIER = 2.0
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Substrings identifying transient GraphQL errors returned with HTTP 200.
# NOTE: "resource limits exceeded" is deliberately NOT here — it is a
# deterministic per-query cost failure (see MAX_QUERY_WEEKS below), so retrying
# it is pointless; we avoid it by chunking instead.
RETRYABLE_GRAPHQL_SIGNALS = (
    "rate limit",
    "timeout",
    "timed out",
    "something went wrong while executing your query",
)

# Fetching the day-level contribution calendar for a full year in a single
# query trips GitHub's GraphQL "Resource limits for this query exceeded" error
# (and sometimes a 502 timeout) for high-activity accounts, because the backend
# has to scan the whole year's contributions to bucket them per day. Requesting
# the days in short windows keeps each query cheap enough to succeed; we then
# stitch the windows back into the full grid ourselves. Four weeks per request
# is comfortably under the threshold observed in practice.
MAX_QUERY_WEEKS = 4
DAYS_PER_WEEK = 7

# Firing the ~13 windowed requests back-to-back trips GitHub's *secondary* rate
# limit (a 403/429 aimed at bursts, distinct from the hourly point budget).
# GitHub advises spacing requests ~1s apart, so we pause between windows.
INTER_REQUEST_DELAY_SECONDS = 1.0


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

    # Cheap scalar query for the year's total — always well under resource limits.
    TOTAL_CONTRIBUTIONS_QUERY = """
        query($username: String!) {
            user(login: $username) {
                contributionsCollection {
                    contributionCalendar {
                        totalContributions
                    }
                }
            }
        }
    """

    # Day-level query scoped to an explicit (from, to) window so it stays cheap.
    CONTRIBUTION_DAYS_QUERY = """
        query($username: String!, $from: DateTime!, $to: DateTime!) {
            user(login: $username) {
                contributionsCollection(from: $from, to: $to) {
                    contributionCalendar {
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

        The day-level calendar is fetched in short date windows and stitched
        back together, so it works even for high-activity accounts whose full
        year trips GitHub's GraphQL resource limits (see MAX_QUERY_WEEKS).

        Args:
            username: GitHub username to fetch data for

        Returns:
            ContributionData with user's contribution information

        Raises:
            GitHubAPIError: If the API request fails
        """
        total_contributions = self._fetch_total_contributions(username)
        day_levels = self._fetch_contribution_days(username)
        weeks = self._build_week_grid(day_levels)

        return {
            "username": username,
            "total_contributions": total_contributions,
            "weeks": weeks,
        }

    def _fetch_total_contributions(self, username: str) -> int:
        """Fetch the year's total contribution count (a cheap scalar query)."""
        data = self._post_graphql(
            self.TOTAL_CONTRIBUTIONS_QUERY, {"username": username}
        )
        user = data.get("data", {}).get("user")
        if not user:
            raise GitHubAPIError(f"User '{username}' not found")
        return user["contributionsCollection"]["contributionCalendar"][
            "totalContributions"
        ]

    def _fetch_contribution_days(self, username: str) -> dict[str, ContributionDay]:
        """Fetch per-day contributions for the last NUM_WEEKS weeks.

        Queries the day-level calendar in windows of at most MAX_QUERY_WEEKS to
        stay under GitHub's per-query resource limit, merging the results into a
        single ``date -> ContributionDay`` map.
        """
        grid_start, today = self._grid_range()
        day_levels: dict[str, ContributionDay] = {}

        window_start = grid_start
        window_span = timedelta(weeks=MAX_QUERY_WEEKS)
        first_window = True
        while window_start <= today:
            # Pace the windowed requests to avoid GitHub's secondary rate limit.
            if not first_window:
                time.sleep(INTER_REQUEST_DELAY_SECONDS)
            first_window = False

            window_end = min(window_start + window_span - timedelta(days=1), today)
            data = self._post_graphql(
                self.CONTRIBUTION_DAYS_QUERY,
                {
                    "username": username,
                    "from": f"{window_start.isoformat()}T00:00:00Z",
                    "to": f"{window_end.isoformat()}T23:59:59Z",
                },
            )
            user = data.get("data", {}).get("user")
            if not user:
                raise GitHubAPIError(f"User '{username}' not found")

            calendar = user["contributionsCollection"]["contributionCalendar"]
            for week_data in calendar["weeks"]:
                for day_data in week_data["contributionDays"]:
                    day_levels[day_data["date"]] = {
                        "date": day_data["date"],
                        "count": day_data["contributionCount"],
                        "level": self._contribution_level_to_int(
                            day_data["contributionLevel"]
                        ),
                    }

            window_start = window_end + timedelta(days=1)

        return day_levels

    def _build_week_grid(
        self, day_levels: dict[str, ContributionDay]
    ) -> list[ContributionWeek]:
        """Reconstruct the NUM_WEEKS x 7 grid from a date -> day map.

        Days with no fetched data (gaps, or the not-yet-happened days of the
        current week) default to zero, matching how GitHub renders them.
        """
        grid_start, _ = self._grid_range()
        weeks: list[ContributionWeek] = []
        cursor = grid_start
        for _ in range(NUM_WEEKS):
            days: list[ContributionDay] = []
            for _ in range(DAYS_PER_WEEK):
                key = cursor.isoformat()
                days.append(
                    day_levels.get(
                        key, {"date": key, "count": 0, "level": 0}
                    )
                )
                cursor += timedelta(days=1)
            weeks.append({"days": days})
        return weeks

    @staticmethod
    def _grid_range() -> tuple[date, date]:
        """Return (grid_start, today) for the trailing NUM_WEEKS-week grid.

        ``grid_start`` is the Sunday NUM_WEEKS-1 weeks before the current week,
        matching GitHub's Sunday-aligned contribution columns.
        """
        today = datetime.now(timezone.utc).date()
        # Python's weekday() is Mon=0..Sun=6; days since the most recent Sunday.
        days_since_sunday = (today.weekday() + 1) % DAYS_PER_WEEK
        current_week_start = today - timedelta(days=days_since_sunday)
        grid_start = current_week_start - timedelta(weeks=NUM_WEEKS - 1)
        return grid_start, today

    def _post_graphql(self, query: str, variables: dict) -> dict:
        """POST a GraphQL query, retrying transient failures.

        Retries on 5xx/429 responses, network/transport errors, and transient
        GraphQL errors (rate limits, timeouts), backing off exponentially
        between attempts. Non-transient failures (a genuine query error, an
        unknown user, a resource-limit error) are raised immediately.

        Returns:
            The parsed JSON response body once a non-error response is received.

        Raises:
            GitHubAPIError: If every attempt fails, or on a non-transient error.
        """
        last_error: GitHubAPIError | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            retry_after: float | None = None
            try:
                response = self.client.post(
                    self.GITHUB_API_URL,
                    json={"query": query, "variables": variables},
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                last_error = GitHubAPIError(
                    f"Failed to fetch data from GitHub API: {e}"
                )
                if (
                    e.response.status_code in RETRYABLE_STATUS_CODES
                    or self._is_secondary_rate_limit(e.response)
                ):
                    retry_after = self._retry_after_seconds(e.response)
                else:
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
                delay = (
                    retry_after
                    if retry_after is not None
                    else INITIAL_BACKOFF_SECONDS * BACKOFF_MULTIPLIER ** (attempt - 1)
                )
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

    @staticmethod
    def _is_secondary_rate_limit(response: httpx.Response) -> bool:
        """Return True if a 403/429 looks like GitHub's secondary rate limit."""
        if response.status_code not in (403, 429):
            return False
        if response.headers.get("retry-after"):
            return True
        return "secondary rate limit" in response.text.lower()

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        """Parse the Retry-After header (seconds) if GitHub sent one."""
        value = response.headers.get("retry-after")
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    LEVEL_MAP = {
        "NONE": 0,
        "FIRST_QUARTILE": 1,
        "SECOND_QUARTILE": 2,
        "THIRD_QUARTILE": 3,
        "FOURTH_QUARTILE": 4,
    }

    def _contribution_level_to_int(self, level: str) -> int:
        return self.LEVEL_MAP.get(level, 0)
