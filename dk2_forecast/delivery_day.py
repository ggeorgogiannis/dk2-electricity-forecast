from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

COPENHAGEN = ZoneInfo("Europe/Copenhagen")
QUARTER_HOUR = timedelta(minutes=15)


def forecast_issue_utc(delivery_date: date) -> datetime:
    issue_date = delivery_date - timedelta(days=1)
    return datetime.combine(issue_date, time(10), COPENHAGEN).astimezone(UTC)


def delivery_intervals_utc(delivery_date: date) -> list[datetime]:
    """Return the UTC start of each quarter-hour on a Danish delivery date."""
    next_date = delivery_date + timedelta(days=1)

    # Convert both local midnights to UTC before measuring the day.
    # A Danish delivery day can span 23, 24, or 25 hours.
    start_utc = datetime.combine(delivery_date, time.min, COPENHAGEN).astimezone(UTC)
    end_utc = datetime.combine(next_date, time.min, COPENHAGEN).astimezone(UTC)

    interval_count = (end_utc - start_utc) // QUARTER_HOUR
    return [start_utc + index * QUARTER_HOUR for index in range(interval_count)]
