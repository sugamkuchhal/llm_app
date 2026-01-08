import re

from domain.barc.barc_mappings import resolve_genre
from domain.barc.barc_defaults import (
    DEFAULT_REGION,
    DEFAULT_TARGET,
    DEFAULT_CHANNEL,
    DEFAULT_TIME_WINDOW
)
from domain.barc.barc_rules import resolve_time_window


def build_filters(metric_manifest, sql_text, planner_text):
    """
    BARC-specific business filter derivation.
    """

    filters = {}

    definition = metric_manifest.get("definition")

    genre = resolve_genre(definition)
    if genre:
        filters["Genre"] = genre

    def extract(sql, key):
        m = re.search(rf"{key}\s*=\s*'([^']+)'", sql, re.IGNORECASE)
        return m.group(1) if m else None

    for key in ["region", "target", "channel"]:
        value = extract(sql_text, key)
        if value:
            filters[key.capitalize()] = value

    time_window = resolve_time_window(planner_text or "")
    if time_window:
        filters["Time Window"] = time_window

    filters.setdefault("Channel", DEFAULT_CHANNEL)

    if "Time Window" not in filters:
        filters["Time Window"] = f"{DEFAULT_TIME_WINDOW} (default)"

    filters.setdefault("Region", DEFAULT_REGION)
    filters.setdefault("Target", DEFAULT_TARGET)


    return [f"{k}: {v}" for k, v in filters.items()]
