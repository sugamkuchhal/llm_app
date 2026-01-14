DEAD_HOURS = {"00", "01", "02", "03", "04", "05"}

DEFAULT_WEEK_WINDOW = 4


def resolve_time_window(planner_text: str):
    """
    Determines time window intent from planner output.
    """
    t = planner_text.lower()

    if "last 4 weeks" in t or "latest 4 weeks" in t:
        return "Last 4 Weeks"

    return None


def has_dead_hours_filter(planner_text: str) -> bool:
    """
    Indicates whether dead hours exclusion is expected.
    """
    return "dead hours" in planner_text.lower()
