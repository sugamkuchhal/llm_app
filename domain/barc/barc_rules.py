import re
from typing import Any

DEAD_HOURS = {"00", "01", "02", "03", "04", "05"}

DEFAULT_WEEK_WINDOW = 4

NO_FILTER_SENTINEL = "__NO_FILTER__"


def is_no_filter_value(v: Any) -> bool:
    """
    Return True if value means "no constraint".
    Accepts legacy 'ALL' as well as the explicit sentinel.
    """
    if v is None:
        return True
    if not isinstance(v, str):
        v = str(v)
    vv = v.strip()
    return vv == "" or vv == NO_FILTER_SENTINEL or vv.upper() == "ALL"


def normalize_no_filter_to_none(v: Any) -> str | None:
    """
    Normalize planner value into either a concrete string or None (no filter).
    """
    if is_no_filter_value(v):
        return None
    if not isinstance(v, str):
        v = str(v)
    return v.strip()


def _question_mentions_time_window(question: str | None) -> bool:
    q = (question or "").lower()
    return bool(
        re.search(r"\b(last|latest|past)\s+\d+\s+(day|days|week|weeks)\b", q)
        or re.search(r"\b(\d{4}-\d{2}-\d{2})\b", q)
        or re.search(r"\bbetween\b.*\band\b", q)
        or re.search(r"\bfrom\b.*\bto\b", q)
        or re.search(r"\bweek_id\b|\bweek\s+\d+\b", q)
    )


def infer_include_dead_hours(question: str | None) -> bool:
    q = (question or "").lower()
    return bool(
        re.search(r"\b(include|including|with)\s+dead\s+hours\b", q)
        or re.search(r"\ball\s+hours\b|\b24x7\b|\b24\s*/\s*7\b", q)
    )


def _token_in_question(question: str, token: str) -> bool:
    """
    Best-effort whole-token match for short codes like 'HSM', '10L+', etc.
    Falls back to substring match for long values.
    """
    q = (question or "").strip()
    t = (token or "").strip()
    if not q or not t:
        return False

    # Use whole-token match for compact alphanumerics (reduces accidental matches).
    if re.fullmatch(r"[A-Za-z0-9]{2,6}", t):
        return bool(re.search(rf"(?i)(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])", q))

    return t.lower() in q.lower()


def infer_user_specified_region_target(
    *,
    question: str | None,
    candidates: list[dict] | None,
    inferred_genre: str | None = None,
) -> tuple[bool, bool]:
    """
    Heuristic: detect whether user specified region/target in the question.

    Important nuance:
    - Short codes like 'HSM' can appear as both Genre and Region. If the only
      match is the inferred genre code and the user did not say 'region/market',
      treat it as NOT specifying region.
    """
    q = (question or "")
    ql = q.lower()
    region_hint = bool(re.search(r"\b(region|market)\b", ql))

    cand_regions = {
        str(r.get("region", "")).strip()
        for r in (candidates or [])
        if isinstance(r, dict) and r.get("region")
    }
    cand_targets = {
        str(r.get("target", "")).strip()
        for r in (candidates or [])
        if isinstance(r, dict) and r.get("target")
    }

    def _contains_any(values: set[str]) -> str | None:
        for v in values:
            if v and _token_in_question(q, v):
                return v
        return None

    region_match = _contains_any(cand_regions)
    target_match = _contains_any(cand_targets)

    # If a region token is used in a clear locative phrase, treat as explicit region.
    if region_match and re.search(
        rf"(?i)\b(in|within|across|for)\s+{re.escape(region_match)}\b",
        q,
    ):
        region_hint = True

    # Disambiguate genre-like codes (HSM/EBN/HBN v1/etc.) being treated as a region.
    if (
        region_match
        and inferred_genre
        and region_match.strip().lower() == inferred_genre.strip().lower()
        and not region_hint
    ):
        region_match = None

    return bool(region_match), bool(target_match)


def resolve_time_window_value(*, planner_value: str | None, question: str | None) -> tuple[str, str]:
    """
    Returns: (value, source) where source ∈ {'explicit','default'}
    """
    if not is_no_filter_value(planner_value):
        return (str(planner_value).strip(), "explicit")
    # If the user didn't specify time, default to last 4 weeks.
    if not _question_mentions_time_window(question):
        return ("Last 4 Weeks", "default")
    # User specified time but planner left it unconstrained: keep explicit no-filter
    # so callers can raise a clear error.
    return (NO_FILTER_SENTINEL, "explicit")


def choose_default_with_constraints(
    *,
    dim: str,
    selected_default_dimensions: dict | None,
    allowed_rows: list[dict] | None,
    constraints: dict[str, str | None],
) -> tuple[str | None, str]:
    """
    Choose a default for 'region' or 'target' under other constraints (genre/channel/other dim).
    Returns: (value_or_none, source) where source ∈ {'default','inferred','none'}
    """
    desired = normalize_no_filter_to_none((selected_default_dimensions or {}).get(dim))

    # Filter allowed rows by constraints (best-effort).
    rows = []
    for r in allowed_rows or []:
        if not isinstance(r, dict):
            continue
        ok = True
        for k, v in constraints.items():
            if v is None:
                continue
            rv = normalize_no_filter_to_none(r.get(k))
            if rv is None or rv.strip().lower() != v.strip().lower():
                ok = False
                break
        if ok:
            rows.append(r)

    # If the default exists within constrained rows, use it.
    if desired:
        for r in rows:
            rv = normalize_no_filter_to_none(r.get(dim))
            if rv and rv.strip().lower() == desired.strip().lower():
                return (rv, "default")

    # If constrained rows imply a single value, use it (inferred).
    uniq = {
        normalize_no_filter_to_none(r.get(dim))
        for r in rows
    }
    uniq = {u for u in uniq if u}
    if len(uniq) == 1:
        return (sorted(uniq)[0], "inferred")

    # Otherwise, no constraint.
    return (None, "none")


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
