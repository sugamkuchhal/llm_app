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
    # Support digit and word numbers (e.g., "latest two weeks").
    word_num = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple)"
    return bool(
        re.search(rf"\b(last|latest|past)\s+(?:\d+|{word_num})\s+(day|days|week|weeks)\b", q)
        or re.search(r"\b(last|latest|past)\s+week\b", q)
        or re.search(r"\b(\d{4}-\d{2}-\d{2})\b", q)
        or re.search(r"\bbetween\b.*\band\b", q)
        or re.search(r"\bfrom\b.*\bto\b", q)
        or re.search(r"\bweek_id\b|\bweek\s+\d+\b", q)
    )


def infer_requested_weeks(question: str | None) -> int | None:
    """
    Best-effort extraction of a requested week window from user text.
    Supports digit and common word numbers (two/three/...).
    """
    q = (question or "").lower()
    m = re.search(r"\b(last|latest|past)\s+(\d+)\s+weeks?\b", q)
    if m:
        try:
            return int(m.group(2))
        except ValueError:
            return None

    word_map = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "couple": 2,
    }
    m2 = re.search(r"\b(last|latest|past)\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple)\s+weeks?\b", q)
    if m2:
        return word_map.get(m2.group(2))

    # Handle "last week"/"latest week" as 1 week.
    if re.search(r"\b(last|latest|past)\s+week\b", q):
        return 1

    return None


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
    # Allow slightly longer codes too (e.g. FACTUAL).
    if re.fullmatch(r"[A-Za-z0-9]{2,10}", t):
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

    # If a region token is used in a clear locative phrase, treat as explicit region,
    # EXCEPT when the token is also the inferred genre code (HSM/EBN/etc.). In that
    # ambiguous case we only treat it as region if the user explicitly says region/market.
    if region_match and re.search(rf"(?i)\b(in|within|across|for)\s+{re.escape(region_match)}\b", q):
        if not (inferred_genre and region_match.strip().lower() == inferred_genre.strip().lower()):
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

    # STRICT DEFAULT RULE:
    # "default" means it comes from rows where is_default == True.
    default_rows = [
        r
        for r in rows
        if isinstance(r, dict) and bool(r.get("is_default")) is True
    ]

    # If the desired default exists within constrained DEFAULT rows, use it.
    if desired:
        for r in default_rows:
            rv = normalize_no_filter_to_none(r.get(dim))
            if rv and rv.strip().lower() == desired.strip().lower():
                return (rv, "default")

    # If constrained DEFAULT rows imply a single value, use it (default).
    uniq_default = {normalize_no_filter_to_none(r.get(dim)) for r in default_rows}
    uniq_default = {u for u in uniq_default if u}
    if len(uniq_default) == 1:
        return (sorted(uniq_default)[0], "default")

    # If there are multiple default candidates, pick deterministically.
    if len(uniq_default) > 1:
        return (sorted(uniq_default)[0], "default")

    # If constrained rows imply a single value, use it (inferred).
    uniq = {
        normalize_no_filter_to_none(r.get(dim))
        for r in rows
    }
    uniq = {u for u in uniq if u}
    if len(uniq) == 1:
        return (sorted(uniq)[0], "inferred")

    # Practical fallback (deep in the tree):
    # If target is still ambiguous for a (genre, region, ...) slice and there is
    # no region-specific is_default=TRUE row, fall back to a genre-level default
    # target derived from is_default=TRUE rows (ignoring region).
    #
    # This chooses a deterministic target instead of returning "no filter".
    # NOTE: This is NOT labeled as "default" because it is not a true default
    # for the (genre, region) slice; we label it as "inferred".
    if dim == "target" and len(uniq) > 1:
        genre = (constraints or {}).get("genre")
        genre_norm = normalize_no_filter_to_none(genre)
        if genre_norm:
            genre_default_targets = sorted(
                {
                    normalize_no_filter_to_none(r.get("target"))
                    for r in (allowed_rows or [])
                    if isinstance(r, dict)
                    and bool(r.get("is_default")) is True
                    and normalize_no_filter_to_none(r.get("genre"))
                    and normalize_no_filter_to_none(r.get("genre")).strip().lower() == genre_norm.strip().lower()
                    and normalize_no_filter_to_none(r.get("target"))
                }
            )
            if genre_default_targets:
                # Prefer the SYSTEM default if it matches one of the genre defaults.
                if desired and any(t.strip().lower() == desired.strip().lower() for t in genre_default_targets):
                    for t in genre_default_targets:
                        if t.strip().lower() == desired.strip().lower():
                            return (t, "inferred")
                return (genre_default_targets[0], "inferred")

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


def sql_has_dim_filter(sql: str, *, col: str, value: str) -> bool:
    """
    Best-effort detection that SQL constrains a dimension column to a value.

    Important: do NOT use a trailing word-boundary after the closing quote
    (e.g. `='HSM'\\b`) because `'` is not a word character and it causes false
    negatives depending on following whitespace/newlines.
    """
    s = sql or ""
    v = (value or "").strip()
    if not v:
        return False
    v_re = re.escape(v)
    c = re.escape(col)

    return bool(
        re.search(rf"\b{c}\b\s*=\s*'{v_re}'", s, flags=re.IGNORECASE)
        or re.search(rf"\blower\s*\(\s*\b{c}\b\s*\)\s*=\s*lower\s*\(\s*'{v_re}'\s*\)", s, flags=re.IGNORECASE)
        or re.search(rf"\bupper\s*\(\s*\b{c}\b\s*\)\s*=\s*upper\s*\(\s*'{v_re}'\s*\)", s, flags=re.IGNORECASE)
        or re.search(rf"\btrim\s*\(\s*\b{c}\b\s*\)\s*=\s*'{v_re}'", s, flags=re.IGNORECASE)
        or re.search(rf"\bupper\s*\(\s*trim\s*\(\s*\b{c}\b\s*\)\s*\)\s*=\s*upper\s*\(\s*'{v_re}'\s*\)", s, flags=re.IGNORECASE)
        or re.search(rf"\blower\s*\(\s*trim\s*\(\s*\b{c}\b\s*\)\s*\)\s*=\s*lower\s*\(\s*'{v_re}'\s*\)", s, flags=re.IGNORECASE)
        or re.search(rf"\b{c}\b\s+in\s*\(\s*'{v_re}'", s, flags=re.IGNORECASE)
    )


def sql_has_any_dim_predicate(sql: str, *, col: str) -> bool:
    """
    Best-effort detection that SQL has *any* predicate on a dimension column.
    Used to assert "no region/target filter should exist".
    """
    s = sql or ""
    c = re.escape(col)
    return bool(
        re.search(rf"\b{c}\b\s*=\s*'[^']+'", s, flags=re.IGNORECASE)
        or re.search(rf"\b{c}\b\s+in\s*\(", s, flags=re.IGNORECASE)
        or re.search(rf"\b(lower|upper|trim)\s*\(\s*[\w\.\s]*\b{c}\b[\w\.\s]*\)\s*(=|in)\b", s, flags=re.IGNORECASE)
    )
