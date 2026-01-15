import os
import json
import re
import logging
import vertexai
import markdown
import hashlib
import uuid
import warnings
import base64
import datetime
from decimal import Decimal
from utils.logging_setup import configure_logging, set_request_id, get_log_buffer

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="vertexai.generative_models"
)


from flask import Flask, request, render_template_string, session
from google.cloud import bigquery
from vertexai.preview.generative_models import GenerativeModel
from validators.answer_validator import validate_answer
from pipeline.analysis_runner import run_analysis
from ui.answer_builder import build_ui_answer
from domain.barc.barc_validation import shadow_resolve_dimensions_bq
from domain.barc.barc_mappings import resolve_genre
from domain.barc.barc_rules import (
    NO_FILTER_SENTINEL,
    is_no_filter_value,
    normalize_no_filter_to_none,
    infer_user_specified_region_target,
    resolve_time_window_value,
    infer_include_dead_hours,
    choose_default_with_constraints,
)
from domain.barc.barc_dimension_reference import (
    fetch_default_dimension_rows,
    fetch_candidate_dimension_rows_for_question,
    merge_dimension_rows,
    pick_selected_default_row,
)
from llm.planner import call_planner
from llm.interpreter import call_interpreter
from config.prompt_guard import assert_prompt_unchanged, hash_text


import sys
import logging

# Centralized logging configuration (adds request_id to every line).
configure_logging()
logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
logger.propagate = True


# --------------------------------------------------
# Config
# --------------------------------------------------
PROJECT_ID = os.getenv("GCP_PROJECT", "prj-uat-data-coe-analytics34")

VERTEX_REGION = os.getenv("VERTEX_REGION", "us-central1")
# VERTEX_REGION = os.getenv("VERTEX_REGION", "asia-south1")
# BQ_REGION = os.getenv("BIGQUERY_REGION", "us-central1")
BQ_REGION = os.getenv("BIGQUERY_REGION", "asia-south1")

SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

MAX_TURNS = int(os.getenv("MAX_TURNS", 10))

MAX_BQ_BYTES = int(
    os.getenv("MAX_BQ_BYTES", 200 * 1024 * 1024 * 1024)
)  # default: 200 GB

MAX_ROWS = int(os.getenv("MAX_ROWS", 50000))

# --------------------------------------------------
# Init
# --------------------------------------------------
_bq_client = None
_planner_model = None
_interpreter_model = None
_vertex_inited = False


def get_bq_client():
    """
    Lazily initialize BigQuery client.

    Cloud Run should provide ADC via the service account. Initializing lazily
    prevents container startup failure when credentials aren't available yet.
    """
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=PROJECT_ID, location=BQ_REGION)
    return _bq_client


def get_models():
    """
    Lazily initialize Vertex AI and the LLM model handles.
    """
    global _vertex_inited, _planner_model, _interpreter_model
    if not _vertex_inited:
        vertexai.init(project=PROJECT_ID, location=VERTEX_REGION)
        _vertex_inited = True
    if _planner_model is None:
        _planner_model = GenerativeModel("gemini-2.5-pro")
    if _interpreter_model is None:
        _interpreter_model = GenerativeModel("gemini-2.5-pro")
    return _planner_model, _interpreter_model

app = Flask(__name__)
app.jinja_env.filters["markdown"] = lambda text: markdown.markdown(text or "")

app.secret_key = SECRET_KEY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_file(path):
    """
    Deterministic file loader:
    - UTF-8 (strict)
    - stable newlines (normalize to \\n)
    """
    full_path = os.path.join(BASE_DIR, path)
    try:
        with open(full_path, "r", encoding="utf-8", errors="strict") as f:
            # Note: Python's text mode already does universal newlines, but we
            # normalize explicitly for hash stability across environments.
            text = f.read()
    except OSError as e:
        raise RuntimeError(f"Failed to read file: {full_path}") from e

    return text.replace("\r\n", "\n").replace("\r", "\n")

DOMAIN = "barc"

SYSTEM_PROMPT = load_file("prompt_system.txt")
INTERPRETER_PROMPT = load_file("prompt_interpreter.txt")
ANSWER_CONTRACT = load_file("contract_answer.json")

PLANNER_PROMPT = load_file(f"domain/{DOMAIN}/{DOMAIN}_context_planner.txt")
META_DATA = load_file(f"domain/{DOMAIN}/{DOMAIN}_meta.json")

try:
    DOMAIN_META = json.loads(META_DATA)
except json.JSONDecodeError as e:
    raise RuntimeError("Invalid domain metadata JSON (barc_meta.json)") from e

SYSTEM_PROMPT_HASH = "8386449207f92e38e138bcfe2d3227865d61d64d5159cacc772c8c6b38b2c3ff"
PLANNER_PROMPT_HASH = "ce28b3fc8687ad1b98943ffe220ad0d3132db737bcb7f8fb42a78285a9e39d1d"

assert_prompt_unchanged("SYSTEM_PROMPT", SYSTEM_PROMPT, SYSTEM_PROMPT_HASH)
assert_prompt_unchanged("PLANNER_PROMPT", PLANNER_PROMPT, PLANNER_PROMPT_HASH)


# --------------------------------------------------
# Session memory
# --------------------------------------------------
def get_chat_history():
    """
    Returns the current chat history from the session.

    Policy A (self-healing): if the stored value is malformed, reset it to [].
    """
    h = session.get("chat_history")
    if h is None:
        session["chat_history"] = []
        return session["chat_history"]

    if not isinstance(h, list):
        session["chat_history"] = []
        return session["chat_history"]

    # Best-effort sanitization: keep only dict entries (stable shape downstream).
    sanitized = [x for x in h if isinstance(x, dict)]
    if sanitized is not h:
        session["chat_history"] = sanitized
    return session["chat_history"]

def append_chat_turn(question, headline):
    h = get_chat_history()

    # Normalize to deterministic, JSON-serializable primitives.
    q = (question or "")
    if not isinstance(q, str):
        q = str(q)
    q = q.strip()

    hl = None if headline is None else headline
    if hl is not None and not isinstance(hl, str):
        hl = str(hl)
    if isinstance(hl, str):
        hl = hl.strip()

    h.append({"question": q, "headline": hl})
    session["chat_history"] = h[-MAX_TURNS:]

def extract_metric_manifest(text: str):
    """
    Fail-fast extraction of the metric manifest.

    Contract:
    - Exactly one BEGIN_METRIC_MANIFEST and one END_METRIC_MANIFEST marker
    - Content between markers must be a JSON array
    """
    begin = "BEGIN_METRIC_MANIFEST"
    end = "END_METRIC_MANIFEST"

    begin_count = text.count(begin)
    end_count = text.count(end)
    if begin_count != 1 or end_count != 1:
        raise RuntimeError(
            f"Invalid METRIC_MANIFEST markers: begin={begin_count}, end={end_count}"
        )

    begin_idx = text.find(begin)
    end_idx = text.find(end)
    if begin_idx == -1 or end_idx == -1 or end_idx <= begin_idx:
        raise RuntimeError("Invalid METRIC_MANIFEST marker ordering")

    block = text[begin_idx + len(begin):end_idx].strip()
    logger.info("RAW METRIC_MANIFEST >>>\n%s\n<<< END METRIC_MANIFEST", block)

    # Deterministic guard: the manifest must be a JSON array.
    if not block.startswith("[") or not block.endswith("]"):
        raise RuntimeError("METRIC_MANIFEST must be a JSON array")

    try:
        return json.loads(block)
    except json.JSONDecodeError as e:
        # Include parse position for debuggability without attempting repair.
        context_start = max(e.pos - 40, 0)
        context_end = min(e.pos + 40, len(block))
        context = block[context_start:context_end].replace("\n", "\\n")
        raise RuntimeError(
            f"Invalid METRIC_MANIFEST JSON at pos={e.pos}: {e.msg} | context='{context}'"
        ) from e


def extract_filters(planner_text: str) -> dict:
    """
    Extract and parse the mandatory BEGIN_FILTERS/END_FILTERS block.
    """
    begin = "BEGIN_FILTERS"
    end = "END_FILTERS"
    if planner_text.count(begin) != 1 or planner_text.count(end) != 1:
        raise RuntimeError("Planner must return exactly one FILTERS block")

    b = planner_text.find(begin)
    e = planner_text.find(end)
    if b == -1 or e == -1 or e <= b:
        raise RuntimeError("Invalid FILTERS marker ordering")

    block = planner_text[b + len(begin):e].strip()
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]

    parsed: dict[str, dict] = {}
    for ln in lines:
        if not ln.startswith("-"):
            continue
        body = ln.lstrip("-").strip()
        if ":" not in body:
            continue
        key, rest = body.split(":", 1)
        key = key.strip().lower()
        parts = [p.strip() for p in rest.split(":") if p.strip()]
        value = parts[-1] if parts else ""
        parsed[key] = {"raw": ln, "value": value}

    required = ["genre", "region", "target", "channel", "time_window"]
    missing = [k for k in required if k not in parsed]
    if missing:
        raise RuntimeError(f"FILTERS block missing required keys: {missing}")

    def _display(v: str) -> str:
        return "(no filter)" if is_no_filter_value(v) else (v or "")

    filters_list = [f"{k}: {_display(parsed[k]['value'])}" for k in required]
    return {"filters": parsed, "filters_list": filters_list, "resolved_filters": {}}


def make_validate_filters(
    *,
    allowed_rows: list[dict],
    candidates: list[dict],
    question: str,
    selected_default_dimensions: dict | None = None,
):
    """
    Create a validator closure that enforces:
    - region/target must be ALL unless user specified them in the question (best-effort)
    - chosen dimension values must exist in allowed_rows (column-wise)
    - if all 4 dims are specified (non-ALL), the tuple must exist in allowed_rows
    - SQL must not contain region/target equality filters when FILTERS says ALL
    """
    q = (question or "")

    # Allowed values by column (derived from allowed rows)
    allowed_by_col: dict[str, set[str]] = {"genre": set(), "region": set(), "target": set(), "channel": set()}
    for r in allowed_rows or []:
        if not isinstance(r, dict):
            continue
        for k in allowed_by_col.keys():
            v = r.get(k)
            if isinstance(v, str) and v.strip():
                allowed_by_col[k].add(v.strip().lower())

    def _suggest(col: str, value: str) -> list[str]:
        v = (value or "").strip().lower()
        if not v:
            return []
        opts = sorted(allowed_by_col.get(col, set()))
        # Prefer substring matches, else show first few options.
        hits = [o for o in opts if v in o or o in v]
        return hits[:8] if hits else opts[:8]

    def _tuple_exists(g: str, r: str, t: str, c: str) -> bool:
        key = (g.lower(), r.lower(), t.lower(), c.lower())
        for row in allowed_rows or []:
            if not isinstance(row, dict):
                continue
            rk = (
                (row.get("genre") or "").strip().lower(),
                (row.get("region") or "").strip().lower(),
                (row.get("target") or "").strip().lower(),
                (row.get("channel") or "").strip().lower(),
            )
            if rk == key:
                return True
        return False

    def _suggest_tuple(g: str, r: str, t: str, c: str) -> dict | None:
        want = {
            "genre": g.lower(),
            "region": r.lower(),
            "target": t.lower(),
            "channel": c.lower(),
        }
        best = None  # (score, tie, row)
        for row in allowed_rows or []:
            if not isinstance(row, dict):
                continue
            score = 0
            for k in ("genre", "region", "target", "channel"):
                rv = (row.get(k) or "").strip().lower()
                if rv and rv == want[k]:
                    score += 1
            tie = (
                (row.get("genre") or ""),
                (row.get("region") or ""),
                (row.get("target") or ""),
                (row.get("channel") or ""),
            )
            if best is None or score > best[0] or (score == best[0] and tie < best[1]):
                best = (score, tie, row)
        return best[2] if best else None

    def validate_filters_impl(*, parsed_filters: dict, sql_blocks: list[dict], question: str, planner_text: str):
        f = parsed_filters["filters"]
        # Canonical resolution policy (single source of truth):
        # - FILTERS values are treated as planner intent, but we resolve region/target/time_window
        #   against SYSTEM defaults and allowed_rows to eliminate mismatches and ambiguity.
        # - SQL validation is always against the resolved filters.
        genre = (f.get("genre", {}).get("value") or "").strip()
        region = (f.get("region", {}).get("value") or "").strip()
        target = (f.get("target", {}).get("value") or "").strip()
        channel = (f.get("channel", {}).get("value") or "").strip()
        time_window = (f.get("time_window", {}).get("value") or "").strip()

        qtext = (question or "")
        qtext_l = qtext.lower()

        def _user_specified_time_window(text: str) -> bool:
            t = (text or "").lower()
            return bool(
                re.search(r"\b(last|latest|past)\s+\d+\s+(day|days|week|weeks)\b", t)
                or re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t)
                or re.search(r"\bbetween\b.*\band\b", t)
                or re.search(r"\bfrom\b.*\bto\b", t)
                or re.search(r"\bweek_id\b|\bweek\s+\d+\b", t)
            )

        user_specified_time = _user_specified_time_window(qtext_l)

        def _sql_touches(sql: str, table: str) -> bool:
            s = (sql or "").lower()
            # handle backticks and optional project prefix
            return f"barc_slm_poc.{table}".lower() in s

        # Dead-hours: exclude by default for time_band/program unless user explicitly includes.
        include_dead_hours = infer_include_dead_hours(qtext)

        # Heuristic correction: prevent genre codes being placed in region (common LLM slip).
        inferred_genre = resolve_genre(question)
        user_specified_region, user_specified_target = infer_user_specified_region_target(
            question=qtext,
            candidates=candidates,
            inferred_genre=inferred_genre,
        )

        if (
            inferred_genre
            and not user_specified_region
            and not is_no_filter_value(region)
            and (region or "").strip().lower() == inferred_genre.strip().lower()
            and is_no_filter_value(genre)
        ):
            f["genre"]["value"] = inferred_genre
            genre = inferred_genre
            f["region"]["value"] = NO_FILTER_SENTINEL
            region = NO_FILTER_SENTINEL

        # Resolve time_window (default to Last 4 Weeks unless user specified time).
        resolved_tw, tw_source = resolve_time_window_value(planner_value=time_window, question=qtext)
        if resolved_tw == NO_FILTER_SENTINEL and user_specified_time:
            raise RuntimeError("User specified a time window, but FILTERS time_window is not constrained")
        f["time_window"]["value"] = resolved_tw
        time_window = resolved_tw

        # If no time is specified by the user, time_window must default to last 4 weeks.
        if not user_specified_time:
            if not re.search(r"\b4\b.*\bweek", (time_window or "").lower()):
                raise RuntimeError(
                    "Time Window must default to Last 4 Weeks when user does not specify a time window"
                )

        # Resolve region/target defaults unless user specified them.
        region_source = "explicit"
        target_source = "explicit"

        if user_specified_region and is_no_filter_value(region):
            raise RuntimeError("User specified a region, but FILTERS region is not constrained")
        if user_specified_target and is_no_filter_value(target):
            raise RuntimeError("User specified a target, but FILTERS target is not constrained")

        genre_norm = normalize_no_filter_to_none(genre)
        channel_norm = normalize_no_filter_to_none(channel)
        target_norm = normalize_no_filter_to_none(target)

        if not user_specified_region and is_no_filter_value(region):
            chosen, src = choose_default_with_constraints(
                dim="region",
                selected_default_dimensions=selected_default_dimensions,
                allowed_rows=allowed_rows,
                constraints={
                    "genre": genre_norm,
                    "channel": channel_norm,
                    "target": target_norm,
                },
            )
            region_source = src
            f["region"]["value"] = chosen or NO_FILTER_SENTINEL
            region = f["region"]["value"]
        else:
            region_source = "explicit"

        region_norm = normalize_no_filter_to_none(region)

        if not user_specified_target and is_no_filter_value(target):
            chosen, src = choose_default_with_constraints(
                dim="target",
                selected_default_dimensions=selected_default_dimensions,
                allowed_rows=allowed_rows,
                constraints={
                    "genre": genre_norm,
                    "channel": channel_norm,
                    "region": region_norm,
                },
            )
            target_source = src
            f["target"]["value"] = chosen or NO_FILTER_SENTINEL
            target = f["target"]["value"]
        else:
            target_source = "explicit"

        # Decide whether dead-hours is relevant for display (only if the query touches those tables).
        touches_dead_hours_tables = False
        for qb in sql_blocks or []:
            sql = qb.get("sql", "") or ""
            if _sql_touches(sql, "time_band_table") or _sql_touches(sql, "program_table"):
                touches_dead_hours_tables = True
                break

        def _sql_has_dim_filter(sql: str, col: str, value: str) -> bool:
            s = sql or ""
            v = (value or "").strip()
            if not v:
                return False
            v_re = re.escape(v)
            # Accept a few common patterns (case-insensitive):
            # - col = 'X'
            # - LOWER(col) = LOWER('X')
            # - UPPER(col) = UPPER('X')
            # - TRIM(col) = 'X' (and combinations)
            # - col IN ('X', ...)
            return bool(
                re.search(rf"\b{re.escape(col)}\b\s*=\s*'{v_re}'\b", s, flags=re.IGNORECASE)
                or re.search(rf"\blower\s*\(\s*\b{re.escape(col)}\b\s*\)\s*=\s*lower\s*\(\s*'{v_re}'\s*\)", s, flags=re.IGNORECASE)
                or re.search(rf"\bupper\s*\(\s*\b{re.escape(col)}\b\s*\)\s*=\s*upper\s*\(\s*'{v_re}'\s*\)", s, flags=re.IGNORECASE)
                or re.search(rf"\btrim\s*\(\s*\b{re.escape(col)}\b\s*\)\s*=\s*'{v_re}'\b", s, flags=re.IGNORECASE)
                or re.search(rf"\bupper\s*\(\s*trim\s*\(\s*\b{re.escape(col)}\b\s*\)\s*\)\s*=\s*upper\s*\(\s*'{v_re}'\s*\)", s, flags=re.IGNORECASE)
                or re.search(rf"\blower\s*\(\s*trim\s*\(\s*\b{re.escape(col)}\b\s*\)\s*\)\s*=\s*lower\s*\(\s*'{v_re}'\s*\)", s, flags=re.IGNORECASE)
                or re.search(rf"\b{re.escape(col)}\b\s+in\s*\(\s*'{v_re}'\b", s, flags=re.IGNORECASE)
            )

        def _sql_has_any_dim_predicate(sql: str, col: str) -> bool:
            s = sql or ""
            c = re.escape(col)
            return bool(
                re.search(rf"\b{c}\b\s*=\s*'[^']+'\b", s, flags=re.IGNORECASE)
                or re.search(rf"\b{c}\b\s+in\s*\(", s, flags=re.IGNORECASE)
                or re.search(rf"\b(lower|upper|trim)\s*\(\s*[\w\.\s]*\b{c}\b[\w\.\s]*\)\s*(=|in)\b", s, flags=re.IGNORECASE)
            )

        def _display_value(v: str, source: str | None = None) -> str:
            if is_no_filter_value(v):
                return "(no filter)"
            suffix = ""
            if source == "default":
                suffix = " (default)"
            elif source == "inferred":
                suffix = " (inferred)"
            return f"{v}{suffix}"

        # Rebuild the user-visible list after resolution (no duplicate "default notes").
        parsed_filters["filters_list"] = [
            f"genre: {_display_value(f['genre']['value'])}",
            f"region: {_display_value(f['region']['value'], region_source)}",
            f"target: {_display_value(f['target']['value'], target_source)}",
            f"channel: {_display_value(f['channel']['value'])}",
            f"time_window: {_display_value(f['time_window']['value'], tw_source)}",
        ]
        if touches_dead_hours_tables:
            parsed_filters["filters_list"].append(
                "dead_hours: included" if include_dead_hours else "dead_hours: excluded (default)"
            )

        parsed_filters["resolved_filters"] = {
            "genre": {"value": normalize_no_filter_to_none(f["genre"]["value"]), "source": "explicit"},
            "region": {"value": normalize_no_filter_to_none(f["region"]["value"]), "source": region_source},
            "target": {"value": normalize_no_filter_to_none(f["target"]["value"]), "source": target_source},
            "channel": {"value": normalize_no_filter_to_none(f["channel"]["value"]), "source": "explicit"},
            "time_window": {"value": normalize_no_filter_to_none(f["time_window"]["value"]), "source": tw_source},
            "dead_hours": {"value": "included" if include_dead_hours else "excluded", "source": "default" if not include_dead_hours else "explicit"},
        }

        def _require_dead_hours_exclusion(sql: str):
            s = (sql or "").lower()
            # Accept common forms.
            ok = bool(
                re.search(r"left\s*\(\s*[\w\.]*time_band_half_hour\s*,\s*2\s*\)\s*not\s+in\s*\(", s)
                or re.search(r"substr\s*\(\s*[\w\.]*time_band_half_hour\s*,\s*1\s*,\s*2\s*\)\s*not\s+in\s*\(", s)
            )
            if not ok:
                raise RuntimeError(
                    "Missing dead-hours exclusion for time_band/program tables. "
                    "Expected predicate like LEFT(time_band_half_hour, 2) NOT IN ('00','01','02','03','04','05')."
                )

        def _require_last_n_weeks_week_id(sql: str, n: int):
            s = (sql or "").lower()
            # Require both:
            # 1) a "latest N week_id" selector (CTE/subquery)
            # 2) usage of those week_ids to filter/join the main query
            pat_latest = rf"select[\s\S]*distinct\s+week_id[\s\S]*order\s+by\s+week_id\s+desc[\s\S]*limit\s+{n}\b"
            pat_use = (
                r"\bweek_id\b\s+in\s*\(\s*select\s+week_id\b"
                r"|join[\s\S]*\bweek_id\b"
                r"|where[\s\S]*\bweek_id\b\s*(=|in|between|>=|>)"
            )
            ok = bool(re.search(pat_latest, s) and re.search(pat_use, s))
            if not ok:
                snippet = (sql or "").strip().replace("\n", " ")
                snippet = (snippet[:800] + "...(truncated)") if len(snippet) > 800 else snippet
                raise RuntimeError(
                    f"Missing 'latest {n} weeks' week_id filter. "
                    f"Expected a DISTINCT week_id ORDER BY week_id DESC LIMIT {n} selector "
                    f"and usage of those week_ids to constrain the query (e.g., week_id IN (...)). "
                    f"SQL_snippet={snippet}"
                )

        def _require_last_n_weeks_date(sql: str, date_col: str, n: int):
            s = (sql or "").lower()
            # Accept either INTERVAL n WEEK or INTERVAL (n*7) DAY
            days = n * 7
            dc = re.escape(date_col.lower())

            # CURRENT_DATE() may optionally include a timezone, e.g. CURRENT_DATE("Asia/Kolkata")
            cur_date = r"current_date\s*\(\s*(?:\"[^\"]+\"|\'.+?\')?\s*\)"
            sub_weeks = rf"date_sub\(\s*{cur_date}\s*,\s*interval\s*{n}\s*week\s*\)"
            sub_days = rf"date_sub\(\s*{cur_date}\s*,\s*interval\s*{days}\s*day\s*\)"

            # Allow DATE(col) wrapper as well.
            col_or_datecol = rf"(?:\b{dc}\b|date\s*\(\s*\b{dc}\b\s*\))"

            ok = bool(
                # >= DATE_SUB(CURRENT_DATE(), INTERVAL n WEEK|days DAY)
                re.search(rf"{col_or_datecol}\s*>=\s*{sub_weeks}", s)
                or re.search(rf"{col_or_datecol}\s*>=\s*{sub_days}", s)
                # BETWEEN DATE_SUB(...) AND CURRENT_DATE()
                or re.search(rf"{col_or_datecol}\s+between\s+{sub_weeks}\s+and\s+{cur_date}", s)
                or re.search(rf"{col_or_datecol}\s+between\s+{sub_days}\s+and\s+{cur_date}", s)
            )
            if not ok:
                snippet = (sql or "").strip().replace("\n", " ")
                snippet = (snippet[:800] + "...(truncated)") if len(snippet) > 800 else snippet
                raise RuntimeError(
                    f"Missing 'last {n} weeks' date filter on {date_col}. "
                    f"Expected patterns like "
                    f"{date_col} >= DATE_SUB(CURRENT_DATE(), INTERVAL {n} WEEK) "
                    f"or {date_col} BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {n} WEEK) AND CURRENT_DATE() "
                    f"(also accepts DATE({date_col}) wrappers and {days} DAY equivalent). "
                    f"SQL_snippet={snippet}"
                )

        def _require_some_date_filter(sql: str, date_col: str):
            s = (sql or "").lower()
            # Basic requirement: the date column is used with a comparison operator
            ok = bool(re.search(rf"\b{re.escape(date_col.lower())}\b\s*(>=|>|between|in)\b", s))
            if not ok:
                raise RuntimeError(
                    f"Missing date filter on {date_col} for an explicit user time window request."
                )

        # Parse time_window value best-effort for enforcement.
        tw = (time_window or "").lower()
        m_weeks = re.search(r"\b(\d+)\s*week", tw)
        n_weeks = int(m_weeks.group(1)) if m_weeks else None

        # SQL must not include region/target filters if FILTERS says ALL
        for qb in sql_blocks or []:
            sql = qb.get("sql", "") or ""
            # Domain-specific time filters and dead-hours rules (BARC tables only)
            touches_channel = _sql_touches(sql, "channel_table")
            touches_time_band = _sql_touches(sql, "time_band_table")
            touches_program = _sql_touches(sql, "program_table")

            # Enforce last 4 weeks by default when user didn't specify.
            if not user_specified_time:
                # Consistent policy: use week_id latest 4 across all BARC tables.
                if touches_channel or touches_time_band or touches_program:
                    _require_last_n_weeks_week_id(sql, 4)
            else:
                # If user specified a time window, enforce something consistent.
                if n_weeks is not None:
                    if touches_channel or touches_time_band or touches_program:
                        _require_last_n_weeks_week_id(sql, n_weeks)
                else:
                    # Unknown explicit window: require some filter on the correct date columns.
                    if touches_time_band:
                        _require_some_date_filter(sql, "time_band_date")
                    if touches_program:
                        _require_some_date_filter(sql, "program_date")
                    if touches_channel:
                        # Require any mention of week_id with a limiting pattern.
                        if "week_id" not in sql.lower():
                            raise RuntimeError("Missing week_id-based time filter for channel_table with explicit time request")

            # Dead hours exclusion for time_band/program unless explicitly included
            if (touches_time_band or touches_program) and not include_dead_hours:
                _require_dead_hours_exclusion(sql)

            resolved_region = normalize_no_filter_to_none(region)
            resolved_target = normalize_no_filter_to_none(target)

            if resolved_region is None:
                if _sql_has_any_dim_predicate(sql, "region"):
                    raise RuntimeError("SQL contains a region filter, but FILTERS region=ALL")
            else:
                if not _sql_has_dim_filter(sql, "region", resolved_region):
                    raise RuntimeError(f"SQL is missing a region filter for FILTERS region={resolved_region}")

            if resolved_target is None:
                if _sql_has_any_dim_predicate(sql, "target"):
                    raise RuntimeError("SQL contains a target filter, but FILTERS target=ALL")
            else:
                if not _sql_has_dim_filter(sql, "target", resolved_target):
                    raise RuntimeError(f"SQL is missing a target filter for FILTERS target={resolved_target}")

        # Validate non-ALL values are in allowed rows (column-wise)
        for col in ("genre", "region", "target", "channel"):
            val = (f[col]["value"] or "").strip()
            if is_no_filter_value(val):
                continue
            if val.lower() not in allowed_by_col.get(col, set()):
                raise RuntimeError(
                    f"FILTERS {col} value not allowed: '{val}'. Suggestions: {_suggest(col, val)}"
                )

        # If all 4 dims are specified, validate full tuple exists
        g = (f["genre"]["value"] or "").strip()
        r = (f["region"]["value"] or "").strip()
        t = (f["target"]["value"] or "").strip()
        c = (f["channel"]["value"] or "").strip()
        if all(x and not is_no_filter_value(x) for x in (g, r, t, c)):
            if not _tuple_exists(g, r, t, c):
                suggestion = _suggest_tuple(g, r, t, c)
                raise RuntimeError(
                    f"FILTERS tuple not found in allowed dimension rows. "
                    f"Provided={{genre:{g},region:{r},target:{t},channel:{c}}}. "
                    f"Suggested={suggestion}"
                )

        return True

    return validate_filters_impl

def validate_metric_sql_binding(metric_manifest, sql_blocks):
    """
    Ensures metric → SQL references are valid.

    Deterministic policy: validate references only.
    - Hard-fail if the manifest references SQL blocks that do not exist
    - Do NOT fail if there are extra SQL blocks not referenced by the manifest
    """
    declared_sql_indices = set()
    for m in metric_manifest:
        for idx in m.get("sql_blocks", []):
            declared_sql_indices.add(idx)

    extracted_sql_indices = {q["metric_index"] for q in sql_blocks}

    missing_sql = declared_sql_indices - extracted_sql_indices

    if missing_sql:
        logger.error(
            "Metric references missing SQL blocks | missing=%s",
            sorted(missing_sql)
        )
        raise RuntimeError(
            f"Metric manifest references missing SQL blocks: {sorted(missing_sql)}"
        )

    logger.info(
        "Metric → SQL references validated | metrics=%d | sql_blocks=%d",
        len(metric_manifest),
        len(sql_blocks)
    )

def build_metric_payload(metric_manifest, sql_results):
    """
    Groups executed SQL results by metric_id based on the manifest.
    """
    results_by_index = {
        r["metric_index"]: r for r in sql_results
    }

    metric_payload = []

    for m in metric_manifest:
        metric_results = []

        for idx in m["sql_blocks"]:
            if idx not in results_by_index:
                metric_id = m.get("metric_id", "<unknown>")
                raise RuntimeError(
                    f"Missing executed SQL result for metric_id={metric_id} sql_block={idx}"
                )
            metric_results.append(results_by_index[idx])

        metric_payload.append({
            "metric_id": m["metric_id"],
            "metric_name": m.get("metric_name"),
            "definition": m.get("definition"),
            "business_question": m.get("business_question"),
            "results": metric_results
        })

    return metric_payload

# --------------------------------------------------
# SQL extraction (MULTI)
# --------------------------------------------------
SQL_READ_ONLY_RE = re.compile(
    r"^\s*(with\b[\s\S]+?\bselect\b|\(?\s*select\b)",
    re.IGNORECASE
)

FORBIDDEN_SQL = re.compile(
    r"\b("
    # DDL/DML
    r"insert|update|delete|merge|create|drop|alter|truncate|replace|rename|"
    # Privilege / procedure
    r"grant|revoke|call|"
    # BigQuery scripting / dynamic SQL (not allowed in read-only flow)
    r"begin|end|declare|set|commit|rollback|execute\s+immediate|execute"
    r")\b",
    re.IGNORECASE
)

def validate_read_only_sql(sql: str, index: int):
    """
    Strict read-only validation:
    - Single statement only (no internal semicolons)
    - Must start with SELECT/WITH (after stripping comments)
    - Must not contain DDL/DML/scripting keywords
    """
    clean_sql = strip_sql_comments(sql).strip()

    # Allow a single trailing semicolon, disallow any internal semicolons.
    if clean_sql.endswith(";"):
        clean_sql = clean_sql[:-1].rstrip()
    if ";" in clean_sql:
        raise RuntimeError(f"Multiple statements are not allowed (query #{index})")

    if not SQL_READ_ONLY_RE.match(clean_sql):
        raise RuntimeError(f"SQL must start with SELECT/WITH (query #{index})")

    match = FORBIDDEN_SQL.search(clean_sql)
    if match:
        logger.warning(
            "Rejected SQL | metric=%s | keyword=%s | sql=%s",
            index,
            match.group(0),
            sql.replace("\n", " ")
        )
        raise RuntimeError(f"Non read-only SQL in query #{index}")

def extract_all_sql(text: str):
    """
    Extracts sequential SQL blocks.

    Deterministic policy: fail fast on gaps/duplicates.
    - If blocks exist, they must be numbered 1..N with no gaps and no duplicates.
    - Each BEGIN_SQL_BLOCK_i must have a matching END_SQL_BLOCK_i.
    """
    begin_matches = re.findall(r"\bBEGIN_SQL_BLOCK_(\d+)\b", text)
    if not begin_matches:
        raise RuntimeError("No SQL blocks found")

    indices = [int(x) for x in begin_matches]
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"Duplicate SQL block numbers detected: {sorted(indices)}")

    present = sorted(indices)
    if present[0] != 1:
        raise RuntimeError(f"SQL blocks must start at 1; found {present[0]}")

    expected = list(range(1, present[-1] + 1))
    if present != expected:
        missing = sorted(set(expected) - set(present))
        raise RuntimeError(
            f"SQL block numbering must be sequential; missing blocks: {missing}"
        )

    sql_blocks = []
    for i in expected:
        start = f"BEGIN_SQL_BLOCK_{i}"
        end = f"END_SQL_BLOCK_{i}"

        if text.count(start) != 1 or text.count(end) != 1:
            raise RuntimeError(f"Invalid SQL block markers for block #{i}")

        m = re.search(
            rf"\b{re.escape(start)}\b\s*(.*?)\s*\b{re.escape(end)}\b",
            text,
            flags=re.DOTALL
        )
        if not m:
            raise RuntimeError(f"Invalid SQL block #{i}")

        block = m.group(1).strip()
        validate_read_only_sql(block, i)
        sql_blocks.append({"metric_index": i, "sql": block})

    return sql_blocks


def extract_all_output_schema(text: str) -> dict[int, list[dict]]:
    """
    Extract OUTPUT_SCHEMA blocks (mandatory).

    Deterministic policy: fail fast on gaps/duplicates.
    - If any BEGIN_OUTPUT_SCHEMA_n exists, blocks must be numbered 1..N.
    - Each BEGIN_OUTPUT_SCHEMA_n must have a matching END_OUTPUT_SCHEMA_n.
    - Each schema must be a JSON array of field descriptors.
    """
    begin_matches = re.findall(r"\bBEGIN_OUTPUT_SCHEMA_(\d+)\b", text)
    if not begin_matches:
        raise RuntimeError("Planner returned no OUTPUT_SCHEMA blocks")

    indices = [int(x) for x in begin_matches]
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"Duplicate OUTPUT_SCHEMA block numbers detected: {sorted(indices)}")

    present = sorted(indices)
    if present[0] != 1:
        raise RuntimeError(f"OUTPUT_SCHEMA blocks must start at 1; found {present[0]}")

    expected = list(range(1, present[-1] + 1))
    if present != expected:
        missing = sorted(set(expected) - set(present))
        raise RuntimeError(
            f"OUTPUT_SCHEMA block numbering must be sequential; missing blocks: {missing}"
        )

    out: dict[int, list[dict]] = {}

    for i in expected:
        start = f"BEGIN_OUTPUT_SCHEMA_{i}"
        end = f"END_OUTPUT_SCHEMA_{i}"

        if text.count(start) != 1 or text.count(end) != 1:
            raise RuntimeError(f"Invalid OUTPUT_SCHEMA markers for block #{i}")

        m = re.search(
            rf"\b{re.escape(start)}\b\s*(.*?)\s*\b{re.escape(end)}\b",
            text,
            flags=re.DOTALL
        )
        if not m:
            raise RuntimeError(f"Invalid OUTPUT_SCHEMA block #{i}")

        block = m.group(1).strip()
        if not block.startswith("[") or not block.endswith("]"):
            raise RuntimeError(f"OUTPUT_SCHEMA_{i} must be a JSON array")
        try:
            schema = json.loads(block)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid OUTPUT_SCHEMA_{i} JSON at pos={e.pos}: {e.msg}"
            ) from e
        if not isinstance(schema, list):
            raise RuntimeError(f"OUTPUT_SCHEMA_{i} must be a JSON array")
        out[i] = schema

    return out


def validate_output_schema(output_schema: list[dict], index: int):
    """
    Validate OUTPUT_SCHEMA for one SQL block.

    Mandatory + fail-fast policy:
    - Required keys and allowed values are enforced
    - Must align with domain metadata for known columns (role/time)
    """
    if not isinstance(output_schema, list) or not output_schema:
        raise RuntimeError(f"OUTPUT_SCHEMA_{index} must be a non-empty JSON array")

    seen = set()

    for item in output_schema:
        if not isinstance(item, dict):
            raise RuntimeError(f"OUTPUT_SCHEMA_{index} items must be objects")

        name = item.get("name")
        role = item.get("role")
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError(f"OUTPUT_SCHEMA_{index} field.name must be a non-empty string")
        name = name.strip()
        if name.lower() in seen:
            raise RuntimeError(f"OUTPUT_SCHEMA_{index} has duplicate field name: {name}")
        seen.add(name.lower())

        if role not in ("dimension", "kpi"):
            raise RuntimeError(
                f"OUTPUT_SCHEMA_{index} field.role must be 'dimension' or 'kpi' (field={name})"
            )

        dimension_type = item.get("dimension_type")
        time_level = item.get("time_level")

        if role == "kpi":
            if "dimension_type" in item or "time_level" in item:
                raise RuntimeError(
                    f"OUTPUT_SCHEMA_{index} KPI field must not include dimension_type/time_level (field={name})"
                )
        else:
            if dimension_type not in ("time", "categorical"):
                raise RuntimeError(
                    f"OUTPUT_SCHEMA_{index} dimension_type must be 'time' or 'categorical' (field={name})"
                )
            if dimension_type == "time":
                if time_level not in ("year", "week", "date", "half_hour", "program_airing"):
                    raise RuntimeError(
                        f"OUTPUT_SCHEMA_{index} time_level invalid (field={name})"
                    )
            else:
                if "time_level" in item:
                    raise RuntimeError(
                        f"OUTPUT_SCHEMA_{index} categorical dimension must not include time_level (field={name})"
                    )

        # Domain metadata alignment checks (only when column is known)
        candidates = DOMAIN_COLUMN_INDEX.get(name.lower(), [])
        if candidates:
            meta_roles = {c.get("role") for c in candidates if c.get("role")}
            if role == "kpi" and "dimension" in meta_roles:
                raise RuntimeError(
                    f"OUTPUT_SCHEMA_{index} role mismatch vs domain meta: {name} declared kpi but meta dimension"
                )
            if role == "dimension" and "kpi" in meta_roles:
                raise RuntimeError(
                    f"OUTPUT_SCHEMA_{index} role mismatch vs domain meta: {name} declared dimension but meta kpi"
                )

        # Time metadata alignment checks (only when time column is known)
        known_time_level = DOMAIN_TIME_INDEX.get(name.lower())
        if role == "dimension" and dimension_type == "time":
            if not known_time_level:
                raise RuntimeError(
                    f"OUTPUT_SCHEMA_{index} time dimension not recognized in domain time hierarchy: {name}"
                )
            if time_level != known_time_level:
                raise RuntimeError(
                    f"OUTPUT_SCHEMA_{index} time_level mismatch for {name}: declared={time_level}, meta={known_time_level}"
                )


def _field_info_from_output_schema(
    *,
    output_schema: list[dict],
    bq_schema: list[dict],
) -> list[dict]:
    """
    Build charting field_info from the declared OUTPUT_SCHEMA, enriched with
    BigQuery types and any available domain metadata.
    """
    bq_by_lower = {
        (f.get("name") or "").lower(): f for f in (bq_schema or []) if isinstance(f, dict) and f.get("name")
    }

    field_info: list[dict] = []
    for item in output_schema:
        name = (item.get("name") or "").strip()
        role = item.get("role")
        dimension_type = item.get("dimension_type")
        time_level = item.get("time_level")

        bq = bq_by_lower.get(name.lower(), {})
        meta_candidates = DOMAIN_COLUMN_INDEX.get(name.lower(), [])
        best = meta_candidates[0] if meta_candidates else {}

        field_info.append(
            {
                "name": name,
                "bq_type": (bq or {}).get("type"),
                "role": role,
                "dimension_type": dimension_type if role == "dimension" else None,
                "time_level": time_level if role == "dimension" and dimension_type == "time" else None,
                "semantic_tags": best.get("semantic_tags", []),
                "business_description": best.get("business_description"),
            }
        )

    return field_info

def strip_sql_comments(sql: str) -> str:
    """
    Remove SQL comments deterministically while preserving comment-like tokens
    inside string/identifier literals.

    Supports:
    - Line comments: -- ... <newline>
    - Block comments: /* ... */

    Respects:
    - Single-quoted strings: '...'
      (handles escaped quotes via doubled single quote: '')
    - Backtick identifiers: `...`
    - Double-quoted identifiers/strings: "..."
      (best-effort; BigQuery typically uses backticks for identifiers)
    """
    out: list[str] = []
    i = 0
    n = len(sql)

    in_single = False
    in_double = False
    in_backtick = False

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        # Toggle literal states (only when not inside another literal type)
        if not in_double and not in_backtick and ch == "'":
            if in_single:
                # Handle escaped single quote: ''
                if nxt == "'":
                    out.append("''")
                    i += 2
                    continue
                in_single = False
                out.append(ch)
                i += 1
                continue
            in_single = True
            out.append(ch)
            i += 1
            continue

        if not in_single and not in_backtick and ch == '"':
            in_double = not in_double
            out.append(ch)
            i += 1
            continue

        if not in_single and not in_double and ch == "`":
            in_backtick = not in_backtick
            out.append(ch)
            i += 1
            continue

        # Comment removal only when not inside a literal
        if not (in_single or in_double or in_backtick):
            # Line comment
            if ch == "-" and nxt == "-":
                # Skip until newline (keep newline if present)
                i += 2
                while i < n and sql[i] != "\n":
                    i += 1
                continue

            # Block comment
            if ch == "/" and nxt == "*":
                i += 2
                while i < n:
                    if sql[i] == "*" and (i + 1) < n and sql[i + 1] == "/":
                        i += 2
                        break
                    i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


# --------------------------------------------------
# BigQuery execution
# --------------------------------------------------
def _build_domain_column_index(domain_meta: dict) -> dict[str, list[dict]]:
    """
    Build a case-insensitive lookup from column name -> metadata entries.
    """
    index: dict[str, list[dict]] = {}

    for table_name, table_meta in domain_meta.items():
        if table_name in ("_meta", "global"):
            continue
        if not isinstance(table_meta, dict):
            continue
        cols = table_meta.get("columns")
        if not isinstance(cols, dict):
            continue

        for col_name, col_meta in cols.items():
            if not isinstance(col_name, str) or not isinstance(col_meta, dict):
                continue
            key = col_name.lower()
            index.setdefault(key, []).append(
                {
                    "table": table_name,
                    "name": col_name,
                    "role": col_meta.get("role"),
                    "data_type": col_meta.get("dataType"),
                    "business_description": col_meta.get("business_description"),
                    "semantic_tags": col_meta.get("semanticTags") or [],
                }
            )

    return index


def _build_time_column_index(domain_meta: dict) -> dict[str, str]:
    """
    Build a case-insensitive lookup from time column -> time level (year/week/date/...).
    """
    out: dict[str, str] = {}
    levels = (
        domain_meta.get("global", {})
        .get("time_hierarchy", {})
        .get("levels", {})
    )
    if not isinstance(levels, dict):
        return out

    for level, info in levels.items():
        cols = (info or {}).get("columns", [])
        if not isinstance(cols, list):
            continue
        for c in cols:
            if isinstance(c, str):
                out[c.lower()] = level
    return out


DOMAIN_COLUMN_INDEX = _build_domain_column_index(DOMAIN_META)
DOMAIN_TIME_INDEX = _build_time_column_index(DOMAIN_META)


def _bq_schema_to_json(schema) -> list[dict]:
    """
    Convert BigQuery SchemaField objects to JSON-serializable dicts.
    """
    if not schema:
        return []
    return [
        {
            "name": f.name,
            "type": getattr(f, "field_type", None),
            "mode": getattr(f, "mode", None),
            "description": getattr(f, "description", None),
        }
        for f in schema
    ]


def _json_safe_value(v):
    """
    Convert common BigQuery value types to JSON-serializable primitives.
    """
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, Decimal):
        # Deterministic numeric representation for JSON encoding.
        return float(v)
    if isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
        return v.isoformat()
    if isinstance(v, bytes):
        return base64.b64encode(v).decode("ascii")
    if isinstance(v, dict):
        return {k: _json_safe_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe_value(x) for x in v]
    return str(v)


def _normalize_bq_rows(rows: list[dict]) -> list[dict]:
    """
    Ensure rows are safe for JSON serialization and stable downstream handling.
    """
    return [{k: _json_safe_value(v) for k, v in r.items()} for r in rows]


def run_all_bigquery(queries):
    results = []

    for q in queries:
        metric_index = q.get("metric_index")
        sql = q.get("sql", "")
        output_schema = q.get("output_schema")
        try:
            if not isinstance(output_schema, list) or not output_schema:
                raise RuntimeError(f"Missing OUTPUT_SCHEMA for query #{metric_index}")

            dry_cfg = bigquery.QueryJobConfig(
                dry_run=True,
                use_query_cache=False,
                maximum_bytes_billed=MAX_BQ_BYTES,
            )
            dry_job = bq_client.query(sql, job_config=dry_cfg, location=BQ_REGION)

            # BigQuery-enforced check: only SELECT statements are allowed.
            stmt_type = getattr(dry_job, "statement_type", None)
            if stmt_type is not None and stmt_type.upper() != "SELECT":
                raise RuntimeError(
                    f"Query #{metric_index} is not read-only (statement_type={stmt_type})"
                )

            if dry_job.total_bytes_processed > MAX_BQ_BYTES:
                raise RuntimeError(
                    f"Query #{metric_index} exceeds byte limit "
                    f"({dry_job.total_bytes_processed} > {MAX_BQ_BYTES})"
                )

            exec_cfg = bigquery.QueryJobConfig(
                use_query_cache=False,
                maximum_bytes_billed=MAX_BQ_BYTES,
            )
            job = bq_client.query(sql, job_config=exec_cfg, location=BQ_REGION)
            result_iter = job.result(max_results=MAX_ROWS)
            rows = [dict(row) for row in result_iter]
            rows = _normalize_bq_rows(rows)
            # Prefer RowIterator schema (more reliable), fall back to job.schema.
            schema_fields = getattr(result_iter, "schema", None) or job.schema
            schema = _bq_schema_to_json(schema_fields)

            # Validate declared output schema against actual BigQuery output.
            declared_names = [x.get("name") for x in output_schema if isinstance(x, dict)]
            if any(not isinstance(n, str) or not n.strip() for n in declared_names):
                raise RuntimeError(f"Invalid OUTPUT_SCHEMA field names for query #{metric_index}")
            declared_names = [n.strip() for n in declared_names]  # type: ignore[assignment]

            actual_names = [f.get("name") for f in schema if isinstance(f, dict) and f.get("name")]
            if not actual_names and rows:
                # Last-resort fallback: infer from row keys (preserves row/dict order).
                actual_names = list(rows[0].keys())
                schema = [{"name": n, "type": None, "mode": None, "description": None} for n in actual_names]

            if not actual_names:
                raise RuntimeError(
                    f"BigQuery returned no output schema columns for query #{metric_index}"
                )

            declared_lower = [n.lower() for n in declared_names]
            actual_lower = [n.lower() for n in actual_names]
            if set(declared_lower) != set(actual_lower):
                raise RuntimeError(
                    f"OUTPUT_SCHEMA mismatch for query #{metric_index}: "
                    f"declared={declared_names} actual={actual_names}"
                )
            if declared_lower != actual_lower:
                raise RuntimeError(
                    f"OUTPUT_SCHEMA column order mismatch for query #{metric_index}: "
                    f"declared={declared_names} actual={actual_names}"
                )

            field_info = _field_info_from_output_schema(
                output_schema=output_schema,
                bq_schema=schema,
            )

            results.append({
                "metric_index": metric_index,
                "sql": sql,
                "rows": rows,
                "schema": schema,
                "field_info": field_info,
                "output_schema": output_schema,
            })
        except Exception as e:
            logger.exception("BigQuery execution failed | metric_index=%s", metric_index)
            raise RuntimeError(
                f"BigQuery execution failed for query #{metric_index}: {e}"
            ) from e

    return results

        
# --------------------------------------------------
# UI Template
# --------------------------------------------------
HTML = """
<!doctype html>
<html>
<head>
<title>Network18 Ask Me Analytics</title>

<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap">
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>

<style>
body { font-family: Inter, system-ui, sans-serif; margin: 40px; }
.container { max-width: 900px; }
details summary { cursor:pointer; font-weight:600; font-size: 1rem; margin: 16px 0 8px 0; list-style: none; }
details summary::-webkit-details-marker { display: none; }
details summary::marker { display: none; }
details summary::before { content:"▶ "; font-size: 0.75em; margin-right: 6px; color: #666; }
details[open] summary::before { content:"▼ "; }
.details-body p { margin: 6px 0; line-height: 1.55; }
.details-body h1,
.details-body h2,
.details-body h3 { font-size: 1.05rem; font-weight: 600; margin: 16px 0 6px 0; }
.details-body p > strong:only-child { display: block; font-weight: 600; margin: 14px 0 4px 0; }
.details-body ul,
.details-body ol { margin: 6px 0 6px 1.2em; }
.details-body li { margin-bottom: 4px; line-height: 1.5; }

details[open] .markdown-body p { margin: 8px 0; line-height: 1.0; }
pre { background:#f8f8f8; padding:12px; white-space:pre-wrap; }
.error { color:darkred; font-weight:600; }
</style>
</head>

<body>
<h2>Network18 Ask Me Analytics</h2>

<form id="askForm" method="post" onsubmit="handleSubmit(event)">
  <input name="question" required style="width:700px;padding:6px">
  <button type="submit">Ask</button>
</form>

<div id="results" class="container">
{% if error %}<p class="error">{{ error }}</p>{% endif %}
{% if answer %}
<hr>

<details open>
  <summary>Question</summary>
  <h3>{{ question }}</h3>
</details>

<details>
  <summary>Metrics & Analysis Planner</summary>
  <pre>{{ planner_text }}</pre>
</details>

{% if filters %}
<details open>
  <summary>Filters</summary>
  <ul>
    {% for f in filters %}
      <li>{{ f }}</li>
    {% endfor %}
  </ul>
</details>
{% endif %}

<details open>
  <summary>Answer</summary>
  <h3>{{ answer.headline }}</h3>
</details>

<details open>
  <summary>Key Takeaways</summary>
  <ul>{% for t in answer.takeaways %}<li>{{ t }}</li>{% endfor %}</ul>
</details>

<details open>
  <summary>Details</summary>
  <div class="markdown-body details-body">
    {{ answer.details | markdown | safe }}
  </div>
</details>

{% for metric in answer.metrics %}
<h4>{{ metric.get("metric_name") or metric.metric_id }}</h4>

{% if metric.definition %}
<p><em>{{ metric.definition }}</em></p>
{% endif %}

{% for r in metric.results %}
  {% if r.visuals %}
    <details open>
      <summary>Visualization</summary>
  
      {% for v in r.visuals %}
        <div id="chart-{{ metric.metric_id }}-{{ loop.index0 }}-{{ loop.index }}"></div>
        <script>
          vegaEmbed(
            "#chart-{{ metric.metric_id }}-{{ loop.index0 }}-{{ loop.index }}",
            {{ v.spec | tojson }}
          );
        </script>
      {% endfor %}
  
    </details>
  {% else %}
  <details>
    <summary>Results (Table)</summary>
    <pre>{{ r.rows }}</pre>
  </details>
  {% endif %}

  <details>
    <summary>Executed Query</summary>
    <pre>{{ r.sql }}</pre>
  </details>
{% endfor %}

{% endfor %}



{% endif %}
</div>

<script>
let countdownInterval;

function handleSubmit(e) {
  e.preventDefault(); // ⬅️ REQUIRED

  let seconds = 99;

  document.getElementById("results").innerHTML = `
    <p><em>Running analysis…</em></p>
    <p id="countdown"><small>~${seconds}s remaining</small></p>
  `;

  countdownInterval = setInterval(() => {
    seconds--;

    if (seconds <= 0) {
      document.getElementById("countdown").innerHTML =
        "<small>Still working…</small>";
      clearInterval(countdownInterval);
      return;
    }

    document.getElementById("countdown").innerHTML =
      `<small>~${seconds}s remaining</small>`;
  }, 1000);

  // ⬇️ allow browser to paint, then submit
  setTimeout(() => {
    document.getElementById("askForm").submit();
  }, 50);
}
</script>


</body>
</html>
"""

# --------------------------------------------------
# Route
# --------------------------------------------------
@app.route("/", methods=["GET","POST"])
def index():

    filters = []
    planner_text = ""
    answer = {}
    error = None
    question = None
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    logger.info(
        "(%s) Request started | request_id=%s | limits={max_bq_bytes:%d,max_turns:%d,max_rows:%d}",
        request.method,
        request_id,
        MAX_BQ_BYTES,
        MAX_TURNS,
        MAX_ROWS
    )

    if request.method == "POST":
        try:
            question = request.form["question"]  # ✅ FIX

            # Lazily initialize clients/models so the container can start even if
            # credentials aren't available during import time.
            bq_client = get_bq_client()
            planner_model, interpreter_model = get_models()

            default_rows = fetch_default_dimension_rows(
                bq_client=bq_client,
                limit=(
                    None
                    if os.getenv("DEFAULT_DIMENSIONS_LIMIT", "100").strip().lower()
                    in {"infinite", "all", "none", "unlimited", "0", "-1"}
                    else int(os.getenv("DEFAULT_DIMENSIONS_LIMIT", "100"))
                ),
            )
            candidates = fetch_candidate_dimension_rows_for_question(
                bq_client=bq_client,
                question=question,
                limit=int(os.getenv("DIMENSION_CANDIDATES_LIMIT", "200")),
            )
            allowed_rows = merge_dimension_rows(candidates, default_rows)
            selected_default = pick_selected_default_row(
                question=question,
                default_rows=(candidates or default_rows),
            )

            result = run_analysis(
                question=question,
                session=session,
                request_id=request_id,
                call_planner=call_planner,
                planner_ctx={
                    "system_prompt": SYSTEM_PROMPT,
                    "planner_prompt": PLANNER_PROMPT,
                    "metadata": META_DATA,
                    "allowed_dimension_rows": allowed_rows,
                    "selected_default_dimensions": selected_default,
                    "model": planner_model,
                },
                extract_metric_manifest=extract_metric_manifest,
                extract_all_sql=extract_all_sql,
                extract_all_output_schema=extract_all_output_schema,
                extract_filters=extract_filters,
                validate_filters=make_validate_filters(
                    allowed_rows=allowed_rows,
                    candidates=candidates,
                    question=question,
                    selected_default_dimensions=selected_default,
                ),
                validate_output_schema=validate_output_schema,
                validate_metric_sql_binding=validate_metric_sql_binding,
                run_all_bigquery=run_all_bigquery,
                build_metric_payload=build_metric_payload,
                call_interpreter=call_interpreter,
                interpreter_ctx={
                    "system_prompt": SYSTEM_PROMPT,
                    "interpreter_prompt": INTERPRETER_PROMPT,
                    "answer_contract": ANSWER_CONTRACT,
                    "model": interpreter_model,
                },
                get_chat_history=get_chat_history,
                append_chat_turn=append_chat_turn
            )
            
            planner_text = result["planner_text"]
            filters = result["filters"]
            resolved_filters = result.get("resolved_filters") or {}

            resolved_genre = (resolved_filters.get("genre") or {}).get("value")
            resolved_region = (resolved_filters.get("region") or {}).get("value")
            resolved_target = (resolved_filters.get("target") or {}).get("value")
            resolved_channel = (resolved_filters.get("channel") or {}).get("value")
            
            shadow_dims = shadow_resolve_dimensions_bq(
                bq_client=bq_client,
                genre=resolved_genre,
                region=resolved_region,
                target=resolved_target,
                channel=resolved_channel
            )
            
            logger.info(
                "SHADOW_DIMENSION_COMPARE | planner=%s | db=%s | match=%s",
                {
                    "genre": resolved_genre,
                    "region": resolved_region,
                    "target": resolved_target,
                    "channel": resolved_channel,
                },
                shadow_dims,
                (
                    shadow_dims.get("genre") == resolved_genre and
                    shadow_dims.get("region") == resolved_region and
                    shadow_dims.get("target") == resolved_target
                )
            )

            core_answer = result["validated_answer"]
            metric_payload = result["metric_payload"]

#            answer = build_ui_answer(core_answer, metric_payload)
            answer = build_ui_answer(core_answer, metric_payload, DOMAIN_META)

        except Exception as e:
            error = str(e)
            logger.exception("Request failed | request_id=%s", request_id)

    return render_template_string(
        HTML,
        question=question,
        planner_text=planner_text,
        filters=filters,
        answer=answer,
        error=error
    )


@app.route("/_debug/logs", methods=["GET"])
def debug_logs():
    """
    Debug helper for retrieving recent logs without Cloud Logging export.

    Enabled only when:
    - ENABLE_LOG_BUFFER=1 (to collect logs)
    - DEBUG_TOKEN is set and provided as ?token=...
    """
    expected = os.getenv("DEBUG_TOKEN")
    token = request.args.get("token")
    if not expected or token != expected:
        return {"error": "forbidden"}, 403

    buf = get_log_buffer()
    if not buf:
        return {"error": "log buffer disabled (set ENABLE_LOG_BUFFER=1)"}, 400

    rid = request.args.get("request_id")
    try:
        limit = int(request.args.get("limit", "500"))
    except ValueError:
        limit = 500

    return {
        "request_id": rid,
        "limit": limit,
        "logs": buf.get(request_id=rid, limit=limit),
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "0").strip() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
