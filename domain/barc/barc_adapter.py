"""
BARC domain adapter.

Step 1-3 separation-of-concerns:
- Provide a single import surface for BARC-specific behavior.
- Move domain assets loading here (prompts/metadata/contracts).
- Move dimension context fetching/merging/default selection here.
"""

import json
import os

from domain.barc.barc_dimension_reference import (
    fetch_default_dimension_rows,
    fetch_candidate_dimension_rows_for_question,
    merge_dimension_rows,
    pick_selected_default_row,
)
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
    sql_has_any_dim_predicate,
    sql_has_dim_filter,
)


def _repo_root() -> str:
    # domain/barc/barc_adapter.py -> /workspace
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _read_text(path_from_repo_root: str) -> str:
    """
    Deterministic file loader:
    - UTF-8 (strict)
    - stable newlines (normalize to \\n)
    """
    full_path = os.path.join(_repo_root(), path_from_repo_root)
    with open(full_path, "r", encoding="utf-8", errors="strict") as f:
        text = f.read()
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_domain_assets() -> dict:
    """
    Load all BARC domain assets needed by the app.
    """
    system_prompt = _read_text("prompt_system.txt")
    interpreter_prompt = _read_text("prompt_interpreter.txt")
    answer_contract = _read_text("contract_answer.json")
    planner_prompt = _read_text("domain/barc/barc_context_planner.txt")
    metadata_text = _read_text("domain/barc/barc_meta.json")

    try:
        domain_meta = json.loads(metadata_text)
    except json.JSONDecodeError as e:
        raise RuntimeError("Invalid domain metadata JSON (barc_meta.json)") from e

    return {
        "domain": "barc",
        "system_prompt": system_prompt,
        "planner_prompt": planner_prompt,
        "interpreter_prompt": interpreter_prompt,
        "answer_contract": answer_contract,
        "metadata_text": metadata_text,
        "domain_meta": domain_meta,
    }


def _parse_limit_env(var_name: str, default: str) -> int | None:
    raw = os.getenv(var_name, default).strip().lower()
    if raw in {"infinite", "all", "none", "unlimited", "0", "-1"}:
        return None
    return int(raw)


def build_dimension_context(*, bq_client, question: str) -> dict:
    """
    Fetch/merge dimension rows and select SYSTEM defaults for this question.

    Returns:
      - default_rows
      - candidates
      - allowed_rows
      - selected_default_dimensions
    """
    default_limit = _parse_limit_env("DEFAULT_DIMENSIONS_LIMIT", "100")
    candidates_limit = int(os.getenv("DIMENSION_CANDIDATES_LIMIT", "200"))

    default_rows = fetch_default_dimension_rows(
        bq_client=bq_client,
        limit=default_limit,
    )
    candidates = fetch_candidate_dimension_rows_for_question(
        bq_client=bq_client,
        question=question,
        limit=candidates_limit,
    )
    allowed_rows = merge_dimension_rows(candidates, default_rows)

    # Defaults must come strictly from is_default=TRUE curated rows.
    selected_default = pick_selected_default_row(question=question, default_rows=default_rows)

    return {
        "default_rows": default_rows,
        "candidates": candidates,
        "allowed_rows": allowed_rows,
        "selected_default_dimensions": selected_default,
    }

