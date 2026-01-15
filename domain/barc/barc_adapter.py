"""
BARC domain adapter.

Step 1 separation-of-concerns:
- Provide a single import surface for BARC-specific behavior.
- Keep behavior unchanged; this file mostly re-exports existing functions.
"""

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

