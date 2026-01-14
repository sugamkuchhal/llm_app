import logging

logger = logging.getLogger("logger")


# -----------------------------
# Helpers: time resolution
# -----------------------------

def resolve_time_column(columns, time_hierarchy):
    """
    Returns (time_level, column_name) or (None, None)
    """
    levels = time_hierarchy["levels"]
    ordering = time_hierarchy["ordering"]

    present_levels = []

    for level, info in levels.items():
        for col in info["columns"]:
            if col in columns:
                present_levels.append(level)
                break

    if not present_levels:
        return None, None

    # pick most granular (last in ordering)
    for level in reversed(ordering):
        if level in present_levels:
            for col in levels[level]["columns"]:
                if col in columns:
                    return level, col

    return None, None


# -----------------------------
# Helpers: roles
# -----------------------------

def resolve_roles(columns, domain_meta):
    dimensions = []
    kpis = []

    tables = domain_meta.get("tables", {})

    for table in tables.values():
        for col, meta in table.get("columns", {}).items():
            if col not in columns:
                continue
            if meta.get("role") == "dimension":
                dimensions.append(col)
            elif meta.get("role") == "kpi":
                kpis.append(col)

    return dimensions, kpis


# -----------------------------
# Vega helpers
# -----------------------------

def vega_text(message):
    return {
        "type": "vega",
        "spec": {
            "data": {"values": [{}]},
            "mark": "text",
            "encoding": {
                "text": {"value": message}
            }
        }
    }


def vega_table(rows):
    if not rows:
        return None

    first_col = list(rows[0].keys())[0]

    return {
        "type": "vega",
        "spec": {
            "data": {"values": rows},
            "mark": "text",
            "encoding": {
                "row": {"field": first_col, "type": "nominal"},
                "text": {"field": first_col, "type": "nominal"}
            }
        }
    }


def vega_line(rows, x, y, color=None):
    enc = {
        "x": {"field": x, "type": "ordinal"},
        "y": {"field": y, "type": "quantitative"}
    }
    if color:
        enc["color"] = {"field": color, "type": "nominal"}

    return {
        "type": "vega",
        "spec": {
            "data": {"values": rows},
            "mark": "line",
            "encoding": enc
        }
    }


def vega_bar(rows, x, y, color=None):
    enc = {
        "x": {"field": x, "type": "nominal"},
        "y": {"field": y, "type": "quantitative"}
    }
    if color:
        enc["color"] = {"field": color, "type": "nominal"}

    return {
        "type": "vega",
        "spec": {
            "data": {"values": rows},
            "mark": "bar",
            "encoding": enc
        }
    }


# -----------------------------
# Core chart builder
# -----------------------------

def build_charts(rows, domain_meta, expected_measures):
    if not rows:
        return [vega_text("No data returned")]

    columns = set(rows[0].keys())

    # time
    time_level, time_col = resolve_time_column(
        columns,
        domain_meta["global"]["time_hierarchy"]
    )

    # roles
    dimensions, kpis = resolve_roles(columns, domain_meta)

    non_time_dims = [d for d in dimensions if d != time_col]

    if not kpis and expected_measures:
        for col in columns:
            if col in expected_measures:
                # keep it simple: trust planner intent; rows already numeric for KPIs
                kpis.append(col)

    if not kpis:
        logger.info("Chart rejected: no KPI columns")
        return [vega_text("No KPI columns found"), vega_table(rows[:5])]

    visuals = []

    for kpi in kpis:
        if time_col:
            if len(non_time_dims) > 1:
                logger.info("Chart rejected: too many dimensions for time series")
                return [
                    vega_text("Too many dimensions for time-series chart"),
                    vega_table(rows[:5])
                ]

            color = non_time_dims[0] if non_time_dims else None
            visuals.append(vega_line(rows, time_col, kpi, color))

        else:
            if len(non_time_dims) > 2:
                logger.info("Chart rejected: too many dimensions for bar chart")
                return [
                    vega_text("Too many dimensions for bar chart"),
                    vega_table(rows[:5])
                ]

            if len(non_time_dims) == 2:
                visuals.append(
                    vega_bar(rows, non_time_dims[0], kpi, non_time_dims[1])
                )
            elif len(non_time_dims) == 1:
                visuals.append(
                    vega_bar(rows, non_time_dims[0], kpi)
                )
            else:
                visuals.append(
                    vega_text("Cannot chart KPI without dimensions")
                )

    return visuals


# -----------------------------
# Public API
# -----------------------------

def build_ui_answer(core_answer, metric_payload, domain_meta):
    ui_answer = dict(core_answer)
    ui_metrics = []

    for metric in metric_payload:
        results = []

        for r in metric["results"]:
            
            visuals = build_charts(
                r["rows"],
                domain_meta,
                metric.get("expected_measures", [])
            )


            results.append({
                "sql": r["sql"],
                "rows": r["rows"],
                "schema": r.get("schema", []),
                "visuals": visuals
            })

        ui_metrics.append({
            "metric_id": metric["metric_id"],
            "metric_name": metric.get("metric_name"),
            "definition": metric.get("definition"),
            "business_question": metric.get("business_question"),
            "results": results
        })

    ui_answer["metrics"] = ui_metrics
    return ui_answer
