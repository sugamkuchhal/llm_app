# charts/builder.py

import re

from types import SimpleNamespace

TIME_GRANULARITY_ORDER = [
    "year", "quarter", "month", "week",
    "date", "day", "hour", "minute"
]


def infer_schema_from_rows(rows):
    schema = []

    for key in rows[0].keys():
        field_type = None

        # 🔧 scan until first non-null value
        for r in rows:
            v = r.get(key)
            if v is None:
                continue
            if isinstance(v, str):
                field_type = "STRING"
                break
            if isinstance(v, (int, float)):
                field_type = "FLOAT"
                break

        if field_type:
            schema.append(
                SimpleNamespace(
                    name=key,
                    field_type=field_type
                )
            )

    return schema


def is_time_like(field):
    name = field.name.lower()
    return any(
        token in name
        for token in ["year", "quarter", "month", "week", "date", "day", "hour", "minute"]
    )

def time_granularity_rank(field):
    name = field.name.lower()
    for i, token in enumerate(TIME_GRANULARITY_ORDER):
        if token in name:
            return i
    return len(TIME_GRANULARITY_ORDER)

def infer_unit(field_name):
    """
    Best-effort unit inference from field name.
    """
    lname = field_name.lower()
    if "percent" in lname or "share" in lname or "%" in lname:
        return "percent"
    if "ama" in lname or "000" in lname:
        return "thousands"
    return "number"

def _infer_unit_from_field_info(field: dict) -> str:
    name = (field.get("name") or "")
    tags = field.get("semantic_tags") or []
    if isinstance(tags, list) and any(isinstance(t, str) and "percent" in t.lower() for t in tags):
        return "percent"
    return infer_unit(name)


def _vega_type_for_field(field: dict, *, for_axis: bool = False) -> str:
    """
    Vega-Lite encoding type: nominal|ordinal|quantitative|temporal
    """
    role = field.get("role")
    bq_type = (field.get("bq_type") or "").upper()
    dimension_type = field.get("dimension_type")

    if role == "kpi":
        return "quantitative"

    if dimension_type == "time":
        # Only use temporal when the underlying type is truly date/time;
        # many "time" concepts (e.g., week_id) are strings/ints and should remain ordinal.
        if bq_type in ("DATE", "DATETIME", "TIMESTAMP", "TIME"):
            return "temporal"
        return "ordinal"

    # categorical dimension
    return "nominal"


def _title_for_field(field: dict) -> str:
    # Keep titles compact; descriptions may be long.
    return field.get("name") or ""


def _tooltip_fields(dim_fields: list[dict], kpi_fields: list[dict]) -> list[dict]:
    out = []
    for f in dim_fields + kpi_fields:
        out.append(
            {
                "field": f.get("name"),
                "type": _vega_type_for_field(f),
                "title": _title_for_field(f),
            }
        )
    return out


def build_bq_style_chart(rows, schema, *, field_info=None, domain_meta=None, top_n: int = 20):
    if not rows:
        return None

    # If field_info is provided (authoritative), use it for roles/types.
    if field_info:
        dim_fields = [f for f in field_info if isinstance(f, dict) and f.get("role") == "dimension"]
        kpi_fields = [f for f in field_info if isinstance(f, dict) and f.get("role") == "kpi"]

        # Pick best time dimension (most granular) if possible.
        time_order = []
        if domain_meta:
            time_order = (
                (domain_meta.get("global", {}) or {})
                .get("time_hierarchy", {})
                .get("ordering", [])
            ) or []
        rank = {lvl: i for i, lvl in enumerate(time_order)} if isinstance(time_order, list) else {}

        time_dims = [f for f in dim_fields if f.get("dimension_type") == "time"]
        cat_dims = [f for f in dim_fields if f.get("dimension_type") != "time"]
        time_dims_sorted = sorted(
            time_dims,
            key=lambda f: rank.get(f.get("time_level"), -1),
        )

        # Prefer most granular time dim (highest rank).
        time_dim = time_dims_sorted[-1] if time_dims_sorted else None

        # Deterministic dimension selection:
        # - if time exists: x=time, color=first categorical dim (optional)
        # - else: x=first categorical dim, color=second categorical dim (optional)
        if time_dim:
            dim_x = time_dim
            dim_color = cat_dims[0] if cat_dims else None
            chart_kind = "line"
        else:
            dim_x = cat_dims[0] if cat_dims else None
            dim_color = cat_dims[1] if len(cat_dims) > 1 else None
            chart_kind = "bar"

        if not kpi_fields:
            return {
                "type": "vega",
                "spec": {
                    "data": {"values": rows[:5]},
                    "mark": "text",
                    "encoding": {"text": {"value": "No KPI fields to chart"}},
                },
            }

        # Guardrail: too many dimensions to chart cleanly.
        dim_count = len(dim_fields)
        if time_dim and dim_count > (1 + (1 if dim_color else 0)):
            return {
                "type": "vega",
                "spec": {
                    "data": {"values": rows[:5]},
                    "mark": "text",
                    "encoding": {"text": {"value": "Too many dimensions for time series chart"}},
                },
            }
        if (not time_dim) and dim_count > (1 + (1 if dim_color else 0)):
            return {
                "type": "vega",
                "spec": {
                    "data": {"values": rows[:5]},
                    "mark": "text",
                    "encoding": {"text": {"value": "Too many dimensions for bar chart"}},
                },
            }

        # Create visuals
        tooltip = _tooltip_fields([x for x in [dim_x, dim_color] if x], kpi_fields)

        visuals = []

        # Special case: 1 categorical dimension + exactly 2 KPIs
        # If same unit -> grouped bars (fold), else dual-axis.
        if (not time_dim) and dim_x and len(kpi_fields) == 2 and not dim_color:
            f1, f2 = kpi_fields
            u1, u2 = _infer_unit_from_field_info(f1), _infer_unit_from_field_info(f2)

            if u1 == u2:
                visuals.append(
                    {
                        "type": "vega",
                        "spec": {
                            "data": {"values": rows},
                            "transform": [
                                {"fold": [f1["name"], f2["name"]], "as": ["metric", "value"]},
                                {"window": [{"op": "rank", "as": "_rank"}], "sort": [{"field": "value", "order": "descending"}]},
                                {"filter": f"datum._rank <= {int(top_n)}"},
                            ],
                            "mark": "bar",
                            "encoding": {
                                "x": {
                                    "field": dim_x["name"],
                                    "type": _vega_type_for_field(dim_x, for_axis=True),
                                    "axis": {"title": _title_for_field(dim_x)},
                                    "sort": {"field": "value", "order": "descending"},
                                },
                                "y": {"field": "value", "type": "quantitative", "axis": {"title": "Value"}},
                                "color": {"field": "metric", "type": "nominal", "legend": {"title": "Metric"}},
                                "tooltip": tooltip,
                            },
                        },
                    }
                )
                return visuals

            visuals.append(
                {
                    "type": "vega",
                    "spec": {
                        "data": {"values": rows},
                        "resolve": {"scale": {"y": "independent"}},
                        "layer": [
                            {
                                "mark": "bar",
                                "encoding": {
                                    "x": {
                                        "field": dim_x["name"],
                                        "type": _vega_type_for_field(dim_x, for_axis=True),
                                        "axis": {"title": _title_for_field(dim_x)},
                                    },
                                    "y": {"field": f1["name"], "type": "quantitative", "axis": {"title": f1["name"]}},
                                    "color": {"value": "#4c78a8"},
                                    "tooltip": tooltip,
                                },
                            },
                            {
                                "mark": "bar",
                                "encoding": {
                                    "x": {
                                        "field": dim_x["name"],
                                        "type": _vega_type_for_field(dim_x, for_axis=True),
                                        "axis": {"title": _title_for_field(dim_x)},
                                    },
                                    "y": {
                                        "field": f2["name"],
                                        "type": "quantitative",
                                        "axis": {"title": f2["name"], "orient": "right"},
                                    },
                                    "color": {"value": "#f58518"},
                                    "tooltip": tooltip,
                                },
                            },
                        ],
                    },
                }
            )
            return visuals

        for kpi in kpi_fields:
            if time_dim and dim_x:
                spec = {
                    "data": {"values": rows},
                    "mark": "line",
                    "encoding": {
                        "x": {
                            "field": dim_x["name"],
                            "type": _vega_type_for_field(dim_x, for_axis=True),
                            "axis": {"title": _title_for_field(dim_x)},
                        },
                        "y": {
                            "field": kpi["name"],
                            "type": "quantitative",
                            "axis": {"title": _title_for_field(kpi)},
                        },
                        "tooltip": tooltip,
                    },
                }
                if dim_color:
                    spec["encoding"]["color"] = {
                        "field": dim_color["name"],
                        "type": _vega_type_for_field(dim_color),
                        "legend": {"title": _title_for_field(dim_color)},
                    }
                visuals.append({"type": "vega", "spec": spec})
            elif dim_x:
                visuals.append(
                    {
                        "type": "vega",
                        "spec": {
                            "data": {"values": rows},
                            "transform": [
                                {"window": [{"op": "rank", "as": "_rank"}], "sort": [{"field": kpi["name"], "order": "descending"}]},
                                {"filter": f"datum._rank <= {int(top_n)}"},
                            ],
                            "mark": "bar",
                            "encoding": {
                                "x": {
                                    "field": dim_x["name"],
                                    "type": _vega_type_for_field(dim_x, for_axis=True),
                                    "axis": {"title": _title_for_field(dim_x)},
                                    "sort": {"field": kpi["name"], "order": "descending"},
                                },
                                "y": {
                                    "field": kpi["name"],
                                    "type": "quantitative",
                                    "axis": {"title": _title_for_field(kpi)},
                                },
                                "tooltip": tooltip,
                            },
                        },
                    }
                )
            else:
                # No dimensions: show KPI as a simple text fallback.
                visuals.append(
                    {
                        "type": "vega",
                        "spec": {
                            "data": {"values": rows[:1] if rows else [{}]},
                            "mark": "text",
                            "encoding": {"text": {"value": f"{kpi.get('name')}: {rows[0].get(kpi.get('name')) if rows else ''}"}},
                        },
                    }
                )

        return visuals

    # Fallback: existing heuristic behavior (schema-based) for callers not providing field_info.

    # 🔧 infer schema if missing
    if not schema:
        schema = infer_schema_from_rows(rows)

    # --------------------------------------------------
    # 1. Classify fields
    # --------------------------------------------------
    dimensions = []
    measures = []

    for field in schema:
        if field.field_type in ("STRING", "DATE", "TIMESTAMP"):
            dimensions.append(field)
        elif field.field_type in ("INTEGER", "FLOAT", "NUMERIC"):
            measures.append(field)

    # --------------------------------------------------
    # 2. Guardrails
    # --------------------------------------------------
    if len(dimensions) > 2:
        return None

    if len(measures) == 0:
        return None

    # --------------------------------------------------
    # 3. Identify time dimension
    # --------------------------------------------------
    time_dims = [d for d in dimensions if is_time_like(d)]
    
    # --------------------------------------------------
    # Dimension ordering rules
    # Time always owns X-axis
    # If both are time, higher granularity goes on X
    # --------------------------------------------------
    if len(dimensions) == 2:
        non_time_dims = [d for d in dimensions if not is_time_like(d)]
    
        # Case 1: one time + one non-time
        if len(time_dims) == 1 and len(non_time_dims) == 1:
            # time on X, category on legend
            dimensions = [time_dims[0], non_time_dims[0]]
    
        # Case 2: two time dimensions
        elif len(time_dims) == 2:
            t1, t2 = time_dims
            r1 = time_granularity_rank(t1)
            r2 = time_granularity_rank(t2)
    
            # same granularity → no chart
            if r1 == r2:
                return None
    
            # higher granularity (finer) on X-axis
            if r1 > r2:
                dimensions = [t1, t2]
            else:
                dimensions = [t2, t1]
        
    chart_type = "line" if time_dims else "bar"

    # --------------------------------------------------
    # 4. CASE: 2 DIMENSIONS → 1 KPI PER CHART
    # --------------------------------------------------
    if len(dimensions) == 2:
        dim_x = dimensions[0].name
        dim_legend = dimensions[1].name

        charts = []

        for measure in measures:
            charts.append({
                "type": "vega",
                "spec": {
                    "data": {"values": rows},
                    "mark": chart_type,
                    "encoding": {
                        "x": {
                            "field": dim_x,
                            "type": "nominal",
                            "axis": {"title": dim_x}
                        },
                        "y": {
                            "field": measure.name,
                            "type": "quantitative",
                            "axis": {"title": measure.name}
                        },
                        "color": {
                            "field": dim_legend,
                            "type": "nominal",
                            "legend": {"title": dim_legend}
                        }
                    }
                }
            })

        # UI will render multiple charts if list
        return charts

    # --------------------------------------------------
    # 5. CASE: 1 DIMENSION
    # --------------------------------------------------
    if len(dimensions) == 1:
        dim = dimensions[0].name

        # ---- 1 KPI
        if len(measures) == 1:
            m = measures[0]
            return {
                "type": "vega",
                "spec": {
                    "data": {"values": rows},
                    "mark": chart_type,
                    "encoding": {
                        "x": {
                            "field": dim,
                            "type": "nominal",
                            "axis": {"title": dim}
                        },
                        "y": {
                            "field": m.name,
                            "type": "quantitative",
                            "axis": {"title": m.name}
                        },
                        "color": {
                            "value": "#4c78a8"
                        }
                    }
                }
            }

        # ---- 2 KPIs
        if len(measures) == 2:
            m1, m2 = measures
            u1, u2 = infer_unit(m1.name), infer_unit(m2.name)

            # Same unit → grouped bars
            if u1 == u2:
                return {
                    "type": "vega",
                    "spec": {
                        "data": {"values": rows},
                        "transform": [
                            {
                                "fold": [m1.name, m2.name],
                                "as": ["metric", "value"]
                            }
                        ],
                        "mark": chart_type,
                        "encoding": {
                            "x": {
                                "field": dim,
                                "type": "nominal",
                                "axis": {"title": dim}
                            },
                            "y": {
                                "field": "value",
                                "type": "quantitative"
                            },
                            "color": {
                                "field": "metric",
                                "type": "nominal",
                                "legend": {"title": "Metric"}
                            }
                        }
                    }
                }

            # Different units → dual axis
            return {
                "type": "vega",
                "spec": {
                    "data": {"values": rows},
            
                    "resolve": {
                        "scale": {
                            "y": "independent"
                        }
                    },
            
                    "layer": [
                        {
                            "mark": chart_type,
                            "encoding": {
                                "x": {"field": dim, "type": "nominal"},
                                "y": {
                                    "field": m1.name,
                                    "type": "quantitative",
                                    "axis": {"title": m1.name}
                                },
                                "color": {"value": "#4c78a8"}
                            }
                        },
                        {
                            "mark": chart_type,
                            "encoding": {
                                "x": {"field": dim, "type": "nominal"},
                                "y": {
                                    "field": m2.name,
                                    "type": "quantitative",
                                    "axis": {"title": m2.name, "orient": "right"}
                                },
                                "color": {"value": "#f58518"}
                            }
                        }
                    ]
                }
            }

        # ---- >2 KPIs → multiple charts
        charts = []
        for m in measures:
            charts.append({
                "type": "vega",
                "spec": {
                    "data": {"values": rows},
                    "mark": chart_type,
                    "encoding": {
                        "x": {
                            "field": dim,
                            "type": "nominal"
                        },
                        "y": {
                            "field": m.name,
                            "type": "quantitative",
                            "axis": {"title": m.name}
                        },
                        "color": {"value": "#4c78a8"}
                    }
                }
            })
        return charts

    # --------------------------------------------------
    # 6. CASE: 0 DIMENSIONS
    # --------------------------------------------------
    if len(dimensions) == 0:
        if len(measures) == 2:
            m1, m2 = measures
            return {
                "type": "vega",
                "spec": {
                    "data": {"values": rows},
    
                    "resolve": {
                        "scale": {
                            "y": "independent"
                        }
                    },
    
                    "layer": [
                        {
                            "mark": "bar",
                            "encoding": {
                                "y": {
                                    "aggregate": "sum",
                                    "field": m1.name,
                                    "axis": {"title": m1.name}
                                },
                                "color": {"value": "#4c78a8"}
                            }
                        },
                        {
                            "mark": "bar",
                            "encoding": {
                                "y": {
                                    "aggregate": "sum",
                                    "field": m2.name,
                                    "axis": {"title": m2.name, "orient": "right"}
                                },
                                "color": {"value": "#f58518"}
                            }
                        }
                    ]
                }
            }

        return None

