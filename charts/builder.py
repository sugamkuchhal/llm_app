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

def build_bq_style_chart(rows, schema):
    if not rows:
        return None

    # 🔧 NEW: infer schema if missing
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


