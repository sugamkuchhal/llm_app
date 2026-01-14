import logging

from charts.builder import build_bq_style_chart

logger = logging.getLogger("logger")


def build_ui_answer(core_answer, metric_payload, domain_meta):
    """
    Build UI answer with visuals using the unified chart builder.
    """
    ui_answer = dict(core_answer)
    ui_metrics = []

    for metric in metric_payload:
        results = []
        for r in metric["results"]:
            rows = r.get("rows") or []
            schema = r.get("schema") or []
            field_info = r.get("field_info") or []

            visuals = build_bq_style_chart(
                rows,
                schema,
                field_info=field_info,
                domain_meta=domain_meta,
            )
            # Normalize to list for the template.
            if visuals is None:
                visuals = []
            elif isinstance(visuals, dict):
                visuals = [visuals]

            results.append(
                {
                    "sql": r.get("sql"),
                    "rows": rows,
                    "schema": schema,
                    "field_info": field_info,
                    "visuals": visuals,
                }
            )

        ui_metrics.append(
            {
                "metric_id": metric.get("metric_id"),
                "metric_name": metric.get("metric_name"),
                "definition": metric.get("definition"),
                "business_question": metric.get("business_question"),
                "results": results,
            }
        )

    ui_answer["metrics"] = ui_metrics
    return ui_answer

