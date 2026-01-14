import json
import logging
from re import search
from utils.json_utils import safe_json_extract

logger = logging.getLogger("logger")

def call_interpreter(
    *,
    question,
    planner_text,
    metrics,
    system_prompt,
    interpreter_prompt,
    answer_contract,
    model
):
    payload = {
        "metric_count": len(metrics),
        "metrics": metrics
    }

    prompt = f"""
{system_prompt}
{interpreter_prompt}

You MUST respond with ONLY valid JSON.

CONTRACT:
{answer_contract}

QUESTION:
{question}

METRICS_AND_ANALYSIS_PLAN:
{planner_text}

EXECUTED_RESULTS:
{json.dumps(payload, indent=2)}
"""

    raw = model.generate_content(prompt).text.strip()
    
    def strip_rows_from_raw(text: str) -> str:
        try:
            obj = safe_json_extract(text)
        except Exception:
            return text  # fallback: log as-is if not valid JSON
    
        def strip(o):
            if isinstance(o, dict):
                return {k: strip(v) for k, v in o.items() if k != "rows"}
            if isinstance(o, list):
                return [strip(v) for v in o]
            return o
    
        return json.dumps(strip(obj), indent=2)
    
    logger.info(
        "INTERPRETER RAW OUTPUT >>>\n%s\n<<< END RAW OUTPUT",
        strip_rows_from_raw(raw)
    )

    try:
        return safe_json_extract(raw)

    except Exception:
        repair_prompt = f"""
Fix the following output so that it becomes VALID JSON
and STRICTLY follows this contract:

{answer_contract}

INVALID OUTPUT:
{raw}

Respond with ONLY corrected JSON.
"""
        repaired = model.generate_content(repair_prompt).text.strip()

        logger.info(
            "INTERPRETER REPAIRED OUTPUT >>>\n%s\n<<< END REPAIRED OUTPUT",
            repaired
        )

        return safe_json_extract(repaired)
