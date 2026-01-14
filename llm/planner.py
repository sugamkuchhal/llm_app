import json
import logging
import os

logger = logging.getLogger("logger")

def call_planner(
    *,
    question,
    chat_history,
    system_prompt,
    planner_prompt,
    metadata,
    default_dimension_rows,
    selected_default_dimensions,
    model
):
    MAX_PLANNER_TURNS = int(os.getenv("MAX_PLANNER_TURNS", 4))
    trimmed_history = chat_history[-MAX_PLANNER_TURNS:]

    prompt = f"""
{system_prompt}
{planner_prompt}

CHAT HISTORY:
{json.dumps(trimmed_history, indent=2)}

METADATA:
{metadata}

DEFAULT_DIMENSION_ROWS (SYSTEM-PROVIDED; VALID VALUES):
{json.dumps(default_dimension_rows, indent=2)}

SELECTED_DEFAULT_DIMENSIONS (USE UNLESS USER OVERRIDES):
{json.dumps(selected_default_dimensions, indent=2)}

QUESTION:
{question}
"""

    logger.info("Planner prompt size | approx_chars=%d", len(prompt))

    return model.generate_content(prompt).text
