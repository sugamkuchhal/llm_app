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
    default_data,
    all_data,
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

DEFAULT DATA:
{default_data}

ALL DATA:
{all_data}

QUESTION:
{question}
"""

    logger.info("Planner prompt size | approx_chars=%d", len(prompt))

    return model.generate_content(prompt).text
