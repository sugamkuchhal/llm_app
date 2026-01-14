# pipeline/analysis_runner.py

import json
import logging
import hashlib
import time
import uuid
from validators.answer_validator import validate_answer
from validators.manifest_validator import validate_manifest
from domain.barc.barc_filters import build_filters

logger = logging.getLogger("logger")


def run_analysis(
    *,
    question,
    session,
    request_id=None,
    call_planner,
    planner_ctx, 
    extract_metric_manifest,
    extract_all_sql,
    extract_all_output_schema,
    validate_output_schema,
    validate_metric_sql_binding,
    run_all_bigquery,
    build_metric_payload,
    call_interpreter,
    interpreter_ctx,
    get_chat_history,
    append_chat_turn
):
    """
    Executes the full planner → SQL → BQ → interpreter pipeline.
    """

    if request_id is None:
        request_id = str(uuid.uuid4())

    t0 = time.perf_counter()

    planner_text = call_planner(
        question=question,
        chat_history=get_chat_history(),
        **planner_ctx
    )
    
    # ===== STEP 4: HARD FAIL ON INVALID PLANNER OUTPUT =====
    
    if planner_text.count("BEGIN_METRIC_MANIFEST") != 1:
        raise RuntimeError("Planner must return exactly ONE METRIC_MANIFEST block")
    
    if "BEGIN_SQL_BLOCK_1" not in planner_text:
        raise RuntimeError("Planner returned no SQL blocks")
    
    # Disallow legacy fenced SQL
    if "```" in planner_text:
        raise RuntimeError("Legacy SQL fences detected; use SQL block markers only")
    
    # Disallow SQL outside SQL blocks (basic guard)
    before_sql_blocks = planner_text.split("BEGIN_SQL_BLOCK_1", 1)[0]
    if "SELECT" in before_sql_blocks.upper():
        raise RuntimeError("SQL detected outside SQL block markers")
    
    # ===== END STEP 4 CHECKS =====

    planner_ms = int((time.perf_counter() - t0) * 1000)
    
    logger.info(
        "phase=planner | request_id=%s | duration_ms=%d",
        request_id,
        planner_ms
    )
    
    metric_manifest = extract_metric_manifest(planner_text)
    validate_manifest(metric_manifest)

    logger.info(
        "Metric manifest extracted | metric_count=%d | metric_ids=%s",
        len(metric_manifest),
        [m.get("metric_id") for m in metric_manifest]
    )
    # ---------------------------
    # SQL extraction + execution
    # ---------------------------
    sql_blocks = extract_all_sql(planner_text)
    validate_metric_sql_binding(metric_manifest, sql_blocks)

    # Output schema blocks (MANDATORY, deterministic)
    output_schemas = extract_all_output_schema(planner_text)
    for q in sql_blocks:
        idx = q.get("metric_index")
        if idx not in output_schemas:
            raise RuntimeError(f"Missing OUTPUT_SCHEMA block for SQL block #{idx}")
        validate_output_schema(output_schemas[idx], idx)
        q["output_schema"] = output_schemas[idx]

    t0 = time.perf_counter()
    raw_bq_results = run_all_bigquery(sql_blocks)
    bq_ms = int((time.perf_counter() - t0) * 1000)
    
    logger.info(
        "phase=bigquery | request_id=%s | duration_ms=%d | queries=%d",
        request_id,
        bq_ms,
        len(sql_blocks)
    )

    metric_payload = build_metric_payload(metric_manifest, raw_bq_results)

    metric = metric_manifest[0]
    sql_text = metric_payload[0]["results"][0]["sql"]

    filters = build_filters(metric, sql_text, planner_text)

    logger.info(
        "Metric payload built for interpreter | metric_count=%d",
        len(metric_payload)
    )

    logger.debug(
        "Metric payload | payload=%s",
        json.dumps(metric_payload, indent=2)
    )

    # ---------------------------
    # Interpreter
    # ---------------------------

    t0 = time.perf_counter()

    validated_answer = call_interpreter(
        question=question,
        planner_text=planner_text,
        metrics=metric_payload,
        **interpreter_ctx
    )

    assert isinstance(validated_answer, dict)

    interp_ms = int((time.perf_counter() - t0) * 1000)
    
    logger.info(
        "phase=interpreter | request_id=%s | duration_ms=%d",
        request_id,
        interp_ms
    )
    
    validate_answer(validated_answer)

    logger.info(
        "Interpreter output validated | metrics=%d",
        len(validated_answer["metrics"])
    )

    append_chat_turn(question, validated_answer.get("headline"))

    return {
        "question": question,
        "planner_text": planner_text,
        "filters": filters,
        "metric_manifest": metric_manifest,
        "sql_blocks": sql_blocks,
        "raw_bq_results": raw_bq_results,
        "metric_payload": metric_payload,
        "validated_answer": validated_answer,
        "request_id": request_id
    }
