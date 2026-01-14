import os
import json
import re
import logging
import vertexai
import markdown
import hashlib
import uuid
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="vertexai.generative_models"
)


from flask import Flask, request, render_template_string, session
from google.cloud import bigquery
from vertexai.preview.generative_models import GenerativeModel
from validators.answer_validator import validate_answer
from charts.builder import build_bq_style_chart
from pipeline.analysis_runner import run_analysis
from ui.answer_adapter import build_ui_answer
from domain.barc.barc_validation import shadow_resolve_dimensions_bq
from llm.planner import call_planner
from llm.interpreter import call_interpreter
from config.prompt_guard import assert_prompt_unchanged, hash_text


import sys
import logging

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)

# Prevent duplicate handlers
if not root.handlers:
    root.addHandler(handler)

logging.getLogger("werkzeug").setLevel(logging.WARNING)

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
logger.propagate = True


# --------------------------------------------------
# Config
# --------------------------------------------------
PROJECT_ID = os.getenv("GCP_PROJECT", "prj-uat-data-coe-analytics34")

VERTEX_REGION = os.getenv("VERTEX_REGION", "us-central1")
# VERTEX_REGION = os.getenv("VERTEX_REGION", "asia-south1")
# BQ_REGION = os.getenv("BIGQUERY_REGION", "us-central1")
BQ_REGION = os.getenv("BIGQUERY_REGION", "asia-south1")

SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

MAX_TURNS = int(os.getenv("MAX_TURNS", 10))

MAX_BQ_BYTES = int(
    os.getenv("MAX_BQ_BYTES", 10 * 1024 * 1024 * 1024)
)  # default: 10 GB

MAX_ROWS = int(os.getenv("MAX_ROWS", 50000))

# --------------------------------------------------
# Init
# --------------------------------------------------
vertexai.init(project=PROJECT_ID, location=VERTEX_REGION)

bq_client = bigquery.Client(location=BQ_REGION)
planner_model = GenerativeModel("gemini-2.5-pro")
interpreter_model = GenerativeModel("gemini-2.5-pro")

app = Flask(__name__)
app.jinja_env.filters["markdown"] = lambda text: markdown.markdown(text or "")

app.secret_key = SECRET_KEY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_file(path):
    """
    Deterministic file loader:
    - UTF-8 (strict)
    - stable newlines (normalize to \\n)
    """
    full_path = os.path.join(BASE_DIR, path)
    try:
        with open(full_path, "r", encoding="utf-8", errors="strict") as f:
            # Note: Python's text mode already does universal newlines, but we
            # normalize explicitly for hash stability across environments.
            text = f.read()
    except OSError as e:
        raise RuntimeError(f"Failed to read file: {full_path}") from e

    return text.replace("\r\n", "\n").replace("\r", "\n")

DOMAIN = "barc"

SYSTEM_PROMPT = load_file("prompt_system.txt")
INTERPRETER_PROMPT = load_file("prompt_interpreter.txt")
ANSWER_CONTRACT = load_file("contract_answer.json")

ALL_DATA = load_file(f"domain/{DOMAIN}/{DOMAIN}_all.json")
PLANNER_PROMPT = load_file(f"domain/{DOMAIN}/{DOMAIN}_context_planner.txt")
DEFAULT_DATA = load_file(f"domain/{DOMAIN}/{DOMAIN}_default.json")
META_DATA = load_file(f"domain/{DOMAIN}/{DOMAIN}_meta.json")

SYSTEM_PROMPT_HASH = "684508a01ca429b46242bb5ac342cf9891a5c65274fb0cc9115e73af89388d0a"
PLANNER_PROMPT_HASH = "423e2b23999532dda04c55f73aa0b69c44a953d9522780d1401460972ee2a117"

assert_prompt_unchanged("SYSTEM_PROMPT", SYSTEM_PROMPT, SYSTEM_PROMPT_HASH)
assert_prompt_unchanged("PLANNER_PROMPT", PLANNER_PROMPT, PLANNER_PROMPT_HASH)


# --------------------------------------------------
# Session memory
# --------------------------------------------------
def get_chat_history():
    """
    Returns the current chat history from the session.

    Policy A (self-healing): if the stored value is malformed, reset it to [].
    """
    h = session.get("chat_history")
    if h is None:
        session["chat_history"] = []
        return session["chat_history"]

    if not isinstance(h, list):
        session["chat_history"] = []
        return session["chat_history"]

    # Best-effort sanitization: keep only dict entries (stable shape downstream).
    sanitized = [x for x in h if isinstance(x, dict)]
    if sanitized is not h:
        session["chat_history"] = sanitized
    return session["chat_history"]

def append_chat_turn(question, headline):
    h = get_chat_history()

    # Normalize to deterministic, JSON-serializable primitives.
    q = (question or "")
    if not isinstance(q, str):
        q = str(q)
    q = q.strip()

    hl = None if headline is None else headline
    if hl is not None and not isinstance(hl, str):
        hl = str(hl)
    if isinstance(hl, str):
        hl = hl.strip()

    h.append({"question": q, "headline": hl})
    session["chat_history"] = h[-MAX_TURNS:]

def extract_metric_manifest(text: str):
    """
    Fail-fast extraction of the metric manifest.

    Contract:
    - Exactly one BEGIN_METRIC_MANIFEST and one END_METRIC_MANIFEST marker
    - Content between markers must be a JSON array
    """
    begin = "BEGIN_METRIC_MANIFEST"
    end = "END_METRIC_MANIFEST"

    begin_count = text.count(begin)
    end_count = text.count(end)
    if begin_count != 1 or end_count != 1:
        raise RuntimeError(
            f"Invalid METRIC_MANIFEST markers: begin={begin_count}, end={end_count}"
        )

    begin_idx = text.find(begin)
    end_idx = text.find(end)
    if begin_idx == -1 or end_idx == -1 or end_idx <= begin_idx:
        raise RuntimeError("Invalid METRIC_MANIFEST marker ordering")

    block = text[begin_idx + len(begin):end_idx].strip()
    logger.info("RAW METRIC_MANIFEST >>>\n%s\n<<< END METRIC_MANIFEST", block)

    # Deterministic guard: the manifest must be a JSON array.
    if not block.startswith("[") or not block.endswith("]"):
        raise RuntimeError("METRIC_MANIFEST must be a JSON array")

    try:
        return json.loads(block)
    except json.JSONDecodeError as e:
        # Include parse position for debuggability without attempting repair.
        context_start = max(e.pos - 40, 0)
        context_end = min(e.pos + 40, len(block))
        context = block[context_start:context_end].replace("\n", "\\n")
        raise RuntimeError(
            f"Invalid METRIC_MANIFEST JSON at pos={e.pos}: {e.msg} | context='{context}'"
        ) from e

def validate_metric_sql_binding(metric_manifest, sql_blocks):
    """
    Ensures metric → SQL references are valid.

    Deterministic policy: validate references only.
    - Hard-fail if the manifest references SQL blocks that do not exist
    - Do NOT fail if there are extra SQL blocks not referenced by the manifest
    """
    declared_sql_indices = set()
    for m in metric_manifest:
        for idx in m.get("sql_blocks", []):
            declared_sql_indices.add(idx)

    extracted_sql_indices = {q["metric_index"] for q in sql_blocks}

    missing_sql = declared_sql_indices - extracted_sql_indices

    if missing_sql:
        logger.error(
            "Metric references missing SQL blocks | missing=%s",
            sorted(missing_sql)
        )
        raise RuntimeError(
            f"Metric manifest references missing SQL blocks: {sorted(missing_sql)}"
        )

    logger.info(
        "Metric → SQL references validated | metrics=%d | sql_blocks=%d",
        len(metric_manifest),
        len(sql_blocks)
    )

def build_metric_payload(metric_manifest, sql_results):
    """
    Groups executed SQL results by metric_id based on the manifest.
    """
    results_by_index = {
        r["metric_index"]: r for r in sql_results
    }

    metric_payload = []

    for m in metric_manifest:
        metric_results = []

        for idx in m["sql_blocks"]:
            if idx not in results_by_index:
                metric_id = m.get("metric_id", "<unknown>")
                raise RuntimeError(
                    f"Missing executed SQL result for metric_id={metric_id} sql_block={idx}"
                )
            metric_results.append(results_by_index[idx])

        metric_payload.append({
            "metric_id": m["metric_id"],
            "metric_name": m.get("metric_name"),
            "definition": m.get("definition"),
            "business_question": m.get("business_question"),
            "results": metric_results
        })

    return metric_payload

# --------------------------------------------------
# SQL extraction (MULTI)
# --------------------------------------------------
SQL_READ_ONLY_RE = re.compile(
    r"^\s*(with\b[\s\S]+?\bselect\b|\(?\s*select\b)",
    re.IGNORECASE
)

FORBIDDEN_SQL = re.compile(
    r"\b("
    # DDL/DML
    r"insert|update|delete|merge|create|drop|alter|truncate|replace|rename|"
    # Privilege / procedure
    r"grant|revoke|call|"
    # BigQuery scripting / dynamic SQL (not allowed in read-only flow)
    r"begin|end|declare|set|commit|rollback|execute\s+immediate|execute"
    r")\b",
    re.IGNORECASE
)

def validate_read_only_sql(sql: str, index: int):
    """
    Strict read-only validation:
    - Single statement only (no internal semicolons)
    - Must start with SELECT/WITH (after stripping comments)
    - Must not contain DDL/DML/scripting keywords
    """
    clean_sql = strip_sql_comments(sql).strip()

    # Allow a single trailing semicolon, disallow any internal semicolons.
    if clean_sql.endswith(";"):
        clean_sql = clean_sql[:-1].rstrip()
    if ";" in clean_sql:
        raise RuntimeError(f"Multiple statements are not allowed (query #{index})")

    if not SQL_READ_ONLY_RE.match(clean_sql):
        raise RuntimeError(f"SQL must start with SELECT/WITH (query #{index})")

    match = FORBIDDEN_SQL.search(clean_sql)
    if match:
        logger.warning(
            "Rejected SQL | metric=%s | keyword=%s | sql=%s",
            index,
            match.group(0),
            sql.replace("\n", " ")
        )
        raise RuntimeError(f"Non read-only SQL in query #{index}")

def extract_all_sql(text: str):
    """
    Extracts sequential SQL blocks.

    Deterministic policy: fail fast on gaps/duplicates.
    - If blocks exist, they must be numbered 1..N with no gaps and no duplicates.
    - Each BEGIN_SQL_BLOCK_i must have a matching END_SQL_BLOCK_i.
    """
    begin_matches = re.findall(r"\bBEGIN_SQL_BLOCK_(\d+)\b", text)
    if not begin_matches:
        raise RuntimeError("No SQL blocks found")

    indices = [int(x) for x in begin_matches]
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"Duplicate SQL block numbers detected: {sorted(indices)}")

    present = sorted(indices)
    if present[0] != 1:
        raise RuntimeError(f"SQL blocks must start at 1; found {present[0]}")

    expected = list(range(1, present[-1] + 1))
    if present != expected:
        missing = sorted(set(expected) - set(present))
        raise RuntimeError(
            f"SQL block numbering must be sequential; missing blocks: {missing}"
        )

    sql_blocks = []
    for i in expected:
        start = f"BEGIN_SQL_BLOCK_{i}"
        end = f"END_SQL_BLOCK_{i}"

        if text.count(start) != 1 or text.count(end) != 1:
            raise RuntimeError(f"Invalid SQL block markers for block #{i}")

        m = re.search(
            rf"\b{re.escape(start)}\b\s*(.*?)\s*\b{re.escape(end)}\b",
            text,
            flags=re.DOTALL
        )
        if not m:
            raise RuntimeError(f"Invalid SQL block #{i}")

        block = m.group(1).strip()
        validate_read_only_sql(block, i)
        sql_blocks.append({"metric_index": i, "sql": block})

    return sql_blocks

def strip_sql_comments(sql: str) -> str:
    # Remove single-line comments
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    # Remove multi-line comments
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


# --------------------------------------------------
# BigQuery execution
# --------------------------------------------------
def run_all_bigquery(queries):
    results = []

    for q in queries:
        dry_cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        dry_job = bq_client.query(q["sql"], job_config=dry_cfg, location=BQ_REGION)

        # BigQuery-enforced check: only SELECT statements are allowed.
        stmt_type = getattr(dry_job, "statement_type", None)
        if stmt_type is not None and stmt_type.upper() != "SELECT":
            raise RuntimeError(
                f"Query #{q['metric_index']} is not read-only (statement_type={stmt_type})"
            )

        if dry_job.total_bytes_processed > MAX_BQ_BYTES:
            raise RuntimeError(
                f"Query #{q['metric_index']} exceeds byte limit "
                f"({dry_job.total_bytes_processed} > {MAX_BQ_BYTES})"
            )

        job = bq_client.query(q["sql"], location=BQ_REGION)
        rows = [dict(row) for row in job.result(max_results=MAX_ROWS)]
        schema = job.schema

        results.append({
            "metric_index": q["metric_index"],
            "sql": q["sql"],
            "rows": rows,
            "schema": schema
        })

    return results

        
# --------------------------------------------------
# UI Template
# --------------------------------------------------
HTML = """
<!doctype html>
<html>
<head>
<title>SLM Analytics</title>

<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap">
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>

<style>
body { font-family: Inter, system-ui, sans-serif; margin: 40px; }
.container { max-width: 900px; }
details summary { cursor:pointer; font-weight:600; font-size: 1rem; margin: 16px 0 8px 0; list-style: none; }
details summary::-webkit-details-marker { display: none; }
details summary::marker { display: none; }
details summary::before { content:"▶ "; font-size: 0.75em; margin-right: 6px; color: #666; }
details[open] summary::before { content:"▼ "; }
.details-body p { margin: 6px 0; line-height: 1.55; }
.details-body h1,
.details-body h2,
.details-body h3 { font-size: 1.05rem; font-weight: 600; margin: 16px 0 6px 0; }
.details-body p > strong:only-child { display: block; font-weight: 600; margin: 14px 0 4px 0; }
.details-body ul,
.details-body ol { margin: 6px 0 6px 1.2em; }
.details-body li { margin-bottom: 4px; line-height: 1.5; }

details[open] .markdown-body p { margin: 8px 0; line-height: 1.0; }
pre { background:#f8f8f8; padding:12px; white-space:pre-wrap; }
.error { color:darkred; font-weight:600; }
</style>
</head>

<body>
<h2>SLM Analytics</h2>

<form id="askForm" method="post" onsubmit="handleSubmit(event)">
  <input name="question" required style="width:700px;padding:6px">
  <button type="submit">Ask</button>
</form>

<div id="results" class="container">
{% if error %}<p class="error">{{ error }}</p>{% endif %}
{% if answer %}
<hr>

<details open>
  <summary>Question</summary>
  <h3>{{ question }}</h3>
</details>

<details>
  <summary>Metrics & Analysis Planner</summary>
  <pre>{{ planner_text }}</pre>
</details>

{% if filters %}
<details open>
  <summary>Filters</summary>
  <ul>
    {% for f in filters %}
      <li>{{ f }}</li>
    {% endfor %}
  </ul>
</details>
{% endif %}

<details open>
  <summary>Answer</summary>
  <h3>{{ answer.headline }}</h3>
</details>

<details open>
  <summary>Key Takeaways</summary>
  <ul>{% for t in answer.takeaways %}<li>{{ t }}</li>{% endfor %}</ul>
</details>

<details open>
  <summary>Details</summary>
  <div class="markdown-body details-body">
    {{ answer.details | markdown | safe }}
  </div>
</details>

{% for metric in answer.metrics %}
<h4>{{ metric.get("metric_name") or metric.metric_id }}</h4>

{% if metric.definition %}
<p><em>{{ metric.definition }}</em></p>
{% endif %}

{% for r in metric.results %}
  {% if r.visuals %}
    <details open>
      <summary>Visualization</summary>
  
      {% for v in r.visuals %}
        <div id="chart-{{ metric.metric_id }}-{{ loop.index0 }}-{{ loop.index }}"></div>
        <script>
          vegaEmbed(
            "#chart-{{ metric.metric_id }}-{{ loop.index0 }}-{{ loop.index }}",
            {{ v.spec | tojson }}
          );
        </script>
      {% endfor %}
  
    </details>
  {% else %}
  <details>
    <summary>Results (Table)</summary>
    <pre>{{ r.rows }}</pre>
  </details>
  {% endif %}

  <details>
    <summary>Executed Query</summary>
    <pre>{{ r.sql }}</pre>
  </details>
{% endfor %}

{% endfor %}



{% endif %}
</div>

<script>
let countdownInterval;

function handleSubmit(e) {
  e.preventDefault(); // ⬅️ REQUIRED

  let seconds = 99;

  document.getElementById("results").innerHTML = `
    <p><em>Running analysis…</em></p>
    <p id="countdown"><small>~${seconds}s remaining</small></p>
  `;

  countdownInterval = setInterval(() => {
    seconds--;

    if (seconds <= 0) {
      document.getElementById("countdown").innerHTML =
        "<small>Still working…</small>";
      clearInterval(countdownInterval);
      return;
    }

    document.getElementById("countdown").innerHTML =
      `<small>~${seconds}s remaining</small>`;
  }, 1000);

  // ⬇️ allow browser to paint, then submit
  setTimeout(() => {
    document.getElementById("askForm").submit();
  }, 50);
}
</script>


</body>
</html>
"""

# --------------------------------------------------
# Route
# --------------------------------------------------
@app.route("/", methods=["GET","POST"])
def index():

    filters = []
    planner_text = ""
    answer = {}
    error = None
    question = None
    request_id = str(uuid.uuid4())

    logger.info(
        "(%s) Request started | request_id=%s | limits={max_bq_bytes:%d,max_turns:%d,max_rows:%d}",
        request.method,
        request_id,
        MAX_BQ_BYTES,
        MAX_TURNS,
        MAX_ROWS
    )

    if request.method == "POST":
        try:
            question = request.form["question"]  # ✅ FIX

            result = run_analysis(
                question=question,
                session=session,
                request_id=request_id,
                call_planner=call_planner,
                planner_ctx={
                    "system_prompt": SYSTEM_PROMPT,
                    "planner_prompt": PLANNER_PROMPT,
                    "metadata": META_DATA,
                    "default_data": DEFAULT_DATA,
                    "all_data": ALL_DATA,
                    "model": planner_model,
                },
                extract_metric_manifest=extract_metric_manifest,
                extract_all_sql=extract_all_sql,
                validate_metric_sql_binding=validate_metric_sql_binding,
                run_all_bigquery=run_all_bigquery,
                build_metric_payload=build_metric_payload,
                call_interpreter=call_interpreter,
                interpreter_ctx={
                    "system_prompt": SYSTEM_PROMPT,
                    "interpreter_prompt": INTERPRETER_PROMPT,
                    "answer_contract": ANSWER_CONTRACT,
                    "model": interpreter_model,
                },
                get_chat_history=get_chat_history,
                append_chat_turn=append_chat_turn
            )
            
            planner_text = result["planner_text"]
            filters = result["filters"]

            def _extract_dim(filters, name):
                for f in filters:
                    if f.lower().startswith(name.lower()):
                        return f.split(":", 1)[-1].strip()
                return None
            
            resolved_genre = _extract_dim(filters, "genre")
            resolved_region = _extract_dim(filters, "region")
            resolved_target = _extract_dim(filters, "target")
            resolved_channel = _extract_dim(filters, "channel")
            
            shadow_dims = shadow_resolve_dimensions_bq(
                bq_client=bq_client,
                genre=resolved_genre,
                region=resolved_region,
                target=resolved_target,
                channel=resolved_channel
            )
            
            logger.info(
                "SHADOW_DIMENSION_COMPARE | planner=%s | db=%s | match=%s",
                {
                    "genre": resolved_genre,
                    "region": resolved_region,
                    "target": resolved_target,
                    "channel": resolved_channel,
                },
                shadow_dims,
                (
                    shadow_dims.get("genre") == resolved_genre and
                    shadow_dims.get("region") == resolved_region and
                    shadow_dims.get("target") == resolved_target
                )
            )

            core_answer = result["validated_answer"]
            metric_payload = result["metric_payload"]

#            answer = build_ui_answer(core_answer, metric_payload)
            answer = build_ui_answer(core_answer, metric_payload, json.loads(META_DATA))

        except Exception as e:
            error = str(e)

    return render_template_string(
        HTML,
        question=question,
        planner_text=planner_text,
        filters=filters,
        answer=answer,
        error=error
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)
