from google.cloud import bigquery


def fetch_default_dimension_rows(
    *,
    bq_client,
    dataset: str = "barc_slm_poc",
    table: str = "barc_dimension_reference",
    limit: int | None = 100,
):
    """
    Fetch curated default dimension rows from BigQuery.

    Table schema: genre, region, target, channel, is_default
    """
    limit_clause = "" if limit is None else "LIMIT @limit"
    sql = f"""
        SELECT
          genre,
          region,
          target,
          channel,
          is_default
        FROM {dataset}.{table}
        WHERE is_default = TRUE
        ORDER BY genre, region, target, channel
        {limit_clause}
    """

    params = []
    if limit is not None:
        params.append(bigquery.ScalarQueryParameter("limit", "INT64", int(limit)))

    job_config = bigquery.QueryJobConfig(query_parameters=params)
    rows = list(bq_client.query(sql, job_config=job_config))
    return [
        {
            "genre": r.genre,
            "region": r.region,
            "target": r.target,
            "channel": r.channel,
            "is_default": bool(r.is_default),
        }
        for r in rows
    ]


def pick_selected_default_row(*, question: str | None, default_rows: list[dict]) -> dict | None:
    """
    Deterministically pick the best matching default row for a question.

    This is intentionally simple and stable:
    - Prefer exact-ish matches by substring against the question text
    - Tie-break by sorted (genre, region, target, channel)
    """
    if not default_rows:
        return None

    q = (question or "").lower()

    def score(r: dict) -> tuple[int, tuple]:
        s = 0
        for k in ("genre", "region", "target", "channel"):
            v = (r.get(k) or "")
            if not isinstance(v, str) or not v:
                continue
            if v.lower() in q:
                s += 1
        # deterministic tie-breaker
        t = (
            (r.get("genre") or ""),
            (r.get("region") or ""),
            (r.get("target") or ""),
            (r.get("channel") or ""),
        )
        return (s, t)

    # pick max score; on tie, pick smallest lexicographically by t (stable)
    best = None
    for r in default_rows:
        if not isinstance(r, dict):
            continue
        sc = score(r)
        if best is None:
            best = (sc, r)
            continue
        if sc[0] > best[0][0]:
            best = (sc, r)
        elif sc[0] == best[0][0] and sc[1] < best[0][1]:
            best = (sc, r)

    return best[1] if best else default_rows[0]

