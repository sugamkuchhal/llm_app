from google.cloud import bigquery

def shadow_resolve_dimensions_bq(
    *,
    bq_client,
    genre,
    region,
    target,
    channel
):
    """
    BARC1 shadow-mode resolution using barc_slm_poc.dimension_reference.
    Read-only. No behavior change.
    """

    conditions = []
    params = []

    if genre:
        conditions.append("LOWER(genre) = LOWER(@genre)")
        params.append(bigquery.ScalarQueryParameter("genre", "STRING", genre))

    if region:
        conditions.append("LOWER(region) = LOWER(@region)")
        params.append(bigquery.ScalarQueryParameter("region", "STRING", region))

    if target:
        conditions.append("LOWER(target) = LOWER(@target)")
        params.append(bigquery.ScalarQueryParameter("target", "STRING", target))

    if channel:
        conditions.append("LOWER(channel) = LOWER(@channel)")
        params.append(bigquery.ScalarQueryParameter("channel", "STRING", channel))

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    sql = f"""
        SELECT
          genre,
          region,
          target,
          channel,
          is_default
        FROM barc_slm_poc.dimension_reference
        WHERE {where_clause}
        ORDER BY is_default DESC
        LIMIT 1
    """

    job_config = bigquery.QueryJobConfig(query_parameters=params)
    rows = list(bq_client.query(sql, job_config=job_config))

    if not rows:
        return {}

    r = rows[0]
    return {
        "genre": r.genre,
        "region": r.region,
        "target": r.target,
        "channel": r.channel,
        "is_default": r.is_default,
    }
