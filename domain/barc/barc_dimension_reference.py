from google.cloud import bigquery


def fetch_default_dimension_rows(
    *,
    bq_client,
    dataset: str = "barc_slm_poc",
    table: str = "barc_dimension_reference",
    limit: int = 100,
):
    """
    Fetch curated default dimension rows from BigQuery.

    Table schema: genre, region, target, channel, is_default
    """
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
        LIMIT @limit
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("limit", "INT64", int(limit))]
    )

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

