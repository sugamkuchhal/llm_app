# Debugging logs (Cloud Run)

This app logs to **stdout/stderr**, which Cloud Run captures in **Cloud Logging**.
There is **no log file inside the container** unless you explicitly create one.

## 1) Recommended runtime env vars (Cloud Run)

Set these on the Cloud Run service:

- `LOG_LEVEL=DEBUG` (or `INFO` when stable)
- `LOG_FORMAT=json` (recommended; makes filtering/export easier)
- `ENABLE_LOG_BUFFER=1` (optional; enables in-memory log buffer for `/\_debug/logs`)
- `LOG_BUFFER_SIZE=2000` (optional; max log entries to keep)
- `DEBUG_TOKEN=<some-long-random-string>` (required if using `/\_debug/logs`)

## 2) Find the `request_id`

Every request logs a line like:

- `Request started | request_id=<uuid>`

Copy that UUID. All other logs for that request use the same `request_id`.

## 3) Logs Explorer queries (copy/paste)

### A) All logs for a single request (JSON logs)

```
resource.type="cloud_run_revision"
resource.labels.service_name="llm-app"
jsonPayload.request_id="REPLACE_WITH_REQUEST_ID"
```

### B) All logs for a single request (text logs)

```
resource.type="cloud_run_revision"
resource.labels.service_name="llm-app"
textPayload:"request_id=REPLACE_WITH_REQUEST_ID"
```

### C) Only errors for a single request

JSON:

```
resource.type="cloud_run_revision"
resource.labels.service_name="llm-app"
jsonPayload.request_id="REPLACE_WITH_REQUEST_ID"
severity>=ERROR
```

Text:

```
resource.type="cloud_run_revision"
resource.labels.service_name="llm-app"
textPayload:"request_id=REPLACE_WITH_REQUEST_ID"
severity>=ERROR
```

## 4) Export/download logs to share

### Option 1 (UI)

In Logs Explorer:
1. Run one of the queries above
2. Set time window to cover the request (e.g. "Last 30 minutes")
3. Click **Download logs**
4. Choose **JSON**

Share the downloaded JSON file (or paste the JSON into chat).

### Option 2 (CLI export to a file)

Replace:
- `PROJECT=prj-uat-data-coe-analytics34`
- `SERVICE=llm-app`
- `RID=<request_id>`

#### Export JSON logs

```
PROJECT=prj-uat-data-coe-analytics34
SERVICE=llm-app
RID=REPLACE_WITH_REQUEST_ID

gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE} AND jsonPayload.request_id=${RID}" \
  --project="${PROJECT}" \
  --limit=2000 \
  --format=json > "logs_${RID}.json"
```

#### Export text logs

```
PROJECT=prj-uat-data-coe-analytics34
SERVICE=llm-app
RID=REPLACE_WITH_REQUEST_ID

gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE} AND textPayload:\"request_id=${RID}\"" \
  --project="${PROJECT}" \
  --limit=2000 \
  --format=json > "logs_${RID}.json"
```

## 5) What to share for debugging

For a failing request, share:
- all log entries for that `request_id` **including the stack trace**
- the user question (if possible)

## 6) Fastest method (no Logs Explorer export): `/_debug/logs`

If you set `ENABLE_LOG_BUFFER=1` and `DEBUG_TOKEN`, you can fetch logs directly from the service:

```
https://<your-cloud-run-url>/_debug/logs?token=DEBUG_TOKEN&request_id=REPLACE_WITH_REQUEST_ID&limit=500
```

This returns JSON that you can copy/paste into chat.

