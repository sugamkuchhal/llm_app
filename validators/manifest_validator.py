import os
import json
from jsonschema import validate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "schemas", "metric_manifest.schema.json")) as f:
    MANIFEST_SCHEMA = json.load(f)

def validate_manifest(manifest):
    validate(instance=manifest, schema=MANIFEST_SCHEMA)
