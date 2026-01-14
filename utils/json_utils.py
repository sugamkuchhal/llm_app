import json
import re

def safe_json_extract(text: str):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise RuntimeError("No JSON object found")
    return json.loads(match.group(0))
