import os
import json
from jsonschema import validate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "schemas", "answer.schema.json")) as f:
    ANSWER_SCHEMA = json.load(f)

def validate_answer(answer):
    """
    Raises jsonschema.ValidationError if invalid
    """
    validate(instance=answer, schema=ANSWER_SCHEMA)
