import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

LOG_FILE = "logs.jsonl"
BERLIN_TZ = ZoneInfo("Europe/Berlin")

def mask_pii(text):
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "[EMAIL]", text)
    text = re.sub(r"\b\d{3,}[- ]?\d{3,}\b", "[NUMMER]", text)
    return text

def log_event(event_type, data):
    event = {
        "timestamp": datetime.now(BERLIN_TZ).isoformat(),
        "event_type": event_type,
        **data
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")