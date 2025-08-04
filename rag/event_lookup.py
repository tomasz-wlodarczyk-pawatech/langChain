from __future__ import annotations
import requests

import json
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "rag/data"
all_path = DATA_DIR / "all.json"


def get_event_by_id(event_id: str) -> dict | None:
    try:
        with open(all_path, encoding="utf-8") as f:
            all_events = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ {all_path} not found.")
        return None

    all_events_map = {
        e["event_id"]: e for e in all_events if "event_id" in e
    }
    return all_events_map.get(event_id)


LIVE_URL = "https://pawa-proxy.replit.app/apiplus/events/live?x-pawa-brand=betpawa-uganda"

def fetch_live_events() -> list[dict]:
    response = requests.get(LIVE_URL)
    response.raise_for_status()
    return response.json()