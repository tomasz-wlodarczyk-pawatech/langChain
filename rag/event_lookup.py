from __future__ import annotations
import requests

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/opt/render/project/src/rag/data")
all_path = DATA_DIR / "all.json"

ALL_EVENTS_MAP: dict[str, dict] = {}

def refresh_endpoint_json():
    global ALL_EVENTS_MAP

    try:
        with open(all_path, encoding="utf-8") as f:
            all_events = json.load(f)
        ALL_EVENTS_MAP = {
            e["event_id"]: e for e in all_events if "event_id" in e
        }
        print(f"✅ Refreshed ALL_EVENTS_MAP with {len(ALL_EVENTS_MAP)} events")
    except Exception as e:
        print(f"❌ Failed to load all.json: {e}")
        ALL_EVENTS_MAP = {}


def get_event_by_id(event_id: str) -> dict | None:

    return ALL_EVENTS_MAP.get(event_id)


LIVE_URL = "https://pawa-proxy.replit.app/apiplus/events/live?x-pawa-brand=betpawa-uganda"


def fetch_live_events() -> list[dict]:
    response = requests.get(LIVE_URL)
    response.raise_for_status()
    return response.json()
