from __future__ import annotations

from pathlib import Path

import requests

import json

DATA_DIR = Path("/var/disk/data")  # Poprawnie jako Path

popular_path = DATA_DIR / "popular.json"
all_path = DATA_DIR / "all.json"

with open(popular_path, encoding="utf-8") as f:
    popular_events = json.load(f)

with open(all_path, encoding="utf-8") as f:
    all_events = json.load(f)

ALL_EVENTS_MAP = {
    e["event_id"]: e for e in popular_events + all_events
}


def get_event_by_id(event_id: str) -> dict | None:
    return ALL_EVENTS_MAP.get(event_id)


LIVE_URL = "https://pawa-proxy.replit.app/apiplus/events/live?x-pawa-brand=betpawa-uganda"


def fetch_live_events() -> list[dict]:
    response = requests.get(LIVE_URL)
    response.raise_for_status()
    return response.json()
