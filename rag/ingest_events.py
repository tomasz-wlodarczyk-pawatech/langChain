import os
import json
import shutil

import requests
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings



load_dotenv()

print("📂 Current working dir:", os.getcwd())
print("📄 This file is in:", Path(__file__).resolve())

EMBEDDING_MODEL = OpenAIEmbeddings()
DATA_DIR = Path("data")
CHROMA_DIR = Path("rag/chroma_db")
# POPULAR_URL = "https://pawa-proxy.replit.app/apiplus/events/popular?x-pawa-brand=betpawa-uganda"
ALL_URL = "https://pawa-proxy.replit.app/apiplus/events/all?x-pawa-brand=betpawa-uganda"

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0
)


# === Helper: fetch + save JSON ===
def fetch_and_store(url: str, filename: str) -> list[dict]:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


def event_to_document(event: dict) -> Document:
    start_time = event.get("start_time", "")
    start_date = start_time[:10] if start_time else ""

    content = f"""
    Event: {event.get("event_name")}
    Competition: {event.get("competition")}
    Start time: {start_time}
    Date only: {start_date}
    Market names: {[market.get("name") for market in event.get("markets", [])]}
    """

    raw_metadata = {
        "event_id": event.get("event_id"),
        "event_name": event.get("event_name"),
        "competition": event.get("competition"),
        "start_time": start_time,
    }

    return Document(page_content=content.strip(), metadata=raw_metadata)


def ingest_to_chroma(events: list[dict], index_name: str):
    print(f"Ingesting {len(events)} events into '{index_name}' index...")

    docs = [event_to_document(e) for e in events]
    docs_split = text_splitter.split_documents(docs)

    persist_path = Path("/opt/render/project/src/rag/chroma_db/all")

    if persist_path.exists():
        print(f"Removing old Chroma DB at {persist_path}")
        shutil.rmtree(persist_path)

    print("💾 Zapisuję do:", persist_path.resolve())
    Chroma.from_documents(
        documents=docs_split,
        embedding=EMBEDDING_MODEL,
        persist_directory=str(persist_path)
    )
    for root, dirs, files in os.walk(persist_path):
        for f in files:
            print("📦", os.path.join(root, f))
    print(f"Saved {len(docs_split)} chunks to Chroma index: '{index_name}'")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # popular_events = fetch_and_store(POPULAR_URL, "popular.json")
    all_events = fetch_and_store(ALL_URL, "all.json")

    # ingest_to_chroma(popular_events, "popular")
    ingest_to_chroma(all_events, "all")


if __name__ == "__main__":
    main()
