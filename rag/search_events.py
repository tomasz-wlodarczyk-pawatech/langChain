import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.event_lookup import get_event_by_id
from rag.llm_chain import llm_chain

load_dotenv()

CHROMA_BASE_DIR = Path("/opt/render/project/src/rag/rag/chroma_db")
EMBEDDING_MODEL = OpenAIEmbeddings()


def get_vectorstore(source: str) -> Chroma:
    assert source in ["popular", "all"], "source must be 'popular' or 'all'"
    db_path = CHROMA_BASE_DIR / source
    print(f"Loading vectorstore from {db_path}")
    return Chroma(
        embedding_function=EMBEDDING_MODEL,
        persist_directory=str(db_path)
    )


def search_events(query: str, source: str = "all", top_k: int = 5) -> list[dict]:
    vectorstore = get_vectorstore(source).as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 50, "score_threshold": 0.65}
    )

    results: list[Document] = vectorstore.invoke(query)
    event_ids = [doc.metadata.get("event_id") for doc in results]
    for doc in results:
        print(doc.metadata.get("score"))
    return [get_event_by_id(eid) for eid in event_ids if eid]


def generate_final_json(query: str, source: str = "all"):
    matches = search_events(query, source)
    if not matches:
        print("❌ Be more specific")
        return

    combined_text = "\n\n".join([
        f"Event: {e['event_name']}\nCompetition: {e['competition']}\nStart: {e['start_time']}\nMarkets: {[m['name'] for m in e.get('markets', [])]}"
        for e in matches
    ])

    response = llm_chain.invoke({
        "user_query": query,
        "events": combined_text
    })

    print("\n✅ JSON Output from LLM:")
    print(json.loads(response.content))


if __name__ == "__main__":
    sample_query = "Give me two events only for Brazil league with under/over market"
    # events = search_events(sample_query, source="popular", top_k=3)
    generate_final_json(sample_query, source="all")
    # for e in events:
    #     print(f"- {e}")
