import json

from rag.event_lookup import fetch_live_events, get_event_by_id
from rag.llm_chain import llm_chain
from rag.search_events import search_events


class EventAgent:
    def __init__(self, source: str = "all"):
        self.source = source  # 'popular' or 'all'
        self.last_events: list[dict] = []

    def ask(self, query: str):
        events = search_events(query, self.source)

        if not events:
            print("⚠️ No events found in vectorstore, trying LIVE fallback...")
            live = fetch_live_events()
            if not live:
                print("❌ No live events available.")
                return

            self.last_events = live
            combined_text = self._build_combined_text(self.last_events)
            self._ask_llm(query, combined_text)
            return



        full_events = [get_event_by_id(e["event_id"]) for e in events if get_event_by_id(e["event_id"])]
        self.last_events = full_events

        combined_text = self._build_combined_text(self.last_events)
        self._ask_llm(query, combined_text)

    def _build_combined_text(self, events: list[dict]) -> str:
        return "\n\n".join([
            f"Event: {e['event_name']}\nCompetition: {e['competition']}\nStart: {e['start_time']}\nID: {e['event_id']}\nMarkets: {[m['name'] for m in e.get('markets', [])]}"
            for e in events
        ])

    def _ask_llm(self, query: str, events_text: str):
        response = llm_chain.invoke({
            "user_query": query,
            "events": events_text
        })

        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            print("⚠️ Failed to parse JSON:\n", response["text"])
            return

        print("✅ JSON from LLM:")
        print(json.dumps(parsed, indent=2))
