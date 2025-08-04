from rag.state import build_event_graph


class AgentSession:
    def __init__(self):
        self.graph = build_event_graph()
        self.last_event = None
        self.last_matches = []
        self.last_results = []

    def reload_graph(self):
        self.graph = build_event_graph()
        self.last_event = None
        self.last_matches = []
        self.last_results = []

    def ask(self, query: str) -> dict:
        input_state = {
            "query": query,
            "matches": self.last_matches,
            "selected_event": self.last_event,
            "results": self.last_results,
            "route": "rag_search",  # initial dummy value; LLM will override it
        }

        output = self.graph.invoke(input_state)

        results = output.get("results", [])
        if results:
            self.last_results = results
            self.last_event = results[0]
            self.last_matches = output.get("matches", self.last_matches)

        return output
