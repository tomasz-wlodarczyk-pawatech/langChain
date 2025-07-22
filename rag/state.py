from __future__ import annotations

import json
from typing import TypedDict, Literal, List, Optional

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

from rag.llm_chain_search import llm_chain_search
from rag.llm_router_chain import llm_router_chain
from rag.search_events import search_events
from rag.event_lookup import get_event_by_id, fetch_live_events
from rag.llm_chain import llm_chain
from graphviz import Digraph


def visualize_custom_graph():
    dot = Digraph()

    dot.node("router")
    dot.node("rag_search")
    dot.node("live_fallback")
    dot.node("output")

    dot.edge("router", "rag_search", )
    dot.edge("router", "live_fallback")
    dot.edge("rag_search", "live_fallback")
    dot.edge("rag_search", "output")
    dot.edge("live_fallback", "output")
    dot.edge("live_fallback", "router")

    dot.render("event_graph", format="png", cleanup=False)
    print("Graph saved as event_graph.png")


visualize_custom_graph()


# === State ===
class AgentState(TypedDict):
    query: str
    matches: List[dict]
    selected_event: Optional[dict]
    route: Literal["rag_search", "live_fallback", "use_previous"]
    results: Optional[List[dict]]


# === Tool: Vector DB search ===
def vector_search(state: AgentState) -> AgentState:
    matches = search_events(state["query"])
    return {
        **state,
        "matches": matches,
    }


# === Tool: LIVE Fallback ===
def fallback_live(state: AgentState) -> AgentState:
    print("⚠️ No events found in vectorstore, trying LIVE fallback...")
    matches = fetch_live_events()
    return {
        **state,
        "matches": matches,
    }


# === Tool: Use Previous ===
def use_previous(state: AgentState) -> AgentState:
    print("💡 Using context from previous event.")
    return state


# === Node: Ask LLM to decide next step ===
def router(state: AgentState) -> AgentState:
    event = state.get("selected_event")

    response = llm_router_chain.invoke({
        "user_query": state["query"],
        "event_summary": None
    })

    route = json.loads(response.content)["route"]
    return {**state, "route": route}


def format_output_rag(state: AgentState) -> dict:
    query = state.get("query", "")
    matches = state.get("matches", [])

    if not matches:
        return {"results": []}

    # Formatuj dane eventów do wejścia dla LLM
    events_text = "\n\n".join([
        f"Event: {e.get('event_name', 'N/A')}\n"
        f"event_id: {e.get('event_id', 'N/A')}\n"
        f"Competition: {e.get('competition', 'N/A')}"
        for e in matches
    ])

    print(len(events_text))
    # Wywołaj LLM z przygotowanym tekstem


    # Debugging output

    if (state["route"] == "rag_search"):
        response = llm_chain_search.invoke({
            "user_query": query,
            "events": events_text
        })
        results = [get_event_by_id(eid.event_id) for eid in response.root if eid]
        return {"results": results}


    return {"results": matches}


# === Node: Format output with LLM ===
def format_output(state: AgentState) -> dict:
    route = state.get("route", "")
    query = state.get("query", "")
    selected = state.get("selected_event")

    if isinstance(selected, str):
        try:
            selected = json.loads(selected)
        except Exception as e:
            print(f"❌ Failed to parse selected_event: {e}")
            return {"results": []}

    if route == "use_previous":
        if not selected:
            return {"results": []}

        markets = selected.get("markets", [])
        if markets and isinstance(markets[0], dict):
            market_names = [m.get("name", "") for m in markets]
        else:
            market_names = markets

        events_text = f"""Event: {selected['event_name']}
Competition: {selected['competition']}
Start: {selected['start_time']}
Markets: {market_names}"""
    else:
        matches = state.get("matches", [])
        if not matches:
            return {"results": []}

        events_text = "\n\n".join([
            f"Event: {e['event_name']}\nevent_id: {e['event_id']}\nCompetition: {e['competition']}\nStart: {e['start_time']}\nMarkets: {[m['name'] for m in e.get('markets', [])]}"
            for e in matches
        ])

    response = llm_chain.invoke({
        "user_query": query,
        "events": events_text
    })

    print(response)
    return {"results": response.root}


# === Build LangGraph ===
def build_event_graph():
    builder = StateGraph(AgentState)

    builder.set_entry_point("router")

    builder.add_node("router", RunnableLambda(router))
    builder.add_node("rag_search", RunnableLambda(vector_search))
    builder.add_node("live_fallback", RunnableLambda(fallback_live))
    builder.add_node("output", RunnableLambda(format_output_rag))

    builder.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "rag_search": "rag_search",
            "live_fallback": "live_fallback",
        }
    )

    # Wszystkie ścieżki kończą się na `output`, gdzie odpytywany jest LLM
    builder.add_edge("rag_search", "output")
    builder.add_edge("live_fallback", "output")

    builder.set_finish_point("output")
    visualize_custom_graph()

    return builder.compile()
