from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import RunnableMap

llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = PromptTemplate.from_template("""
You are a routing assistant that decides which path to take to find the right sport event.

Available routes:
- "rag_search" → use semantic search (vector DB)
- "live_fallback" → use real-time live API
- "use_previous" → use the last known event if the query refers to it or query contains this, for this event, for this match

User query: {user_query}
Previous event: {event_summary}

Respond ONLY with one of the following JSONs:
{{
  "route": "rag_search"
}}
{{
  "route": "live_fallback"
}}
{{
  "route": "use_previous"
}}
""")

llm_router_chain = prompt | llm
