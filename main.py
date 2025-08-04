from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from rag.ingest_events import refresh_endpoint
from session import AgentSession

load_dotenv()

app = FastAPI()
agent = AgentSession()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    response = agent.ask(query.question)
    return { "response": response.get("results", [])}

@app.get("/refresh")
def refresh():
    refresh_endpoint()
    return { "response": "OK"}
@app.get("/")
def root():
    return {"status": "Agent API running"}

def cli():
    while True:
        query = input(">>> ")
        if query.lower() in ["exit", "quit"]:
            break

        response = agent.ask(query)
        print("✅ JSON:")
        print(response.get("results", []))

if __name__ == "__main__":
    print("🤖 Agent CLI started. Type 'exit' to quit.")
    cli()
