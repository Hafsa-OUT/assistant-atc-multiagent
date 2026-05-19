"""
Agent Researcher - Effectue le RAG multi-corpus ATC 
"""
from pathlib import Path
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage
from tools.rag_tools import get_rag_tools

default_llm = Ollama(model="llama3")
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "researcher.txt"

def load_researcher_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def researcher_node(state: dict) -> dict:
    question = state.get("question", "")
    print(f"[Researcher] Recherche pour: {question}")

    tools = get_rag_tools()
    results = []

    for tool in tools:
        try:
            result = tool.func(question)
            results.append(f"[{tool.name}]: {result}")
        except Exception as e:
            results.append(f"[{tool.name}]: Erreur - {str(e)}")

    research_result = "\n\n".join(results) if results else "Aucun résultat trouvé."
    print(f"[Researcher] Terminé.")

    return {
        **state,
        "research_result": research_result,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"[Researcher]: {research_result}")
        ],
    }