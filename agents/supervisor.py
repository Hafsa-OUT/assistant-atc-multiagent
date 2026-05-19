"""
Agent Supervisor - Orchestre les agents via LangGraph.
"""
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms import Ollama
default_llm = Ollama(model="llama3")

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "supervisor.txt"

def load_supervisor_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def supervisor_node(state: dict) -> dict:
    system_prompt = load_supervisor_prompt()

    context_parts = [f"Question du contrôleur: {state.get('question', '')}"]
    if state.get("research_result"):
        context_parts.append(f"Résultat Researcher:\n{state['research_result']}")
    if state.get("analysis_result"):
        context_parts.append(f"Résultat Analyst:\n{state['analysis_result']}")
    if state.get("final_answer"):
        context_parts.append(f"Réponse finale:\n{state['final_answer']}")

    context = "\n\n".join(context_parts)
    context += "\n\nQuel est le prochain agent à appeler?"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    response = default_llm.invoke(messages)
    next_agent = response.content.strip().lower()

    valid_agents = {"researcher", "analyst", "responder", "finish"}
    if next_agent not in valid_agents:
        if not state.get("research_result"):
            next_agent = "researcher"
        elif not state.get("analysis_result"):
            next_agent = "analyst"
        elif not state.get("final_answer"):
            next_agent = "responder"
        else:
            next_agent = "finish"

    print(f"[Supervisor] Prochain agent: {next_agent}")
    return {**state, "next": next_agent}