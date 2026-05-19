"""
Agent Responder - Formule la réponse finale ATC.
"""
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from config.llm_config import default_llm

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "responder.txt"

def load_responder_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def responder_node(state: dict) -> dict:
    system_prompt = load_responder_prompt()
    question = state.get("question", "")
    analysis_result = state.get("analysis_result", "")
    human_approved = state.get("human_approved", False)
    needs_human = state.get("needs_human_validation", False)

    print(f"[Responder] Formulation de la réponse...")

    if needs_human and not human_approved:
        return {
            **state,
            "final_answer": "__AWAITING_HUMAN_VALIDATION__",
        }

    prompt_content = f"""
Question du contrôleur: {question}

Analyse produite:
{analysis_result}

{"✅ Réponse validée par superviseur humain." if human_approved else ""}

Formule la réponse opérationnelle finale.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_content),
    ]

    response = default_llm.invoke(messages)
    final_answer = response.content

    print(f"[Responder] Réponse prête.")

    return {
        **state,
        "final_answer": final_answer,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"[Responder]: {final_answer}")
        ],
    }