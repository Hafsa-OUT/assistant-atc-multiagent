"""
Agent Analyst - Analyse les résultats du Researcher ATC.
"""
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from config.llm_config import default_llm

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "analyst.txt"

CRITICAL_KEYWORDS = [
    "mayday", "detresfa", "détresfa", "7700", "7500",
    "hijacking", "détournement", "windshear sévère",
    "panne radio", "nordo", "incendie à bord",
    "dépressurisation", "collision", "urgence immédiate",
]

def load_analyst_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def detect_critical_case(question: str, research_result: str) -> bool:
    text = (question + " " + research_result).lower()
    return any(kw in text for kw in CRITICAL_KEYWORDS)

def analyst_node(state: dict) -> dict:
    system_prompt = load_analyst_prompt()
    question = state.get("question", "")
    research_result = state.get("research_result", "")

    print(f"[Analyst] Analyse en cours...")
    needs_human = detect_critical_case(question, research_result)

    prompt_content = f"""
Question du contrôleur: {question}

Résultats de recherche:
{research_result}

{"⚠️ ATTENTION: Situation critique détectée. Évalue si validation humaine nécessaire." if needs_human else ""}

Produis ton analyse complète.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_content),
    ]

    response = default_llm.invoke(messages)
    analysis_result = response.content

    if "validation humaine requise**: oui" in analysis_result.lower():
        needs_human = True

    print(f"[Analyst] Validation humaine: {needs_human}")

    return {
        **state,
        "analysis_result": analysis_result,
        "needs_human_validation": needs_human,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"[Analyst]: {analysis_result}")
        ],
    }