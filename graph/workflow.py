"""
Graphe LangGraph principal - Orchestration du système multi-agent ATC.
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langchain_core.messages import BaseMessage
from agents.supervisor import supervisor_node
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.responder import responder_node


class AgentState(TypedDict):
    question: str
    messages: list
    research_result: str
    analysis_result: str
    final_answer: str
    needs_human_validation: bool
    human_approved: bool
    human_feedback: str
    next: str


def human_checkpoint_node(state: AgentState) -> AgentState:
    print("[Human-in-the-Loop]  Validation requise!")

    context_for_human = {
        "question": state.get("question"),
        "analysis": state.get("analysis_result"),
        "message": "Situation critique ATC. Validation superviseur requise.",
    }

    human_response = interrupt(context_for_human)

    return {
        **state,
        "human_approved": human_response.get("human_approved", False),
        "human_feedback": human_response.get("human_feedback", ""),
    }


def route_after_supervisor(state: AgentState):
    next_agent = state.get("next", "researcher")

    if (next_agent == "responder"
        and state.get("needs_human_validation")
        and not state.get("human_approved")
    ):
        return "human_checkpoint"

    if next_agent == "finish":
        return "__end__"

    return next_agent


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("supervisor",       supervisor_node)
    builder.add_node("researcher",       researcher_node)
    builder.add_node("analyst",          analyst_node)
    builder.add_node("responder",        responder_node)
    builder.add_node("human_checkpoint", human_checkpoint_node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "researcher":       "researcher",
            "analyst":          "analyst",
            "responder":        "responder",
            "human_checkpoint": "human_checkpoint",
            "__end__":          END,
        },
    )

    builder.add_edge("researcher",       "supervisor")
    builder.add_edge("analyst",          "supervisor")
    builder.add_edge("responder",        "supervisor")
    builder.add_edge("human_checkpoint", "responder")

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


def run_agent(question: str, thread_id: str = "default") -> dict:
    graph = build_graph()

    initial_state: AgentState = {
        "question": question,
        "messages": [],
        "research_result": "",
        "analysis_result": "",
        "final_answer": "",
        "needs_human_validation": False,
        "human_approved": False,
        "human_feedback": "",
        "next": "researcher",
    }

    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(initial_state, config=config)


def resume_after_human(thread_id: str, approved: bool, feedback: str = "") -> dict:
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    return graph.invoke(
        input={
            "human_approved": approved,
            "human_feedback": feedback,
        },
        config=config,
    )
