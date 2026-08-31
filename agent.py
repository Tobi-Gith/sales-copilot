from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

INVENTORY = {"X-123": 12, "Y-456": 0}


class WorkflowState(TypedDict):
    customer_question: str
    part_number: str
    stock_quantity: int
    draft: str
    error: str


def check_inventory_node(state: WorkflowState) -> WorkflowState:
    part_number = state["part_number"]
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            quantity = INVENTORY[part_number]
            return {**state, "stock_quantity": quantity, "error": ""}
        except KeyError:
            if attempt == max_retries:
                return {**state, "error": f"Ersatzteil {part_number} nicht gefunden."}
    return state


def draft_response_node(state: WorkflowState) -> WorkflowState:
    if state.get("error"):
        return {**state, "draft": ""}

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Kundenfrage: {state['customer_question']}\n"
                f"Lagerbestand fuer {state['part_number']}: {state['stock_quantity']} Stueck.\n"
                "Formuliere eine kurze, freundliche Kundenantwort auf Deutsch. "
                "Wenn der Bestand 0 ist, informiere hoeflich ueber die Nichtverfuegbarkeit "
                "und eine ungefaehre Nachlieferzeit von 2 Wochen."
            )
        }]
    )
    return {**state, "draft": response.choices[0].message.content}


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("check_inventory", check_inventory_node)
    graph.add_node("draft_response", draft_response_node)
    graph.set_entry_point("check_inventory")
    graph.add_edge("check_inventory", "draft_response")
    graph.add_edge("draft_response", END)
    return graph.compile()