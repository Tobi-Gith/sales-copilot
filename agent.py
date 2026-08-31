import json
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

INVENTORY = {"X-123": 12, "Y-456": 0}


class WorkflowState(TypedDict):
    customer_question: str
    part_number: Optional[str]
    stock_quantity: int
    draft: str
    error: str


def extract_part_number_node(state: WorkflowState) -> WorkflowState:
    known_parts = list(INVENTORY.keys())
    prompt = (
        f"Bekannte Ersatzteilnummern: {', '.join(known_parts)}.\n"
        f"Kundenanfrage: \"{state['customer_question']}\"\n"
        "Welche dieser Teilenummern ist gemeint? Antworte ausschliesslich als JSON: "
        '{"part_number": "<Teilenummer>"}. Falls keine eindeutig erkennbar ist, '
        'antworte mit {"part_number": null}.'
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )
    parsed = json.loads(response.choices[0].message.content)
    part_number = parsed.get("part_number")

    if part_number not in known_parts:
        return {**state, "part_number": None,
                "error": "Konnte kein bekanntes Ersatzteil eindeutig aus der Anfrage erkennen."}
    return {**state, "part_number": part_number, "error": ""}


def check_inventory_node(state: WorkflowState) -> WorkflowState:
    if state.get("error"):
        return state 

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

    prompt = (
        f"Kundenanfrage (Originaltext, kann von den unten genannten verifizierten Fakten "
        f"abweichende Bezeichnungen enthalten): \"{state['customer_question']}\"\n\n"
        f"Verifizierte Fakten (verbindlich, nutze ausschliesslich diese Werte):\n"
        f"- Ersatzteilnummer: {state['part_number']}\n"
        f"- Lagerbestand: {state['stock_quantity']} Stueck\n\n"
        "Formuliere eine kurze, freundliche Kundenantwort auf Deutsch. Beziehe dich dabei "
        "ausschliesslich auf die oben genannte Ersatzteilnummer und den Lagerbestand - "
        "nicht auf eine eventuell abweichende Bezeichnung aus der Kundenanfrage. "
        "Wenn der Lagerbestand groesser als 0 ist, bestaetige die Verfuegbarkeit mit der "
        "genannten Stueckzahl. Wenn der Lagerbestand 0 ist, informiere hoeflich ueber die "
        "Nichtverfuegbarkeit und eine ungefaehre Nachlieferzeit von 2 Wochen. "
        "Erfinde keine zusaetzlichen Informationen, insbesondere keine anderen Zahlen."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return {**state, "draft": response.choices[0].message.content}


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("extract_part_number", extract_part_number_node)
    graph.add_node("check_inventory", check_inventory_node)
    graph.add_node("draft_response", draft_response_node)
    graph.set_entry_point("extract_part_number")
    graph.add_edge("extract_part_number", "check_inventory")
    graph.add_edge("check_inventory", "draft_response")
    graph.add_edge("draft_response", END)
    return graph.compile()

def build_manual_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("check_inventory", check_inventory_node)
    graph.add_node("draft_response", draft_response_node)
    graph.set_entry_point("check_inventory")
    graph.add_edge("check_inventory", "draft_response")
    graph.add_edge("draft_response", END)
    return graph.compile()