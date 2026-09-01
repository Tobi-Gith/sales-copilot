import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from ingest import load_and_chunk_documents, build_vector_store
from logging_utils import log_event, mask_pii
from agent import build_graph, build_manual_graph, INVENTORY

load_dotenv()
client = OpenAI()

st.title("Sales & Service Copilot (Prototyp)")
st.markdown('<div id="top"></div>', unsafe_allow_html=True)
st.markdown("""
<style>
#jump-to-top {
    position: fixed;
    bottom: 25px;
    right: 25px;
    background-color: #0e6efd;
    color: white;
    padding: 12px 16px;
    border-radius: 50%;
    text-decoration: none;
    font-size: 20px;
    font-weight: bold;
    z-index: 1000;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}
</style>
<a id="jump-to-top" href="#top">⬆</a>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vector_store():
    chunks = load_and_chunk_documents()
    return build_vector_store(chunks)

@st.cache_resource
def get_agent_graph():
    return build_graph()

@st.cache_resource
def get_manual_agent_graph():
    return build_manual_graph()

collection = get_vector_store()
workflow_graph = get_agent_graph()
manual_workflow_graph = get_manual_agent_graph()

tab1, tab2 = st.tabs(["Fragen & Antworten", "Agentic Workflow"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()

    chat_container = st.container(height=500)
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and i not in st.session_state.feedback_given:
                    col1, col2 = st.columns([1, 10])
                    if col1.button("👍", key=f"up_{i}"):
                        log_event("user_feedback", {"message_index": i, "feedback": "positive"})
                        st.session_state.feedback_given.add(i)
                        st.rerun()
                    if col2.button("👎", key=f"down_{i}"):
                        log_event("user_feedback", {"message_index": i, "feedback": "negative"})
                        st.session_state.feedback_given.add(i)
                        st.rerun()

        # Falls die letzte Nachricht eine noch unbeantwortete Nutzerfrage ist: jetzt beantworten
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            question = st.session_state.messages[-1]["content"]
            with st.chat_message("assistant"):
                with st.spinner("Antwort wird generiert..."):
                    try:
                        q_embedding = client.embeddings.create(
                            model="text-embedding-3-small", input=question
                        ).data[0].embedding

                        results = collection.query(query_embeddings=[q_embedding], n_results=3)
                        chunks = results["documents"][0]
                        sources = [m["source"] for m in results["metadatas"][0]]
                        context = "\n\n".join(f"[Quelle: {s}]\n{c}" for s, c in zip(sources, chunks))

                        system_prompt = (
                            "Du bist ein Assistent fuer Vertriebs- und Servicemitarbeiter. "
                            "Antworte ausschliesslich basierend auf dem gegebenen Kontext. "
                            "Wenn die Information nicht im Kontext steht, sage das explizit "
                            "statt zu raten. Nenne bei jeder Aussage die Quelle in eckigen Klammern."
                        )

                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"Kontext:\n{context}\n\nFrage: {question}"}
                            ]
                        )
                        answer = response.choices[0].message.content
                        sources_line = "\n\n*Quellen: " + ", ".join(set(sources)) + "*"
                        full_answer = answer + sources_line

                        log_event("rag_query", {
                            "prompt": mask_pii(question),
                            "retrieved_sources": list(set(sources)),
                            "response": mask_pii(answer),
                        })
                        st.session_state.messages.append({"role": "assistant", "content": full_answer})

                    except Exception as e:
                        log_event("error", {
                            "prompt": mask_pii(question),
                            "error_message": str(e),
                        })
                        error_msg = "Entschuldigung, bei der Verarbeitung deiner Frage ist ein Fehler aufgetreten. Bitte versuche es erneut."
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

            st.rerun()

    new_question = st.chat_input("Deine Frage an die Dokumentation:")
    if new_question:
        st.session_state.messages.append({"role": "user", "content": new_question})
        st.rerun()

with tab2:
    st.subheader("Ersatzteil-Anfrage bearbeiten")
    st.caption(f"Simulierter Lagerbestand: {INVENTORY}")

    if "workflow_result" not in st.session_state:
        customer_question = st.text_area(
            "Kundenanfrage:", "", placeholder="Ist Ersatzteil X-123 verfuegbar und wie lange dauert die Lieferung?"
        )

        if st.button("Workflow starten", disabled=not customer_question.strip()):
            try:
                result = workflow_graph.invoke({
                    "customer_question": customer_question,
                    "part_number": None,
                    "stock_quantity": 0,
                    "unit_price": 0.0,
                    "draft": "",
                    "error": "",
                })
                log_event("agentic_workflow_run", {
                    "customer_question": mask_pii(customer_question),
                    "part_number": result.get("part_number"),
                    "stock_quantity": result.get("stock_quantity"),
                    "error": result.get("error", ""),
                })
                st.session_state.workflow_result = result
                st.session_state.workflow_customer_question = customer_question
                st.session_state.workflow_decided = False
                st.rerun()
            except Exception as e:
                log_event("error", {"context": "agentic_workflow", "error_message": str(e)})
                st.error("Beim Ausfuehren des Workflows ist ein Fehler aufgetreten.")

    else:
        result = st.session_state.workflow_result
        customer_question = st.session_state.workflow_customer_question

        st.write(f"**Kundenanfrage:** {customer_question}")

        if result.get("error"):
            st.error(f"Workflow-Fehler: {result['error']}")
            st.caption("Automatische Erkennung fehlgeschlagen – bitte Teilenummer manuell waehlen und erneut starten.")
            manual_part = st.selectbox("Ersatzteil manuell auswaehlen:", list(INVENTORY.keys()))
            if st.button("Mit manueller Auswahl erneut versuchen"):
                result2 = manual_workflow_graph.invoke({
                    "customer_question": customer_question,
                    "part_number": manual_part,
                    "stock_quantity": 0,
                    "unit_price": 0.0,
                    "draft": "",
                    "error": "",
                })
                log_event("agentic_workflow_run", {
                    "customer_question": mask_pii(customer_question),
                    "part_number": manual_part,
                    "stock_quantity": result2.get("stock_quantity"),
                    "error": result2.get("error", ""),
                    "manual_override": True,
                })
                st.session_state.workflow_result = result2
                st.rerun()
        else:
            st.caption(f"Erkanntes Ersatzteil: {result['part_number']}")
            st.write(f"**Lagerbestand:** {result['stock_quantity']} Stueck")
            st.write(f"**Stueckpreis (netto):** {result['unit_price']:.2f} EUR")
            st.write("**Entwurf zur Pruefung:**")
            st.info(result["draft"])

            if not st.session_state.workflow_decided:
                col1, col2 = st.columns(2)
                if col1.button("✅ Freigeben und senden"):
                    log_event("agentic_workflow_approval", {"decision": "approved", "draft": mask_pii(result["draft"])})
                    st.session_state.workflow_decided = True
                    st.session_state.workflow_decision = "approved"
                    st.rerun()
                if col2.button("❌ Ablehnen"):
                    log_event("agentic_workflow_approval", {"decision": "rejected", "draft": mask_pii(result["draft"])})
                    st.session_state.workflow_decided = True
                    st.session_state.workflow_decision = "rejected"
                    st.rerun()
            else:
                if st.session_state.workflow_decision == "approved":
                    st.success("Freigegeben und (simuliert) gesendet.")
                else:
                    st.warning("Entwurf abgelehnt. Kein Versand erfolgt.")

        if st.button("🔄 Neue Anfrage starten"):
            for key in ["workflow_result", "workflow_customer_question", "workflow_decided", "workflow_decision"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()