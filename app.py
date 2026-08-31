import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from ingest import load_and_chunk_documents, build_vector_store
from logging_utils import log_event, mask_pii
from agent import build_graph, INVENTORY

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

collection = get_vector_store()
workflow_graph = get_agent_graph()

tab1, tab2 = st.tabs(["Fragen & Antworten", "Agentic Workflow"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()

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

    question = st.chat_input("Deine Frage an die Dokumentation:")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

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
            st.rerun()

        except Exception as e:
            log_event("error", {
                "prompt": mask_pii(question),
                "error_message": str(e),
            })
            error_msg = "Entschuldigung, bei der Verarbeitung deiner Frage ist ein Fehler aufgetreten. Bitte versuche es erneut."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.rerun()

with tab2:
    st.subheader("Ersatzteil-Anfrage bearbeiten")
    st.caption(f"Simulierter Lagerbestand: {INVENTORY}")

    customer_question = st.text_area(
        "Kundenanfrage:", "Ist Ersatzteil X-123 verfuegbar und wie lange dauert die Lieferung?"
    )
    part_number = st.selectbox("Betroffenes Ersatzteil:", list(INVENTORY.keys()))

    if st.button("Workflow starten"):
        try:
            result = workflow_graph.invoke({
                "customer_question": customer_question,
                "part_number": part_number,
                "stock_quantity": 0,
                "draft": "",
                "error": "",
            })
            log_event("agentic_workflow_run", {
                "customer_question": mask_pii(customer_question),
                "part_number": part_number,
                "stock_quantity": result.get("stock_quantity"),
                "error": result.get("error", ""),
            })
            st.session_state.workflow_result = result
            st.session_state.workflow_decided = False
        except Exception as e:
            log_event("error", {"context": "agentic_workflow", "error_message": str(e)})
            st.error("Beim Ausfuehren des Workflows ist ein Fehler aufgetreten.")

    if "workflow_result" in st.session_state:
        result = st.session_state.workflow_result

        if result.get("error"):
            st.error(f"Workflow-Fehler: {result['error']}")
        else:
            st.write(f"**Lagerbestand fuer {part_number}:** {result['stock_quantity']} Stueck")
            st.write("**Entwurf zur Pruefung:**")
            st.info(result["draft"])

            if not st.session_state.get("workflow_decided", False):
                col1, col2 = st.columns(2)
                if col1.button("✅ Freigeben und senden"):
                    log_event("agentic_workflow_approval", {
                        "decision": "approved",
                        "draft": mask_pii(result["draft"]),
                    })
                    st.session_state.workflow_decided = True
                    st.success("Freigegeben und (simuliert) gesendet.")
                if col2.button("❌ Ablehnen"):
                    log_event("agentic_workflow_approval", {
                        "decision": "rejected",
                        "draft": mask_pii(result["draft"]),
                    })
                    st.session_state.workflow_decided = True
                    st.warning("Entwurf abgelehnt. Kein Versand erfolgt.")