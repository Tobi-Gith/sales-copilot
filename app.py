import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from ingest import load_and_chunk_documents, build_vector_store

load_dotenv()
client = OpenAI()

st.title("Sales & Service Copilot (Prototyp)")

@st.cache_resource
def get_vector_store():
    chunks = load_and_chunk_documents()
    return build_vector_store(chunks)

collection = get_vector_store()

question = st.text_input("Deine Frage an die Dokumentation:")

if question:
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

    st.write(response.choices[0].message.content)
    st.caption("Quellen: " + ", ".join(set(sources)))