import os, glob
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import chromadb

load_dotenv()
client = OpenAI()

def load_and_chunk_documents(folder="documents", chunk_size=500, overlap=50):
    all_chunks = []
    for filepath in glob.glob(f"{folder}/*.pdf"):
        reader = PdfReader(filepath)
        text = "\n".join(page.extract_text() for page in reader.pages)
        words = text.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            all_chunks.append({"text": chunk, "source": os.path.basename(filepath)})
            if end >= len(words):
                break
            start += chunk_size - overlap
    return all_chunks

def build_vector_store(chunks):
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection("docs")
    for i, chunk in enumerate(chunks):
        embedding = client.embeddings.create(
            model="text-embedding-3-small", input=chunk["text"]
        ).data[0].embedding
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"]}]
        )
    return collection

if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    print(f"{len(chunks)} Chunks aus den PDFs erzeugt.")
    collection = build_vector_store(chunks)
    print("Vector Store erfolgreich befuellt (In-Memory, nur fuer diesen Lauf gueltig).")