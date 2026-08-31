# AI sales and service copilot - technical note

## 1. Overview

This prototype implements a RAG chat interface which answers questions based on a supplied set of artificial product, service and sales documents. 
It is combined with an agentic workflow for automatically drafting a customer-service response based on retrieved information and a human approval checkpoint. 

## 2. Architecture and technology choices 

| Component | Choice | Rationale |
| --- | --- | --- |
| Frontend | Streamlit | Enables a working chat UI in pure python within the given time budget |
| LLM and embeddings | OpenAI API ('gpt-4o-mini' and 'text-embedding-3-small') | fast, low-cost, combines text generation and embedding in one provider |
| Agent framework | --- | --- |
| Vector Store | ChromaDB (in-memory) | Runs entirely locally without external service etc., so minimal setup friction. In-memory because of avoiding persistence issues on streamlit community cloud when restarting |
| Document parsing | pypdf | allows using real pdf documents, which is more realistic and better for demonstration in front of the client |

**Prototype vs target architecture:** The protoype intentionally runs on a single cloud-hosted Streamlit app with one API-key. Section 5 describes how this would map to proper hyperscaler deployment.

## 3. RAG implementation

**Document set:** 6 synthetic documents (2 product documents, 1 troubleshooting guide, 1 FAQ, 1 customer quote, 1 support ticket thread). Those cover three test cases: single document questions, questions requiring information from two documents, questions with no answer in the documents (to test grounded behavior).

**Chunking:** Fixed size, word based chunking (500 words, 50 words overlap). Overlapping is used to not miss information near a chunk boundary. Semantic chunking was considered but not necessary given the size and structure of the documents and the use as a prototype. Basic function definitions are in ingest.py and then used in the app. 

**Embeddings:** The model 'text-embedding-3-small' was used to compute once per chunk at app start and cached for the session, and once per user question at query time.

**Retrieval:** Cosine similarity was used with the three closest chunks against the in-memory chroma collection. The three closest ones are used to ensure a balance between getting all the relevant information and ensuring transparency because of the overall size of only 6 documents in total. 

**Generating and grounding:** Retrieved chunks are inserted into the prompt with explicit source tags. The system prompt instructs the model to answer only from the provided context and to explicitly say when information is not available which is the primary mechanism against hallucination. 

**Citations:** Each answer displays the set of the 3 documents used as sources with their respective names. Also the system prompt states to always name the source where the explicit infomation is taken from at each statement.

**Evaluation approach:** TODO

## 4. Agentic workflow

TODO

## 5. Hyperscaler deployment assumptions and miminum setup required

TODO

## 6. Security, Governance, responsible-AI controls

TODO 

## 7. Limitations, trade-offs, and next improvements

TODO

## 8. Run instructions

TODO