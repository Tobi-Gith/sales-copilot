# AI sales and service copilot - technical note

## 1. Overview

This prototype implements a RAG chat interface which answers questions based on a supplied set of artificial product, service and sales documents. 
It is combined with an agentic workflow for automatically drafting a customer-service response based on retrieved information and a human approval checkpoint. 

## 2. Architecture and technology choices 

| Component | Choice | Rationale |
| --- | --- | --- |
| Frontend | Streamlit | Enables a working chat UI in pure python within the given time budget |
| LLM and embeddings | OpenAI API ('gpt-4o-mini' and 'text-embedding-3-small') | fast, low-cost, combines text generation and embedding in one provider |
| Agent framework | LangGraph | structured approach with nodes allowing for failure handling, more realistic enterprise styls approach |
| Vector Store | ChromaDB (in-memory) | Runs entirely locally without external service etc., so minimal setup friction. In-memory because of avoiding persistence issues on streamlit community cloud when restarting |
| Document parsing | pypdf | allows using real pdf documents, which is more realistic and better for demonstration in front of the client |

**Prototype vs target architecture:** During development the prototype runs locally via `streamlit run app.py`. For submission it is additionally deployed to Streamlit Community cloud with a single shared API key, giving a publicly accessible URL. Section 5 describes how this would map to a proper hyperscaler deployment in production.

## 3. RAG implementation

**Document set:** 6 synthetic documents (2 product documents, 1 troubleshooting guide, 1 FAQ, 1 customer quote, 1 support ticket thread). Those cover three test cases: single document questions, questions requiring information from two documents, questions with no answer in the documents (to test grounded behavior).

**Chunking:** Fixed size, word based chunking (500 words, 50 words overlap). Overlapping is used to not miss information near a chunk boundary. Semantic chunking was considered but not necessary given the size and structure of the documents and the use as a prototype. Basic function definitions are in ingest.py and then used in the app. 

**Embeddings:** The model 'text-embedding-3-small' was used to compute once per chunk at app start and cached for the session, and once per user question at query time.

**Retrieval:** Cosine similarity was used with the three closest chunks against the in-memory chroma collection. The three closest ones are used to ensure a balance between getting all the relevant information and ensuring transparency because of the overall size of only 6 documents in total. 

**Generating and grounding:** Retrieved chunks are inserted into the prompt with explicit source tags. The system prompt instructs the model to answer only from the provided context and to explicitly say when information is not available which is the primary mechanism against hallucination. 

**Citations:** Each answer displays the set of the 3 documents used as sources with their respective names. Also the system prompt states to always name the source where the explicit infomation is taken from at each statement.

**Evaluation approach:** The system was manually tested against three categories of questions, corresponding to the three test cases in the document set. 

| Example question | Category | Result |
| --- | --- | --- |
| "Welche Betriebstemperatur verträgt die Pumpe X-123?" | single document required | correct answer, correctly cited to product data sheet only |
| "Was kostet die Pumpe X-123 und wie oft muss sie gewartet werden?" | two documents required | correct answer, correctly cited |
| "Welche Farbe hat die Pumpe X-123?" | Unanswerable | correctly stated that information is not available instead of guessing |

No automated evaluation framework was set up given the time budget. This is noted as one of the next steps in section 7.


## 4. Agentic workflow

**Scenario:** A customer service employee receives a customer question about a spare part. The workflow checks the (simulated) inventory for the relevant part and drafts a customer-facing response, which a human must approve before it is considered "sent".

**Steps (LangGraph nodes):** 

1. **`Extract part number`** - customers text question is passed to LLM which looks for the product number involved and returns it in JSON format (`response_format={"type": "json_object"}`). If no suitable product number is found by the LLM, a human needs to select the correct product number.
2. **`check_inventory`** - Looks up the extracted part number in the simulated inventory dictionary. This is wrapped in a retry loop with up to 2 retries to demonstrate resilience against transient failure. 
3. **`draft_response`** - Generates short customer-facing response in German, based only on verified part number and stock quantity, including approximate restock time if out of stock. 

**State:** A shared `TypeDict`(`WorkFlowState`) is passed between nodes, carrying the customer question, extracted part number, stock quantity, drafted response and a potential error. Each node checks for an existing error and skips its own logic if one exists, so a fail propagates cleanly to the end.

**Human approval checkpoint:** After the draft is generated, it is displayed to the user together with the part number and stock level that were used. The user has to decide to approve or reject - nothing is sent automatically. If approved, an approval message is displayed (and in reality the message would be send), if rejected, this is also displayed. Both the approval and the rejection are logged separately. There is then also the option to start a new workflow.

**Failure handling and fallback:** If `extract_part_number` cannot identify a known parameter with confidence, the workflow stops with an error instead of guessing. The user is then asked to provide the correct part number in a dropdown menu as a fallback, re-running only the remaining steps with the manually selected part number. If `check_inventory`cant find the part number in the inventory, there is a maximum of 2 retries before an error is returned and this is displayed.

**Grounding issue which was fixed:** Early testing revealed that supplying both customer question and verified inventory facts lead to drafting a wrong response when the part number had to be manually adjusted before. This was fixed by instructing the model to mark the verified facts as authoritative.

## 5. Hyperscaler deployment assumptions and miminum setup required

**Existing Cloud Infrastructure:** As stated, the organization has already initiated the migration of selected workloads to a public cloud hyperscaler (e.g. AWS, Azure, ...). The application is built in a modular, container-ready way (via streamlit and python) which can easily be integrated to existing cloud environments. Due to the lack of an in-house GenAI platform, the protoype relies on external APIs (openAI) which can also be changed to services offered by the respictive hyperscaler. 

An example for service mapping on the Azure infrastructure could be the following:

| Prototype component | Target Azure service |
| --- | --- |
| LLM & embeddings (OpenAI API) | Azure OpenAI Service |
| Vector store (in-memory ChromaDB) | Azure AI Search |
| Document storage (local `documents/` folder) | Azure Blob Storage |
| Secrets (`.env` file) | Azure Key Vault |
| Hosting (Streamlit Community Cloud) | Azure Container Apps via Docker|
| Logging (`logs.jsonl`) | Azure Monitor / Application Insights |

**Minimum setup for an initial pilot:** one Azure tenant with Azure OpenAI access approved, one Blob Storage container for source documents, one Azure AI Search instance, basic IAM roles for the application identity, and a single containerized deployment of the application.

**At scale:** separate dev/test/prod environments, an automated document ingestion pipeline (e.g. triggered on document upload rather than manual/full re-ingestion), load balancing for concurrent users, per-department access roles (see Section 6), and a monitoring dashboard tracking response quality and cost.


## 6. Security, Governance, responsible-AI controls

The client raised 5 specific concerns which are addressed as follows:

| Concern | approach |
| --- | --- |
| data confidentiality | Only synthetic documents in prototype. Data is logged without confidential information by finding and replacing it with placeholders using RegEx. In reality, data would remain in Hyperscaler tenant, no training on documents. |
| access controlls | Not implemented in prototype, because only synthetic documents. In reality there is a login which controlls for access permissions (e.g. metadata tags per document) |
| hallucinations | Addressed via grounded RAG prompting, citations of sources and in the agentic workflow with the precedence given to verified facts (see section 4) |
| operational ownership | There should be a cross-functional AI-platform team monitoring and updating the system. |
| vendor lock-in | the client is isolated behind the application's own functions (ingest.py, agent.py) which makes swapping relatively easy. |

Responsibility and transparency is in general also adressd by the telemetry which lets you track everything down to the sources.

**Secrets management:** The API key is stored in local `.env` file which is excluded from version control via `.gitignore` and never hard-coded. On the streamlit community cloud it is stored in the platform's separate "secrets" section. This can also be done in e.g. Azure or other hyperscaler services.

**Logging:** Every RAG query logs the prompt, retrieved sources, and response. Every agentic workflow run logs the extracted/selected part number, stock result, and any error. The human approval decision (approved/rejected) is logged separately from the automated run. Errors in either flow are caught and logged with the error message. All entries are timestamped in `Europe/Berlin` local time and written as JSON Lines (`logs.jsonl`).


## 7. Limitations, trade-offs, and next improvements

TODO

## 8. Run instructions

TODO