# Clinical Trial Inspector Agent 🕵️‍♂️💊

**Clinical Trial Inspector** is an advanced AI agent designed to revolutionize how researchers, clinicians, and analysts explore clinical trial data. By combining **Semantic Search**, **Retrieval-Augmented Generation (RAG)**, and **Visual Analytics**, it transforms raw data from [ClinicalTrials.gov](https://clinicaltrials.gov/) into actionable insights.

Built with **LangChain**, **LlamaIndex**, **Streamlit**, **Altair**, and **Google Gemini**, this tool goes beyond simple keyword search. It understands natural language, generates inline visualizations, and performs complex multi-dimensional analysis on the entire dataset.

## ✨ Key Features

### 🧠 Intelligent Search & RAG
- **Natural Language Queries**: Ask complex questions like *"Find Phase 3 Pfizer studies for diabetes started after 2022"* or *"What are the inclusion criteria for recent Moderna trials?"*.
- **Semantic Understanding**: Powered by **PubMedBERT** embeddings (`pritamdeka/S-PubMedBert-MS-MARCO`) to understand medical context better than keyword matching.
- **AI Synthesis**: **Google Gemini (gemini-2.5-flash)** summarizes findings, compares studies, and answers follow-up questions with citations.

### 📊 Visual Analytics & Insights
- **Inline Charts (Contextual)**: The agent automatically generates **Bar Charts** and **Line Charts** directly in the chat stream when you ask aggregation questions (e.g., *"Top sponsors for Multiple Myeloma"*).
- **Sidebar Dashboard (Global)**: The "Analytics & Export" sidebar provides a high-level view of the **entire dataset** (60,000+ studies). It is always available and independent of your current chat query, allowing you to explore global trends while conversing.
- **Precise Formatting**: Year-based trends are displayed with precision (e.g., "2023") using **Altair**.

### 🔍 Multi-Filter Analysis
- **Complex Filtering**: Answer sophisticated questions by applying multiple filters simultaneously.
    - *Example*: *"For **Phase 2 and 3** studies, what are **Pfizer's** most common study indications?"*
- **Full Dataset Scope**: General analytics questions (e.g., *"Who are the top sponsors in the dataset?"*) analyze the **entire 60,000+ study database**, not just a sample.
- **Smart Retrieval**: For specific queries, the agent retrieves up to **5,000 relevant studies** to ensure comprehensive analysis.

### 📂 Comprehensive Data Management
- **Raw Data Export**: View and download the full dataset as a CSV, including **NCT ID**, **Title**, **Sponsor**, **Phase**, and **Conditions**.
- **Local Vector Store**: Efficiently stores and retrieves tens of thousands of studies using **ChromaDB**.

## 🤖 Agent Capabilities & Tools

The agent is equipped with specialized tools to handle different types of requests:

### 1. `search_trials`
*   **Purpose**: Finds specific clinical trials based on natural language queries.
*   **Capabilities**:
    *   **Semantic Search**: Uses vector embeddings to find relevant studies even if keywords don't match exactly.
    *   **Smart Filtering**: Automatically extracts filters for **Phase**, **Status**, and **Sponsor** from your query (e.g., "Recruiting Phase 3 studies by Moderna").
    *   **Limit**: Returns the top 50 most relevant results for detailed inspection.

### 2. `get_study_analytics`
*   **Purpose**: Aggregates data to reveal trends and insights.
*   **Capabilities**:
    *   **Multi-Filtering**: Can filter by **Phase**, **Status**, and **Sponsor** *before* aggregation (e.g., "Phase 2 studies by Pfizer").
    *   **Full Dataset Access**: For general questions (e.g., "Top sponsors overall"), it scans the entire 60,000+ study database.
    *   **Visuals**: Triggers inline **Bar** and **Line** charts in the chat.

### 3. `find_similar_studies`
*   **Purpose**: Discovers studies that are semantically similar to a specific trial.
*   **Capabilities**:
    *   **Vector Similarity**: Calculates cosine similarity between study descriptions.
    *   **Contextual**: Great for finding "more like this" when you've identified a study of interest.

## ⚙️ How It Works (RAG Pipeline)

1.  **Ingestion**: `ingest_ct.py` fetches study data from ClinicalTrials.gov. It creates a rich text representation of each study (Title, Summary, Criteria) and extracts structured metadata (Phase, Sponsor, Status).
2.  **Embedding**: The text is converted into vector embeddings using `PubMedBERT`, a model optimized for biomedical text. These vectors are stored locally in **ChromaDB**.
3.  **Retrieval (Hybrid Search)**: The agent uses a multi-stage retrieval process to ensure both semantic relevance and strict criteria matching:
    *   **Semantic Search**: Your query is embedded and compared against the database to find conceptually similar studies (e.g., "heart failure" matches "cardiac insufficiency").
    *   **Pre-Retrieval Filtering**: Strict filters (Status, Year, NCT ID) are applied *before* vector search to narrow the search space efficiently.
    *   **Post-Retrieval Filtering**: Complex logic (e.g., "Phase 2 or 3", Sponsor aliases) is applied *after* fetching candidates to ensure precision.
    *   **Re-Ranking**: A Cross-Encoder model (`ms-marco-MiniLM`) re-scores the final results to rank the most relevant studies at the top.
4.  **Synthesis**: The top relevant studies are passed to **Google Gemini**, which synthesizes the information to answer your specific question, citing the source studies.

## 🛠️ Tech Stack

- **Frontend**: Streamlit, Altair
- **LLM**: Google Gemini (`gemini-2.5-flash`)
- **Orchestration**: LangChain (Agents, Tool Calling)
- **Retrieval (RAG)**: LlamaIndex (VectorStoreIndex, QueryEngine)
- **Vector Database**: ChromaDB (Local)
- **Embeddings**: HuggingFace (`pritamdeka/S-PubMedBert-MS-MARCO`)

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Google Cloud API Key with access to Gemini

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd clinical_trial_agent
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables**
   Create a `.env` file in the root directory and add your Google API Key:
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## 📖 Usage

### 1. Ingest Data
Before running the agent, populate the local database. The ingestion script fetches studies from ClinicalTrials.gov and builds the vector index.

```bash
# Recommended: Ingest 5000 recent studies
python ingest_ct.py --limit 5000 --years 5

# Ingest ALL studies (Warning: Large download!)
python ingest_ct.py --limit -1
```

### 2. Run the Agent
Launch the Streamlit application:

```bash
streamlit run ct_agent_app.py
```

The app will open in your browser at `http://localhost:8501`.

### 3. Ask Questions!
Try these queries to see the agent in action:

- **Search**: *"Find recruiting studies for Alzheimer's in the US."*
- **Analytics**: *"Who are the top sponsors for Breast Cancer?"* (Triggers Inline Chart)
- **Multi-Filter**: *"For Phase 3 studies, what are the top conditions studied by Merck?"*
- **Trends**: *"How has the number of gene therapy studies changed over time?"*

## 📂 Project Structure

- `ct_agent_app.py`: Main application logic, including the LangChain agent, Streamlit UI, and Analytics tools.
- `ingest_ct.py`: Data pipeline script for fetching, processing, and embedding ClinicalTrials.gov data.
- `ct_gov_index/`: Persisted ChromaDB vector store.
- `requirements.txt`: Project dependencies.

## ⚠️ Note on Quotas
This project uses the free tier of Google Gemini API. If you encounter a "ResourceExhausted" error, please wait a minute before retrying.
