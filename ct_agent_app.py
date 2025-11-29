
import streamlit as st
import pandas as pd
import os
import altair as alt

__version__ = "1.0.0"

import logging
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from typing import List, Optional
import chromadb

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="Clinical Trial Inspector", layout="wide")
st.title("🧬 Clinical Trial Inspector Agent")

# 1. Setup LLM (Gemini) & Embeddings (Local)


# 1. Setup LLM
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Please set GOOGLE_API_KEY in .env")
    st.stop()

# Configure LlamaIndex to use Gemini
Settings.llm = Gemini(model="models/gemini-2.5-flash", temperature=0)
Settings.embed_model = HuggingFaceEmbedding(model_name="pritamdeka/S-PubMedBert-MS-MARCO")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 2. Load LlamaIndex
@st.cache_resource
def load_index():
    print("🧠 Loading LlamaIndex...")
    # Initialize Embedding Model
    embed_model = HuggingFaceEmbedding(model_name="pritamdeka/S-PubMedBert-MS-MARCO")
    
    # Initialize ChromaDB Client
    db = chromadb.PersistentClient(path="./ct_gov_index")
    chroma_collection = db.get_or_create_collection("clinical_trials")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Load Index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index

index = load_index()

# 3. Create LlamaIndex Tool
from langchain.tools import tool as langchain_tool
from langchain_core.prompts import MessagesPlaceholder
 
def normalize_sponsor(sponsor: str) -> str:
    """Normalizes sponsor names to handle aliases."""
    if not sponsor:
        return None
    
    s = sponsor.lower().strip()
    aliases = {
        "gsk": "GlaxoSmithKline",
        "glaxo": "GlaxoSmithKline",
        "glaxosmithkline": "GlaxoSmithKline",
        "j&j": "Janssen",
        "johnson & johnson": "Janssen",
        "johnson and johnson": "Janssen",
        "janssen": "Janssen",
        "bms": "Bristol-Myers Squibb",
        "bristol myers squibb": "Bristol-Myers Squibb"
    }
    
    for alias, canonical in aliases.items():
        if alias in s:
            return canonical
    return sponsor


class LocalMetadataPostFilter(BaseNodePostprocessor):
    phase: Optional[str] = None
    sponsor: Optional[str] = None
    
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle=None
    ) -> List[NodeWithScore]:
        filtered_nodes = []
        for node in nodes:
            meta = node.metadata
            match = True
            
            # Handle Phase (Multi-value support)
            if self.phase:
                # Split input "PHASE2, PHASE3" -> ["phase2", "phase3"]
                target_phases = [p.strip().lower() for p in self.phase.split(',')]
                node_phase = meta.get("phase", "").lower()
                # Check if ANY of the target phases are in the node's phase string
                if not any(tp in node_phase for tp in target_phases):
                    match = False
            
            # Handle Sponsor (Multi-value support & Normalization)
            if self.sponsor:
                 # Normalize input sponsors
                 raw_sponsors = [s.strip() for s in self.sponsor.split(',')]
                 target_sponsors = [normalize_sponsor(s).lower() for s in raw_sponsors]
                 
                 node_sponsor = meta.get("org", "").lower()
                 if not any(ts in node_sponsor for ts in target_sponsors):
                     match = False
            
            if match:
                filtered_nodes.append(node)
        return filtered_nodes

@langchain_tool("search_trials")
def search_trials(query: str = None, status: str = None, phase: str = None, sponsor: str = None, year: int = None):
    """
    Searches for clinical trials using semantic search with optional strict filters.
    
    Args:
        query: The search query (e.g., "diabetes treatment"). Optional if filters are provided.
        status: Filter by status (e.g., "RECRUITING", "COMPLETED").
        phase: Filter by phase (e.g., "PHASE2", "PHASE3"). Accepts comma-separated values for multiple phases.
        sponsor: Filter by sponsor name (e.g., "Pfizer"). Accepts comma-separated values.
        year: Filter for studies starting after this year (e.g., 2020).
    """
    # Handle missing query by constructing one from filters
    if not query:
        parts = []
        if sponsor: parts.append(sponsor)
        if phase: parts.append(phase)
        if status: parts.append(status)
        query = " ".join(parts) if parts else "clinical trial"
    else:
        # Inject sponsor and phase into query if they are not already present
        # This improves semantic retrieval recall for specific entities
        if sponsor:
            norm_sponsor = normalize_sponsor(sponsor)
            if norm_sponsor.lower() not in query.lower():
                query = f"{norm_sponsor} {query}"
        if phase and phase.lower() not in query.lower():
            query = f"{phase} {query}"

    # 1. Pre-retrieval filters (supported by Chroma)
    filters = []
    
    # Detect if query is an NCT ID
    import re
    nct_match = re.search(r"NCT\d+", query, re.IGNORECASE)
    if nct_match:
        nct_id = nct_match.group(0).upper()
        print(f"🎯 Detected NCT ID: {nct_id}. Switching to exact match.")
        filters.append(MetadataFilter(key="nct_id", value=nct_id, operator=FilterOperator.EQ))
        # If looking for a specific ID, we don't need semantic search as much, but we keep it for context if needed.
        # However, filter is strict, so it will only return that doc.
    
    if status:
        filters.append(MetadataFilter(key="status", value=status.upper(), operator=FilterOperator.EQ))
    if year:
        filters.append(MetadataFilter(key="start_year", value=year, operator=FilterOperator.GTE))
        
    metadata_filters = MetadataFilters(filters=filters) if filters else None
    
    print(f"🔍 Tool Called: search_trials(query='{query}', status='{status}', phase='{phase}', sponsor='{sponsor}')")

    # 2. Post-retrieval filters (custom logic)
    post_filters = []
    if phase or sponsor:
        post_filters.append(LocalMetadataPostFilter(phase=phase, sponsor=sponsor))
    
    # 3. Re-ranker
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5) # Increased top_n
    post_filters.append(reranker)
    
    # Increase retrieval limit if filters are present to ensure we get enough candidates
    # If we have post-filters, we need a larger pool.
    top_k = 200 if (phase or sponsor) else 50
    
    query_engine = index.as_query_engine(
        similarity_top_k=top_k, 
        node_postprocessors=post_filters,
        filters=metadata_filters
    )
    response = query_engine.query(query)
    print(f"✅ Retrieved {len(response.source_nodes)} nodes after filtering.")
    return str(response)

@langchain_tool("find_similar_studies")
def find_similar_studies(query: str):
    """
    Finds studies similar to a given query or study description. 
    Returns the top studies with their similarity scores.
    """
    # Initialize Re-ranker
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5)
    
    # Check if query is an NCT ID
    import re
    nct_match = re.search(r"NCT\d+", query, re.IGNORECASE)
    search_query = query
    
    if nct_match:
        nct_id = nct_match.group(0).upper()
        print(f"🎯 Detected NCT ID for similarity: {nct_id}. Fetching content...")
        
        # Fetch the content of the study
        filters = [MetadataFilter(key="nct_id", value=nct_id, operator=FilterOperator.EQ)]
        retriever = index.as_retriever(filters=MetadataFilters(filters=filters))
        nodes = retriever.retrieve(nct_id)
        
        if nodes:
            search_query = nodes[0].text
            print(f"✅ Found content for {nct_id}. Using it for similarity search.")
        else:
            print(f"⚠️ Could not find content for {nct_id}. Using original query.")

    query_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[reranker]
    )
    response = query_engine.query(f"Find studies similar to: {search_query}")
    
    # Format the response to include scores if available (LlamaIndex response might need parsing or custom retrieval)
    # For better control, let's use the retriever directly to get scores
    retriever = index.as_retriever(similarity_top_k=10, node_postprocessors=[reranker])
    nodes = retriever.retrieve(f"Find studies similar to: {search_query}")
    
    result_str = f"Found {len(nodes)} similar studies:\n\n"
    for i, node in enumerate(nodes):
        # Filter out the original study if it appears in results
        if nct_match and nct_id in node.metadata.get('nct_id', ''):
            continue
            
        score = node.score if node.score else 0.0
        meta = node.metadata
        result_str += (
            f"{i+1}. **{meta.get('nct_id')}** (Score: {score:.4f})\n"
            f"   - **Title**: {meta.get('title')}\n"
            f"   - **Status**: {meta.get('status')}\n"
        f"   - **Summary**: {node.text[:200]}...\n\n" # Provide snippet for comparison
        )
        
    return result_str

@langchain_tool("get_study_analytics")
def get_study_analytics(query: str, group_by: str, phase: Optional[str] = None, status: Optional[str] = None, sponsor: Optional[str] = None) -> str:
    """
    Aggregates clinical trial data based on a search query and groups by a specific field.
    Supports optional filtering by phase, status, and sponsor before grouping.
    
    Args:
        query: The search topic (e.g., "Multiple Myeloma", "Pfizer", "dataset").
        group_by: The field to group by. Options: "phase", "org" (sponsor), "status", "condition", "start_year", "study_type".
        phase: Optional. Filter by phase(s) (e.g., "Phase 2, Phase 3"). Case-insensitive.
        status: Optional. Filter by status (e.g., "Completed"). Case-insensitive.
        sponsor: Optional. Filter by sponsor (e.g., "Pfizer"). Case-insensitive.
        
    Returns:
        A summary string of the top counts and a note that a chart has been generated.
    """
    print(f"📊 Tool Called: get_study_analytics(query='{query}', group_by='{group_by}', phase='{phase}', status='{status}', sponsor='{sponsor}')")
    
    # Normalize query if it looks like a sponsor
    norm_query = normalize_sponsor(query)
    if norm_query != query:
        print(f"   Normalized query '{query}' to '{norm_query}'")
        query = norm_query
    
    # Check for "general" or "dataset" queries
    is_general = False
    general_terms = ["dataset", "all studies", "general", "overall"]
    if not query or query.strip() == "" or any(t in query.lower() for t in general_terms):
        is_general = True
        print("   Detected general analytics request. Using full dataset.")
    
    data = []
    if is_general:
        # Fetch ALL metadata directly
        try:
            db_client = chromadb.PersistentClient(path="./ct_gov_index")
            collection = db_client.get_collection("clinical_trials")
            all_docs = collection.get(include=['metadatas'])
            data = all_docs['metadatas']
        except Exception as e:
            return f"Error fetching full dataset: {e}"
    else:
        # Fetch a VERY LARGE number of results for specific aggregation (effectively "all relevant")
        # 5000 should cover almost any specific condition/sponsor subset
        retriever = index.as_retriever(similarity_top_k=5000)
        nodes = retriever.retrieve(query)
        
        if not nodes:
            return "No studies found to analyze."
            
        for node in nodes:
            data.append(node.metadata)
        
    df = pd.DataFrame(data)
    
    # --- APPLY FILTERS ---
    if phase:
        # Split by comma, strip, upper, remove spaces to match metadata format (e.g. "PHASE2")
        target_phases = [p.strip().upper().replace(" ", "") for p in phase.split(',')]
        # Filter: check if any target phase is in the study's phase string
        # Handle NaN/None in 'phase' column
        df['phase_upper'] = df['phase'].astype(str).str.upper().str.replace(" ", "")
        # We want rows where the study phase contains ANY of the target phases
        mask = df['phase_upper'].apply(lambda x: any(tp in x for tp in target_phases))
        df = df[mask]
        
    if status:
        target_status = status.strip().lower()
        df = df[df['status'].astype(str).str.lower() == target_status]
        
    if sponsor:
        # Normalize target sponsor
        target_sponsor = normalize_sponsor(sponsor).lower()
        # Normalize dataframe sponsor column
        df['org_lower'] = df['org'].astype(str).apply(normalize_sponsor).str.lower()
        # Exact match (or contains? Sponsor names can be long. Let's try contains for flexibility, or exact for precision)
        # Given normalize_sponsor handles aliases, exact match on normalized name is safer to avoid "Pfizer" matching "Pfizer Foundation" if unwanted.
        # But "Pfizer" should match "Pfizer"
        df = df[df['org_lower'].str.contains(target_sponsor, regex=False)]

    if df.empty:
        return f"No studies found matching the criteria (Query: {query}, Filters: Phase={phase}, Status={status}, Sponsor={sponsor})."

    # Map group_by to metadata key
    group_key = group_by
    if group_by == "sponsor": group_key = "org"
    
    # Special handling for comma-separated fields (like conditions)
    if group_key == "condition":
        # Split strings into lists, explode, strip whitespace
        all_values = df[group_key].dropna().astype(str).str.split(',')
        exploded = all_values.explode().str.strip()
        # Filter out empty or "Unknown"
        exploded = exploded[exploded != "Unknown"]
        exploded = exploded[exploded != ""]
        counts = exploded.value_counts().head(20)
    else:
        counts = df[group_key].value_counts().head(20)
    
    summary = f"Analysis of relevant studies for '{query}' grouped by '{group_by}'"
    if phase or status or sponsor:
        summary += f" (Filtered by: Phase='{phase}', Status='{status}', Sponsor='{sponsor}')"
    summary += ":\n\n"
    
    for name, count in counts.items():
        summary += f"- {name}: {count}\n"
        
    # --- INLINE CHART LOGIC ---
    # Store data for rendering in the chat interface
    try:
        chart_type = "line" if group_key == "start_year" else "bar"
        title = f"Top {group_by} for '{query}'"
        if phase: title += f" (Phase: {phase})"
        if status: title += f" (Status: {status})"
        if sponsor: title += f" (Sponsor: {sponsor})"
        
        st.session_state['inline_chart_data'] = {
            "data": counts,
            "type": chart_type,
            "title": title
        }
    except Exception as e:
        print(f"Error setting inline chart data: {e}")
        
    return summary

# 4. Define Agent
tools = [search_trials, find_similar_studies, get_study_analytics]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Clinical Trial Expert Assistant. "
            "Use `search_trials` to find specific studies based on the user's query. It supports multiple filters (e.g., 'Phase 2, Phase 3'). "
            "Use `find_similar_studies` when the user asks for similar trials. This tool returns similarity scores and summaries. "
            "Use `get_study_analytics` when the user asks for aggregations, counts, or lists of entities (e.g., 'Which sponsors...', 'Most common condition...', 'Study types'). "
            "Always provide the NCT ID, Title, Status, and a brief summary for each study found in search results. "
            "If asked about **Inclusion/Exclusion Criteria**, extract this information directly from the study text provided in the search results. "
            "When explaining similarity, use the provided summaries to highlight shared features (e.g., same drug class, similar patient population) and reference the similarity score."
            "If you use `get_study_analytics`, mention that a chart has been generated below."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Check if this message had an associated chart (stored in history? No, complex. Just render current)
        # For simplicity, we only render the chart immediately after generation. 
        # To persist charts in history, we'd need to store them in the message object, which is harder with Streamlit's chat model.
        # We will accept that charts are ephemeral or we need a custom message type.
        # Actually, we can check if 'chart_data' is in the message dict if we add it there.
        if "chart_data" in message:
            c_data = message["chart_data"]
            if c_data["type"] == "bar":
                st.bar_chart(c_data["data"])
            elif c_data["type"] == "line":
                # Re-render Altair chart for history
                c_df = c_data["data"].reset_index()
                c_df.columns = ['Year', 'Count']
                c_df['Year'] = pd.to_numeric(c_df['Year'], errors='coerce')
                c_df = c_df.dropna(subset=['Year'])
                
                chart = alt.Chart(c_df).mark_line(point=True).encode(
                    x=alt.X('Year', axis=alt.Axis(format='d', title='Start Year')),
                    y=alt.Y('Count', title='Number of Studies'),
                    tooltip=['Year', 'Count']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

if prompt := st.chat_input("Ask about clinical trials..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing clinical trials..."):
            try:
                # Clear previous inline chart data
                if 'inline_chart_data' in st.session_state:
                    del st.session_state['inline_chart_data']
                
                # Construct chat history
                chat_history = []
                for msg in st.session_state.messages[:-1]: # Exclude the latest user message which is passed as input
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))
                
                response = agent_executor.invoke({
                    "input": prompt,
                    "chat_history": chat_history
                })
                output = response["output"]
                st.markdown(output)
                
                # Check for inline chart data
                chart_data = None
                if 'inline_chart_data' in st.session_state:
                    chart_data = st.session_state['inline_chart_data']
                    st.caption(chart_data["title"])
                    if chart_data["type"] == "bar":
                        st.bar_chart(chart_data["data"])
                    elif chart_data["type"] == "line":
                        # Use Altair for Year charts to format X-axis without commas
                        # Convert Series to DataFrame
                        c_df = chart_data["data"].reset_index()
                        c_df.columns = ['Year', 'Count']
                        # Ensure Year is numeric
                        c_df['Year'] = pd.to_numeric(c_df['Year'], errors='coerce')
                        c_df = c_df.dropna(subset=['Year'])
                        
                        chart = alt.Chart(c_df).mark_line(point=True).encode(
                            x=alt.X('Year', axis=alt.Axis(format='d', title='Start Year')),
                            y=alt.Y('Count', title='Number of Studies'),
                            tooltip=['Year', 'Count']
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)
                    
                    # Clean up
                    del st.session_state['inline_chart_data']
                
                # Save message with chart data if present
                msg_obj = {"role": "assistant", "content": output}
                if chart_data:
                    msg_obj["chart_data"] = chart_data
                st.session_state.messages.append(msg_obj)
        
            except Exception as e:
                st.error(f"An error occurred: {e}")

# 6. Analytics & Export (On-Demand)
with st.sidebar:
    st.header("📊 Analytics & Export")
    st.caption(f"v{__version__}")
    
    # --- Analytics Scope ---
    st.write("Analyze trends in the clinical trial data.")
    
    # Simplified: Always Overall Dataset
    if st.button("Load Analytics (Overall Dataset)"):
        with st.spinner("Generating analytics for entire dataset..."):
            data = []
            st.info("Analyzing **Entire Dataset** (60,000+ studies)")
            
            try:
                # Fetch ALL metadata directly from ChromaDB
                db_client = chromadb.PersistentClient(path="./ct_gov_index")
                collection = db_client.get_collection("clinical_trials")
                # Get all metadata
                all_docs = collection.get(include=['metadatas'])
                metadatas = all_docs['metadatas']
                data = metadatas # List of dicts
            except Exception as e:
                st.error(f"Error fetching full dataset: {e}")

            if data:
                # Normalize keys if needed
                def get_val(d, k, default="Unknown"):
                    return d.get(k, default)
                
                processed_data = []
                for meta in data:
                    processed_data.append({
                        "NCT ID": get_val(meta, "nct_id", "Unknown"),
                        "Title": get_val(meta, "title", "Unknown"),
                        "Phase": get_val(meta, "phase", "NA"),
                        "Sponsor": get_val(meta, "org", "Unknown"),
                        "Year": get_val(meta, "start_year", 0),
                        "Status": get_val(meta, "status", "Unknown"),
                        "Study Type": get_val(meta, "study_type", "Unknown"),
                        "Condition": get_val(meta, "condition", "Unknown"),
                        "Country": get_val(meta, "country", "Unknown")
                    })
                    
                df = pd.DataFrame(processed_data)
                
                # Convert Year to string to remove commas in charts/tables
                df['Year'] = df['Year'].astype(str).str.replace(',', '')

                # --- TABS FOR ORGANIZED VIEW ---
                tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Conditions", "Sponsors", "Data"])
                
                with tab1:
                    st.subheader("Study Status Distribution")
                    status_counts = df['Status'].value_counts()
                    st.bar_chart(status_counts)
                    
                    st.subheader("Start Year Trend")
                    # Convert to numeric for Altair plotting
                    df['YearNum'] = pd.to_numeric(df['Year'], errors='coerce')
                    year_data = df['YearNum'].value_counts().sort_index().reset_index()
                    year_data.columns = ['Year', 'Count']
                    # Filter out 0 or NaN
                    year_data = year_data[year_data['Year'] > 0]
                    
                    chart = alt.Chart(year_data).mark_line(point=True).encode(
                        x=alt.X('Year', axis=alt.Axis(format='d', title='Start Year')),
                        y=alt.Y('Count', title='Number of Studies'),
                        tooltip=['Year', 'Count']
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)

                with tab2:
                    st.subheader("Top Conditions Studied")
                    # Split comma-separated conditions
                    all_conditions = df['Condition'].dropna().astype(str).str.split(',')
                    exploded_conditions = all_conditions.explode().str.strip()
                    # Filter out "Unknown" or empty
                    exploded_conditions = exploded_conditions[exploded_conditions != "Unknown"]
                    exploded_conditions = exploded_conditions[exploded_conditions != ""]
                    
                    top_conditions = exploded_conditions.value_counts().head(15)
                    st.bar_chart(top_conditions)

                with tab3:
                    st.subheader("Top Sponsors")
                    top_sponsors = df['Sponsor'].value_counts().head(10)
                    st.bar_chart(top_sponsors)
                    
                    st.subheader("Sponsor Strategy (Phase Breakdown)")
                    # Filter for top 10 sponsors to keep chart readable
                    top_sponsor_names = top_sponsors.index.tolist()
                    filtered_df = df[df['Sponsor'].isin(top_sponsor_names)]
                    
                    # Pivot: Rows=Sponsor, Cols=Phase, Values=Count
                    if not filtered_df.empty:
                        sponsor_phase = pd.crosstab(filtered_df['Sponsor'], filtered_df['Phase'])
                        st.bar_chart(sponsor_phase)
                    else:
                        st.write("Not enough data for sponsor breakdown.")

                with tab4:
                    st.subheader(f"Raw Data ({len(df)} studies)")
                    # Use st.dataframe with column configuration to enable better interaction
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Year": st.column_config.TextColumn("Start Year"),
                            "NCT ID": st.column_config.TextColumn("NCT ID"),
                        }
                    )
                    
                    # Export Button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV",
                        csv,
                        "clinical_trials_analytics.csv",
                        "text/csv",
                        key='download-csv'
                    )
            else:
                st.warning("No data found.")