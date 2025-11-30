"""
LangChain Tools for the Clinical Trial Agent.

This module defines the tools that the agent can use to interact with the clinical trial data.
Tools include:
1.  **search_trials**: Semantic search with optional strict filtering.
2.  **find_similar_studies**: Finding studies semantically similar to a given text.
3.  **get_study_analytics**: Aggregating data for trends and insights (with inline charts).
"""

import pandas as pd
import streamlit as st
from typing import Optional
from langchain.tools import tool as langchain_tool
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core import Settings
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from modules.utils import (
    load_index,
    normalize_sponsor,
    get_sponsor_variations,
    LocalMetadataPostFilter,
    get_hybrid_retriever,
)
import re
import traceback

# --- Tools ---


def expand_query(query: str) -> str:
    """Expands a search query with synonyms using the LLM."""
    if not query or len(query.split()) > 10:  # Skip expansion for long queries
        return query
    
    # Skip expansion if it looks like an NCT ID
    if re.search(r"NCT\d+", query, re.IGNORECASE):
        return query

    prompt = (
        f"You are a helpful medical assistant. "
        f"Expand the following search query with relevant medical synonyms and acronyms. "
        f"Return ONLY the expanded query string combined with OR operators. "
        f"Do not add any explanation.\n\n"
        f"Query: {query}\n"
        f"Expanded Query:"
    )
    try:
        # Use the global Settings.llm
        if not Settings.llm:
            # Fallback if not initialized (though load_index does it)
            from modules.utils import setup_llama_index

            setup_llama_index()

        response = Settings.llm.complete(prompt)
        expanded = response.text.strip()
        # Clean up if LLM is chatty
        if "Expanded Query:" in expanded:
            expanded = expanded.split("Expanded Query:")[-1].strip()
        print(f"✨ Expanded Query: '{query}' -> '{expanded}'")
        return expanded
    except Exception as e:
        print(f"⚠️ Query expansion failed: {e}")
        return query


@langchain_tool("search_trials")
def search_trials(
    query: str = None,
    status: str = None,
    phase: str = None,
    sponsor: str = None,
    intervention: str = None,
    year: int = None,
):
    """
    Searches for clinical trials using semantic search with robust filtering.

    Args:
        query (str, optional): The natural language search query.
        status (str, optional): Filter by recruitment status.
        phase (str, optional): Filter by trial phase.
        sponsor (str, optional): Filter by sponsor name.
        intervention (str, optional): Filter by intervention/drug name.
        year (int, optional): Filter for studies starting on or after this year.

    Returns:
        str: A structured list of relevant studies.
    """
    index = load_index()
    
    # Constants
    TOP_K_STRICT = 500  # High recall for pre-filtered search
    TOP_K_HYBRID = 500  # Higher recall for hybrid search (fuzzy matching)
    
    # --- Query Construction ---
    if not query:
        parts = [p for p in [sponsor, intervention, phase, status] if p]
        query = " ".join(parts) if parts else "clinical trial"
    else:
        # Inject context for vector search
        if sponsor and normalize_sponsor(sponsor).lower() not in query.lower():
            query = f"{normalize_sponsor(sponsor)} {query}"
        if intervention and intervention.lower() not in query.lower():
            query = f"{intervention} {query}"
        
        query = expand_query(query)

    print(f"🔍 Tool Called: search_trials(query='{query}', sponsor='{sponsor}')")

    # --- Strategy 1: Strict Pre-Retrieval Filtering (High Precision) ---
    # Try to filter by Sponsor/Status/Year at the database level first.
    pre_filters = []
    
    # NCT ID Match
    nct_match = re.search(r"NCT\d+", query, re.IGNORECASE)
    if nct_match:
        nct_id = nct_match.group(0).upper()
        pre_filters.append(MetadataFilter(key="nct_id", value=nct_id, operator=FilterOperator.EQ))

    if status:
        pre_filters.append(MetadataFilter(key="status", value=status.upper(), operator=FilterOperator.EQ))
    if year:
        pre_filters.append(MetadataFilter(key="start_year", value=year, operator=FilterOperator.GTE))
        
    # Sponsor Pre-Filter (The Fix!)
    if sponsor:
        from modules.utils import get_sponsor_variations
        variations = get_sponsor_variations(sponsor)
        if variations:
            print(f"🎯 Applying strict pre-filter for sponsor '{sponsor}' ({len(variations)} variants)")
            pre_filters.append(MetadataFilter(key="org", value=variations, operator=FilterOperator.IN))
        else:
            print(f"⚠️ No strict mapping for sponsor '{sponsor}'. Will rely on fuzzy post-filtering.")

    metadata_filters = MetadataFilters(filters=pre_filters) if pre_filters else None
    
    # Post-processors (Reranking)
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-12-v2", top_n=50)
    
    # --- HYBRID SEARCH IMPLEMENTATION ---
    # We use the new get_hybrid_retriever which combines Vector + BM25
    try:
        retriever = get_hybrid_retriever(index, similarity_top_k=TOP_K_STRICT, filters=metadata_filters)
        nodes = retriever.retrieve(query)
        
        # Apply Reranker manually since we are using a retriever, not a query engine
        # (QueryFusionRetriever returns nodes, but we want to rerank them)
        if nodes:
            from llama_index.core.schema import QueryBundle
            nodes = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=query))

    except Exception as e:
        print(f"⚠️ Hybrid search failed: {e}. Falling back to standard vector search.")
        traceback.print_exc()
        query_engine = index.as_query_engine(
            similarity_top_k=TOP_K_STRICT,
            filters=metadata_filters,
            node_postprocessors=[reranker]
        )
        response = query_engine.query(query)
        nodes = response.source_nodes

    # --- Strict Metadata Filtering (Post-Fusion) ---
    # BM25 results might not respect the vector filters, so we must filter them out.
    final_nodes = []
    for node in nodes:
        meta = node.metadata
        keep = True
        
        # Re-apply filters to ensure BM25 results are valid
        if status and meta.get("status", "").upper() != status.upper():
            keep = False
        if year:
            try:
                if int(meta.get("start_year", 0)) < year:
                    keep = False
            except:
                pass
        if sponsor:
            # We already have strict logic for sponsor in pre-filters, but BM25 ignores it.
            # So we check if the sponsor matches one of the variations OR fuzzy match
            # If we had strict variations, we enforce them.
            variations = get_sponsor_variations(sponsor)
            node_org = meta.get("org", "")
            if variations:
                if node_org not in variations:
                    keep = False
            else:
                # Fuzzy fallback
                if normalize_sponsor(sponsor).lower() not in normalize_sponsor(node_org).lower():
                    keep = False
        
        if keep:
            final_nodes.append(node)
    
    nodes = final_nodes

    # --- Strict Keyword Filtering (Refined) ---
    # With BM25, we don't need the aggressive "substring check" as much, 
    # but for "Analytics" consistency, we might still want to prioritize exact title matches.
    # However, BM25 should handle this naturally. 
    # We will keep the logic but make it less destructive (just boosting or trusted).
    # Actually, the user wants "consistent counts". 
    # If the user searches "Multiple Myeloma", they want studies about that, not just mentioning it.
    # BM25 is good at this.
    # We will trust Hybrid Search + Reranker.
    
    # Update response object structure to match expected format if we used retriever
    class MockResponse:
        def __init__(self, nodes):
            self.source_nodes = nodes
    
    response = MockResponse(nodes)
    
    # --- Strategy 2: Hybrid Search (Fallback) ---
    # The original code had a fallback. With Hybrid Search enabled by default, 
    # we might not need a separate fallback unless the strict filters were too aggressive.
    # But we already handle strict filters in the post-processing above.
    # So we can simplify.


    # --- Formatting Output ---
    if not response.source_nodes:
        return "No matching studies found. Try broadening your search terms or filters."

    # Filter by Relevance Score for display
    MIN_SCORE = 1.5
    relevant_nodes = [node for node in response.source_nodes if node.score > MIN_SCORE]
    
    # If strict filtering removes too much, show at least top 3 to be helpful
    if len(relevant_nodes) < 3 and len(response.source_nodes) > 0:
        relevant_nodes = response.source_nodes[:3]
        
    display_limit = 20
    display_nodes = relevant_nodes[:display_limit]
    
    results = []
    for node in display_nodes:
        meta = node.metadata
        entry = (
            f"**{meta.get('title', 'Untitled')}**\n"
            f"   - ID: {meta.get('nct_id')}\n"
            f"   - Phase: {meta.get('phase', 'N/A')}\n"
            f"   - Status: {meta.get('status', 'N/A')}\n"
            f"   - Sponsor: {meta.get('org', 'Unknown')}\n"
            f"   - Relevance: {node.score:.2f}"
        )
        results.append(entry)

    return f"Found {len(results)} relevant studies:\n\n" + "\n\n".join(results)


@langchain_tool("find_similar_studies")
def find_similar_studies(query: str):
    """
    Finds studies semantically similar to a given query or study description.

    This tool is useful for "more like this" functionality. It relies purely
    on vector similarity without strict metadata filtering.

    Args:
        query (str): The text to match against (e.g., a study title or description).

    Returns:
        str: A string containing the top 5 similar studies with their titles and summaries.
    """
    index = load_index()
    
    # 1. Check if query is an NCT ID
    nct_match = re.search(r"NCT\d+", query, re.IGNORECASE)
    target_nct = None
    search_text = query

    if nct_match:
        target_nct = nct_match.group(0).upper()
        print(f"🎯 Detected NCT ID for similarity: {target_nct}")
        
        # Fetch the study content to use as the semantic query
        # We use the vector store directly to get the text
        retriever = index.as_retriever(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="nct_id", value=target_nct, operator=FilterOperator.EQ)]
            ),
            similarity_top_k=1
        )
        nodes = retriever.retrieve(target_nct)
        
        if nodes:
            # Use the study's text (Title + Summary) as the query
            search_text = nodes[0].text
            print(f"✅ Found study content. Using {len(search_text)} chars for semantic search.")
        else:
            print(f"⚠️ Study {target_nct} not found. Falling back to text search.")

    # 2. Perform Semantic Search
    # Fetch more candidates (10) to allow for filtering
    retriever = index.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(search_text)

    results = []
    count = 0
    for node in nodes:
        # 3. Self-Exclusion
        if target_nct and node.metadata.get("nct_id") == target_nct:
            continue
            
        # Deduplication (if multiple chunks of same study appear)
        if any(r["nct_id"] == node.metadata.get("nct_id") for r in results):
            continue

        results.append({
            "nct_id": node.metadata.get("nct_id"),
            "text": f"Study: {node.metadata['title']} (NCT: {node.metadata.get('nct_id')})\nScore: {node.score:.4f}\nSummary: {node.text[:200]}..."
        })
        
        count += 1
        if count >= 5:  # Limit to top 5 unique results
            break

    if not results:
        return "No similar studies found."

    return "\n\n".join([r["text"] for r in results])


def fetch_study_analytics_data(
    query: str,
    group_by: str,
    phase: Optional[str] = None,
    status: Optional[str] = None,
    sponsor: Optional[str] = None,
    intervention: Optional[str] = None,
    start_year: Optional[int] = None,
    study_type: Optional[str] = None,
) -> str:
    """
    Underlying logic for fetching and aggregating clinical trial data.
    See get_study_analytics for full docstring.
    """
    index = load_index()

    # 1. Retrieve Data
    if query.lower() == "overall":
        try:
            collection = index.vector_store._collection
            result = collection.get(include=["metadatas"])
            data = result["metadatas"]
        except Exception as e:
            return f"Error fetching full dataset: {e}"
    else:
        filters = []
        if status:
            filters.append(
                MetadataFilter(
                    key="status", value=status.upper(), operator=FilterOperator.EQ
                )
            )
        if phase and "," not in phase:
            pass

        if sponsor:
            # Use the helper to get all variations (e.g. "Pfizer" -> ["Pfizer", "Pfizer Inc."])
            sponsor_variations = get_sponsor_variations(sponsor)
            if sponsor_variations:
                print(f"🎯 Using strict pre-filter for sponsor '{sponsor}': {len(sponsor_variations)} variations found.")
                filters.append(
                    MetadataFilter(
                        key="org", value=sponsor_variations, operator=FilterOperator.IN
                    )
                )

        metadata_filters = MetadataFilters(filters=filters) if filters else None

        search_query = query
        if sponsor and sponsor.lower() not in query.lower():
            search_query = f"{sponsor} {query}"

        retriever = index.as_retriever(similarity_top_k=5000, filters=metadata_filters)
        nodes = retriever.retrieve(search_query)
        
        # --- Strict Keyword Filtering (Deterministic) ---
        # We strictly check if the query appears in Title or Conditions.
        # This is crucial for accurate counting (e.g. "Multiple Myeloma" -> 7 studies vs 514).
        if query.lower() != "overall":
            q_term = query.lower()
            filtered_nodes = []
            for node in nodes:
                meta = node.metadata
                title = meta.get("title", "").lower()
                conditions = meta.get("condition", "").lower() # Note: key is 'condition' in DB, not 'conditions'
                
                if q_term in title or q_term in conditions:
                    filtered_nodes.append(node)
            
            print(f"📉 Strict Filter: {len(nodes)} -> {len(filtered_nodes)} nodes for '{query}'")
            nodes = filtered_nodes
        
        data = [node.metadata for node in nodes]

    df = pd.DataFrame(data)
    
    if "nct_id" in df.columns:
        df = df.drop_duplicates(subset="nct_id")

    if df.empty:
        return "No studies found for analytics."

    # --- APPLY FILTERS (Pandas) ---
    if phase:
        target_phases = [p.strip().upper().replace(" ", "") for p in phase.split(",")]
        df["phase_upper"] = df["phase"].astype(str).str.upper().str.replace(" ", "")
        mask = df["phase_upper"].apply(lambda x: any(tp in x for tp in target_phases))
        df = df[mask]

    if status:
        df = df[df["status"].str.upper() == status.upper()]

    if sponsor:
        target_sponsor = normalize_sponsor(sponsor).lower()
        df["org_lower"] = df["org"].astype(str).apply(normalize_sponsor).str.lower()
        df = df[df["org_lower"].str.contains(target_sponsor, regex=False)]

    if intervention:
        target_intervention = intervention.lower()
        df["intervention_lower"] = df["intervention"].astype(str).str.lower()
        df = df[df["intervention_lower"].str.contains(target_intervention, regex=False)]

    if start_year:
        df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce").fillna(0)
        df = df[df["start_year"] >= start_year]

    if study_type:
        df = df[df["study_type"].str.upper() == study_type.upper()]

    if df.empty:
        return "No studies found after applying filters."

    key_map = {
        "phase": "phase",
        "status": "status",
        "sponsor": "org",
        "start_year": "start_year",
        "condition": "condition",
        "intervention": "intervention",
        "study_type": "study_type",
        "country": "country",
        "state": "state",
    }

    if group_by not in key_map:
        return f"Invalid group_by field: {group_by}. Valid options: phase, status, sponsor, start_year, condition, intervention, study_type, country, state"

    col = key_map[group_by]

    if col == "start_year":
        df[col] = pd.to_numeric(df[col], errors="coerce")
        counts = df[col].value_counts().sort_index()
    elif col == "condition":
        counts = df[col].astype(str).str.split(", ").explode().value_counts().head(10)
    elif col == "intervention":
        all_interventions = []
        for interventions in df[col].dropna():
            parts = [i.strip() for i in interventions.split(";") if i.strip()]
            all_interventions.extend(parts)
        counts = pd.Series(all_interventions).value_counts().head(10)
    else:
        counts = df[col].value_counts().head(10)

    summary = counts.to_string()

    chart_df = counts.reset_index()
    chart_df.columns = ["category", "count"]
    
    chart_data = {
        "type": "bar",
        "title": f"Studies by {group_by.capitalize()}",
        "data": chart_df.to_dict("records"),
        "x": "category",
        "y": "count",
    }

    if "inline_chart_data" not in st.session_state:
        st.session_state["inline_chart_data"] = chart_data
    else:
        st.session_state["inline_chart_data"] = chart_data

    return f"Found {len(df)} studies. Top counts:\n{summary}\n\n(Chart generated in UI)"


@langchain_tool("get_study_analytics")
def get_study_analytics(
    query: str,
    group_by: str,
    phase: Optional[str] = None,
    status: Optional[str] = None,
    sponsor: Optional[str] = None,
    intervention: Optional[str] = None,
    start_year: Optional[int] = None,
    study_type: Optional[str] = None,
):
    """
    Aggregates clinical trial data based on a search query and groups by a specific field.
    
    This tool performs the following steps:
    1.  Retrieves a large number of relevant studies (up to 500).
    2.  Applies strict filters (Phase, Status, Sponsor) in memory (Pandas).
    3.  Groups the data by the requested field (e.g., Sponsor).
    4.  Generates a summary string for the LLM.
    5.  **Side Effect**: Injects chart data into `st.session_state` to trigger an inline chart in the UI.

    Args:
        query (str): The search query to filter studies (e.g., "cancer").
        group_by (str): The field to group by. Options: "phase", "status", "sponsor", "start_year", "condition".
        phase (Optional[str]): Optional filter for phase (e.g., "PHASE2").
        status (Optional[str]): Optional filter for status (e.g., "RECRUITING").
        sponsor (Optional[str]): Optional filter for sponsor (e.g., "Pfizer").
        intervention (Optional[str]): Optional filter for intervention (e.g., "Keytruda").

    Returns:
        str: A summary string of the top counts and a note that a chart has been generated.
    """
    return fetch_study_analytics_data(
        query=query,
        group_by=group_by,
        phase=phase,
        status=status,
        sponsor=sponsor,
        intervention=intervention,
        start_year=start_year,
        study_type=study_type,
    )


@langchain_tool("compare_studies")
def compare_studies(query: str):
    """
    Compares multiple studies or answers complex multi-part questions using query decomposition.

    Use this tool when the user asks to "compare", "contrast", or analyze differences/similarities
    between specific studies, sponsors, or phases. It breaks down the question into sub-questions.

    Args:
        query (str): The complex comparison query (e.g., "Compare the primary outcomes of Keytruda vs Opdivo").

    Returns:
        str: A detailed response synthesizing the answers to sub-questions.
    """
    index = load_index()

    # Create a base query engine for the sub-questions
    # We increase top_k and add re-ranking to improve recall for comparison queries
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-12-v2", top_n=10)
    
    base_engine = index.as_query_engine(
        similarity_top_k=50,
        node_postprocessors=[reranker]
    )

    # Wrap it in a QueryEngineTool
    query_tool = QueryEngineTool(
        query_engine=base_engine,
        metadata=ToolMetadata(
            name="clinical_trials_db",
            description="Vector database of clinical trial protocols, results, and metadata.",
        ),
    )

    # Create the SubQuestionQueryEngine
    # We explicitly define the question generator to use our configured LLM (Gemini)
    # This avoids the default behavior which might try to import OpenAI modules
    from llama_index.core.question_gen import LLMQuestionGenerator
    from llama_index.core import Settings

    question_gen = LLMQuestionGenerator.from_defaults(llm=Settings.llm)

    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[query_tool],
        question_gen=question_gen,
        use_async=True,
    )

    try:
        response = query_engine.query(query)
        return str(response) + "\n\n(Note: This analysis is based on the most relevant studies retrieved from the database, not necessarily an exhaustive list.)"
    except Exception as e:
        return f"Error during comparison: {e}"


@langchain_tool("get_study_details")
def get_study_details(nct_id: str):
    """
    Retrieves the full details of a specific clinical trial by its NCT ID.

    Use this tool when the user asks for specific information about a single study,
    such as "What are the inclusion criteria for NCT12345678?" or "Give me a summary of study NCT...".
    It returns the full text content of the study document, including criteria, outcomes, and contacts.

    Args:
        nct_id (str): The NCT ID of the study (e.g., "NCT01234567").

    Returns:
        str: The full text content of the study, or a message if not found.
    """
    index = load_index()

    # Clean the ID
    clean_id = nct_id.strip().upper()

    # Use a retriever with a strict metadata filter for the ID
    # We set top_k=20 to capture all chunks if the document was split
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="nct_id", value=clean_id, operator=FilterOperator.EQ)
        ]
    )

    retriever = index.as_retriever(similarity_top_k=20, filters=filters)
    nodes = retriever.retrieve(clean_id)

    if not nodes:
        return f"Study {clean_id} not found in the database."

    # Sort nodes by their position in the document to reconstruct full text
    # LlamaIndex nodes usually have 'start_char_idx' in metadata or relationships
    # We'll try to sort by node ID or just concatenate them

    # Simple concatenation (assuming retrieval order is roughly correct or sufficient)
    full_text = "\n\n".join([node.text for node in nodes])

    return f"Details for {clean_id} (Combined {len(nodes)} parts):\n\n{full_text}"
