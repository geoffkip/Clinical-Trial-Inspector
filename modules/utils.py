"""
Utility functions for the Clinical Trial Agent.

This module handles:
1.  **Configuration**: Setting up LlamaIndex settings (LLM, Embeddings).
2.  **Index Loading**: Loading the persisted ChromaDB vector index.
3.  **Normalization**: Helper functions for standardizing data (e.g., sponsor names).
4.  **Filtering**: Custom post-processors for filtering retrieval results.
"""

import os
import streamlit as st
from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.lancedb import LanceDBVectorStore
import lancedb
from dotenv import load_dotenv




def load_environment():
    """Loads environment variables from .env file."""
    load_dotenv()


# --- Configuration ---
def setup_llama_index():
    """
    Configures the global LlamaIndex settings.

    Sets up:
    - **LLM**: Google Gemini (gemini-2.5-flash) with temperature=0 for deterministic outputs.
    - **Embeddings**: PubMedBERT (pritamdeka/S-PubMedBert-MS-MARCO) for biomedical domain specificity.

    Raises:
        SystemExit: If GOOGLE_API_KEY is not found in environment variables.
    """
    if "GOOGLE_API_KEY" not in os.environ:
        st.error("Please set GOOGLE_API_KEY in .env")
        st.stop()

    Settings.llm = Gemini(model="models/gemini-2.5-flash", temperature=0)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="pritamdeka/S-PubMedBert-MS-MARCO"
    )


@st.cache_resource
def load_index() -> VectorStoreIndex:
    """
    Loads the persistent LanceDB index.

    Uses Streamlit's @st.cache_resource to load the index only once per session.

    Returns:
        VectorStoreIndex: The loaded LlamaIndex vector store index.
    """
    setup_llama_index()
    
    # Initialize LanceDB
    db_path = "./ct_gov_lancedb"
    db = lancedb.connect(db_path)

    # Create the vector store wrapper
    # mode="read" ensures we don't accidentally overwrite or create new tables here
    vector_store = LanceDBVectorStore(
        uri=db_path, 
        table_name="clinical_trials",
        query_mode="hybrid"
    )

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    return index


def get_hybrid_retriever(index: VectorStoreIndex, similarity_top_k: int = 50, filters=None):
    """
    Creates a Hybrid Retriever using LanceDB's native hybrid search.
    
    Args:
        index (VectorStoreIndex): The loaded vector index.
        similarity_top_k (int): Number of top results to retrieve.
        filters (MetadataFilters, optional): Filters to apply.
        
    Returns:
        VectorIndexRetriever: The configured retriever.
    """
    # LanceDB supports native hybrid search via query_mode="hybrid"
    # We pass this configuration to the retriever
    return index.as_retriever(
        similarity_top_k=similarity_top_k, 
        filters=filters,
        vector_store_query_mode="hybrid"
    )


# --- Normalization ---
def normalize_sponsor(sponsor: str) -> Optional[str]:
    """
    Normalizes sponsor names to handle common aliases and variations.

    This is crucial for accurate filtering and aggregation, as sponsor names
    in the raw data can vary (e.g., "Merck", "MSD", "Merck Sharp & Dohme").

    Args:
        sponsor (str): The raw sponsor name.

    Returns:
        Optional[str]: The normalized canonical sponsor name, or None if input is empty.
    """
    if not sponsor:
        return None

    s = sponsor.lower().strip()
    # Mapping of common aliases to canonical names
    aliases = {
        "gsk": "GlaxoSmithKline",
        "glaxo": "GlaxoSmithKline",
        "glaxosmithkline": "GlaxoSmithKline",
        "j&j": "Janssen",
        "johnson & johnson": "Janssen",
        "johnson and johnson": "Janssen",
        "janssen": "Janssen",
        "bms": "Bristol-Myers Squibb",
        "bristol myers squibb": "Bristol-Myers Squibb",
        "merck": "Merck Sharp & Dohme",
        "msd": "Merck Sharp & Dohme",
    }

    for alias, canonical in aliases.items():
        if alias in s:
            return canonical
    return sponsor


def get_sponsor_variations(sponsor: str) -> Optional[List[str]]:
    """
    Returns a list of exact database 'org' values for a given sponsor alias.
    This enables strict pre-filtering using the IN operator.
    """
    if not sponsor:
        return None

    s = sponsor.lower().strip()

    # Hardcoded mapping based on DB analysis
    # This can be expanded or moved to a config file/DB later
    mappings = {
        "pfizer": ["Pfizer"],
        "janssen": [
            "Janssen Research & Development, LLC",
            "Janssen Vaccines & Prevention B.V.",
            "Janssen Pharmaceutical K.K.",
            "Janssen-Cilag International NV",
            "Janssen Sciences Ireland UC",
            "Janssen Pharmaceutica N.V., Belgium",
            "Janssen Scientific Affairs, LLC",
            "Janssen-Cilag Ltd.",
            "Xian-Janssen Pharmaceutical Ltd.",
            "Janssen Korea, Ltd., Korea",
            "Janssen-Cilag G.m.b.H",
            "Janssen-Cilag, S.A.",
            "Janssen BioPharma, Inc.",
        ],
        "j&j": [
            "Janssen Research & Development, LLC",
            "Janssen Vaccines & Prevention B.V.",
            "Janssen Pharmaceutical K.K.",
            "Janssen-Cilag International NV",
            "Janssen Sciences Ireland UC",
            "Janssen Pharmaceutica N.V., Belgium",
            "Janssen Scientific Affairs, LLC",
            "Janssen-Cilag Ltd.",
            "Xian-Janssen Pharmaceutical Ltd.",
            "Janssen Korea, Ltd., Korea",
            "Janssen-Cilag G.m.b.H",
            "Janssen-Cilag, S.A.",
            "Janssen BioPharma, Inc.",
        ],
        "merck": ["Merck Sharp & Dohme LLC"],  # Based on analyze_db output
        "msd": ["Merck Sharp & Dohme LLC"],
        "astrazeneca": ["AstraZeneca"],
        "lilly": ["Eli Lilly and Company"],
        "eli lilly": ["Eli Lilly and Company"],
        "bms": ["Bristol-Myers Squibb"],
        "bristol": ["Bristol-Myers Squibb"],
        "bristol myers squibb": ["Bristol-Myers Squibb"],
        "sanofi": ["Sanofi"],
        "novartis": ["Novartis"],
        "gsk": ["GlaxoSmithKline"],
        "glaxo": ["GlaxoSmithKline"],
    }

    for key, variations in mappings.items():
        if key in s:
            return variations

    return None





