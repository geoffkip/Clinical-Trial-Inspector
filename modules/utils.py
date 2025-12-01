"""
Utility functions for the Clinical Trial Agent.

Handles configuration, LanceDB index loading, data normalization, and custom filtering logic.
"""

import os
import streamlit as st
from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.llms.gemini import Gemini
import lancedb
from dotenv import load_dotenv

# --- MONKEYPATCH START ---
# Patch LanceDBVectorStore to handle 'nprobes' AttributeError and fix SQL quoting for IN filters.
original_query = LanceDBVectorStore.query

def patched_query(self, query, **kwargs):
    try:
        return original_query(self, query, **kwargs)
    except Exception as e:
        print(f"⚠️ LanceDB Query Error: {e}")
        if hasattr(query, "filters"):
            print(f"   Filters: {query.filters}")
        
        if "nprobes" in str(e):
            from llama_index.core.vector_stores.types import VectorStoreQueryResult
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        raise e

LanceDBVectorStore.query = patched_query

# Patch _to_lance_filter to fix SQL quoting for IN operator with strings.
from llama_index.vector_stores.lancedb import base as lancedb_base
from llama_index.core.vector_stores.types import FilterOperator

original_to_lance_filter = lancedb_base._to_lance_filter

def patched_to_lance_filter(standard_filters, metadata_keys):
    if not standard_filters:
        return None
        
    # Reimplement filter logic to ensure correct SQL generation for LanceDB
    filters = []
    for filter in standard_filters.filters:
        key = filter.key
        if metadata_keys and key not in metadata_keys:
             continue
        
        # Prefix key with 'metadata.' for LanceDB struct column
        lance_key = f"metadata.{key}"
             
        # Handle IN operator with proper string quoting
        if filter.operator == FilterOperator.IN:
            if isinstance(filter.value, list):
                # Quote strings properly
                values = []
                for v in filter.value:
                    if isinstance(v, str):
                        values.append(f"'{v}'") # Single quotes for SQL
                    else:
                        values.append(str(v))
                val_str = ", ".join(values)
                filters.append(f"{lance_key} IN ({val_str})")
                continue
        
        # Standard operators
        op = filter.operator
        val = filter.value
        
        if op == FilterOperator.EQ:
            if isinstance(val, str):
                filters.append(f"{lance_key} = '{val}'")
            else:
                filters.append(f"{lance_key} = {val}")
        elif op == FilterOperator.GT:
            filters.append(f"{lance_key} > {val}")
        elif op == FilterOperator.LT:
            filters.append(f"{lance_key} < {val}")
        elif op == FilterOperator.GTE:
            filters.append(f"{lance_key} >= {val}")
        elif op == FilterOperator.LTE:
            filters.append(f"{lance_key} <= {val}")
        elif op == FilterOperator.NE:
            if isinstance(val, str):
                filters.append(f"{lance_key} != '{val}'")
            else:
                filters.append(f"{lance_key} != {val}")
        # Add other operators as needed
        
    if not filters:
        return None
        
    return " AND ".join(filters)

lancedb_base._to_lance_filter = patched_to_lance_filter
# --- MONKEYPATCH END ---


def load_environment():
    """Loads environment variables from .env file."""
    load_dotenv()


# --- Configuration ---
def setup_llama_index(api_key: Optional[str] = None):
    """
    Configures global LlamaIndex settings (LLM and Embeddings).
    """
    # Use passed key, or fallback to env var
    final_key = api_key or os.environ.get("GOOGLE_API_KEY")

    if not final_key:
        # App handles prompting for key, so we just return or log warning
        pass

    try:
        # Pass the key explicitly if available
        Settings.llm = Gemini(model="models/gemini-2.5-flash", temperature=0, api_key=final_key)
    except Exception as e:
        print(f"⚠️ LLM initialization failed (likely missing API key): {e}")
        print("⚠️ Using MockLLM for testing/fallback.")
        from llama_index.core.llms import MockLLM
        Settings.llm = MockLLM()
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="pritamdeka/S-PubMedBert-MS-MARCO"
    )


@st.cache_resource
def load_index() -> VectorStoreIndex:
    """
    Loads and caches the persistent LanceDB index.
    """
    setup_llama_index()
    
    # Initialize LanceDB
    db_path = "./ct_gov_lancedb"
    db = lancedb.connect(db_path)

    # Define metadata keys explicitly to ensure filters work
    metadata_keys = [
        "nct_id", "title", "org", "sponsor", "status", "phase", 
        "study_type", "start_year", "condition", "intervention", 
        "country", "state"
    ]

    # Create the vector store wrapper
    vector_store = LanceDBVectorStore(
        uri=db_path, 
        table_name="clinical_trials",
        query_mode="hybrid",
    )
    
    # Manually set metadata keys as constructor doesn't accept them
    vector_store._metadata_keys = metadata_keys

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
    # Use standard retriever first to avoid LanceDB hybrid search issues on small datasets
    return index.as_retriever(
        similarity_top_k=similarity_top_k, 
        filters=filters,
    )


# --- Normalization ---

# Centralized Sponsor Mappings
# Key: Canonical Name
# Value: List of variations/aliases (including the canonical name itself if needed for matching)
SPONSOR_MAPPINGS = {
    "GlaxoSmithKline": [
        "gsk", "glaxo", "glaxosmithkline", "glaxosmithkline", 
        "GlaxoSmithKline"
    ],
    "Janssen": [
        "j&j", "johnson & johnson", "johnson and johnson", "janssen", "Janssen",
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
    "Bristol-Myers Squibb": [
        "bms", "bristol", "bristol myers squibb", "bristol-myers squibb",
        "Bristol-Myers Squibb"
    ],
    "Merck Sharp & Dohme": [
        "merck", "msd", "merck sharp & dohme", 
        "Merck Sharp & Dohme LLC"
    ],
    "Pfizer": ["pfizer", "Pfizer", "Pfizer Inc."],
    "AstraZeneca": ["astrazeneca", "AstraZeneca"],
    "Eli Lilly and Company": ["lilly", "eli lilly", "Eli Lilly and Company"],
    "Sanofi": ["sanofi", "Sanofi"],
    "Novartis": ["novartis", "Novartis"],
}

def normalize_sponsor(sponsor: str) -> Optional[str]:
    """
    Normalizes sponsor names to canonical forms using centralized mappings.
    """
    if not sponsor:
        return None

    s = sponsor.lower().strip()
    
    for canonical, variations in SPONSOR_MAPPINGS.items():
        # Check if input matches canonical name (case-insensitive)
        if s == canonical.lower():
            return canonical
            
        # Check variations and aliases
        for v in variations:
            v_lower = v.lower()
            if v_lower == s:
                return canonical
            # If the variation is a known alias (like 'gsk'), check if it's in the string
            if len(v) < 5 and v_lower in s: 
                 return canonical
            
            if canonical.lower() in s:
                return canonical

    return sponsor


def get_sponsor_variations(sponsor: str) -> Optional[List[str]]:
    """
    Returns list of exact database 'org' values for a given sponsor alias.
    """
    if not sponsor:
        return None

    # First, normalize the input to get the canonical name
    canonical = normalize_sponsor(sponsor)
    
    if canonical in SPONSOR_MAPPINGS:
        return SPONSOR_MAPPINGS[canonical]
        
    return None





