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
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
import chromadb
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
    Loads the persistent ChromaDB index.

    Uses Streamlit's @st.cache_resource to load the index only once per session.

    Returns:
        VectorStoreIndex: The loaded LlamaIndex vector store index.
    """
    setup_llama_index()
    # Initialize ChromaDB client pointing to the local persistence directory
    db = chromadb.PersistentClient(path="./ct_gov_index")

    # Get or create the collection for clinical trials
    chroma_collection = db.get_or_create_collection("clinical_trials")

    # Create the vector store wrapper
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    return index


def get_hybrid_retriever(index: VectorStoreIndex, similarity_top_k: int = 50, filters=None):
    """
    Creates a Hybrid Retriever (Vector + BM25) using Reciprocal Rank Fusion.
    
    Args:
        index (VectorStoreIndex): The loaded vector index.
        similarity_top_k (int): Number of top results to retrieve from EACH retriever.
        filters (MetadataFilters, optional): Filters to apply to the vector retriever.
        
    Returns:
        QueryFusionRetriever: The combined retriever.
    """
    # 1. Vector Retriever
    vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k, filters=filters)

    # 2. BM25 Retriever
    # We need to ensure BM25 has access to the nodes.
    # Since we are loading from a VectorStore, the docstore might be empty in memory.
    # We'll try to retrieve nodes from the docstore, or fallback to rebuilding from the vector store if needed.
    # For now, we assume the index (if loaded correctly) provides access to the docstore or we can pass the docstore.
    # NOTE: If docstore is empty, we might need to fetch all nodes from Chroma.
    # Let's check if we can get nodes.
    
    # Strategy: Use the docstore attached to the index.
    # If this fails in practice (empty results), we might need to explicitly load nodes.
    # But typically StorageContext should handle it if persisted.
    # However, ChromaVectorStore usually doesn't persist the docstore in the same way simple index does.
    # So we might need to fetch from vector store.
    
    try:
        # Try to get all nodes from the docstore
        nodes = list(index.docstore.docs.values())
        if not nodes:
            # Fallback: Fetch from Chroma directly to build BM25
            print("⚠️ Docstore empty. Fetching nodes from Chroma for BM25...")
            try:
                # Access the underlying Chroma collection
                # We assume index.vector_store is ChromaVectorStore
                if hasattr(index.vector_store, "_collection"):
                    result = index.vector_store._collection.get()
                    # result is a dict with 'ids', 'documents', 'metadatas'
                    ids = result["ids"]
                    documents = result["documents"]
                    metadatas = result["metadatas"]
                    
                    nodes = []
                    for i, doc_id in enumerate(ids):
                        text = documents[i]
                        meta = metadatas[i] if metadatas else {}
                        node = TextNode(text=text, id_=doc_id, metadata=meta)
                        nodes.append(node)
                    
                    print(f"✅ Reconstructed {len(nodes)} nodes from Chroma for BM25.")
            except Exception as e:
                print(f"❌ Failed to fetch from Chroma: {e}")
            
        if nodes:
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=similarity_top_k
            )
        else:
            # If we can't build BM25, return just vector retriever
            print("⚠️ Could not build BM25 index (no nodes found). Returning Vector Retriever only.")
            return vector_retriever

    except Exception as e:
        print(f"⚠️ Error building BM25 retriever: {e}. Returning Vector Retriever only.")
        return vector_retriever

    # 3. Fusion
    return QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=similarity_top_k,
        num_queries=1,  # No query generation, just use the original query
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
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





