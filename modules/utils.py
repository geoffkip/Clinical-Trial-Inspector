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


# --- Constants ---
COUNTRY_COORDINATES = {
    "United States": [37.0902, -95.7129],
    "Canada": [56.1304, -106.3468],
    "United Kingdom": [55.3781, -3.4360],
    "Germany": [51.1657, 10.4515],
    "France": [46.2276, 2.2137],
    "China": [35.8617, 104.1954],
    "Japan": [36.2048, 138.2529],
    "Australia": [-25.2744, 133.7751],
    "Brazil": [-14.2350, -51.9253],
    "India": [20.5937, 78.9629],
    "Russia": [61.5240, 105.3188],
    "South Korea": [35.9078, 127.7669],
    "Italy": [41.8719, 12.5674],
    "Spain": [40.4637, -3.7492],
    "Netherlands": [52.1326, 5.2913],
    "Belgium": [50.5039, 4.4699],
    "Switzerland": [46.8182, 8.2275],
    "Sweden": [60.1282, 18.6435],
    "Israel": [31.0461, 34.8516],
    "Poland": [51.9194, 19.1451],
    "Taiwan": [23.6978, 120.9605],
    "Mexico": [23.6345, -102.5528],
    "Argentina": [-38.4161, -63.6167],
    "South Africa": [-30.5595, 22.9375],
    "Turkey": [38.9637, 35.2433],
    "Denmark": [56.2639, 9.5018],
    "New Zealand": [-40.9006, 174.8860],
    "Czech Republic": [49.8175, 15.4730],
    "Hungary": [47.1625, 19.5033],
    "Finland": [61.9241, 25.7482],
    "Norway": [60.4720, 8.4689],
    "Austria": [47.5162, 14.5501],
    "Greece": [39.0742, 21.8243],
    "Ireland": [53.1424, -7.6921],
    "Portugal": [39.3999, -8.2245],
    "Ukraine": [48.3794, 31.1656],
    "Egypt": [26.8206, 30.8025],
    "Thailand": [15.8700, 100.9925],
    "Singapore": [1.3521, 103.8198],
    "Malaysia": [4.2105, 101.9758],
    "Vietnam": [14.0583, 108.2772],
    "Philippines": [12.8797, 121.7740],
    "Indonesia": [-0.7893, 113.9213],
    "Saudi Arabia": [23.8859, 45.0792],
    "United Arab Emirates": [23.4241, 53.8478],
}

STATE_COORDINATES = {
    "Alabama": [32.806671, -86.791130],
    "Alaska": [61.370716, -152.404419],
    "Arizona": [33.729759, -111.431221],
    "Arkansas": [34.969704, -92.373123],
    "California": [36.116203, -119.681564],
    "Colorado": [39.059811, -105.311104],
    "Connecticut": [41.597782, -72.755371],
    "Delaware": [39.318523, -75.507141],
    "District of Columbia": [38.897438, -77.026817],
    "Florida": [27.766279, -81.686783],
    "Georgia": [33.040619, -83.643074],
    "Hawaii": [21.094318, -157.498337],
    "Idaho": [44.240459, -114.478828],
    "Illinois": [40.349457, -88.986137],
    "Indiana": [39.849426, -86.258278],
    "Iowa": [42.011539, -93.210526],
    "Kansas": [38.526600, -96.726486],
    "Kentucky": [37.668140, -84.670067],
    "Louisiana": [31.169546, -91.867805],
    "Maine": [44.693947, -69.381927],
    "Maryland": [39.063946, -76.802101],
    "Massachusetts": [42.230171, -71.530106],
    "Michigan": [43.326618, -84.536095],
    "Minnesota": [45.694454, -93.900192],
    "Mississippi": [32.741646, -89.678696],
    "Missouri": [38.456085, -92.288368],
    "Montana": [46.921925, -110.454353],
    "Nebraska": [41.125370, -98.268082],
    "Nevada": [38.313515, -117.055374],
    "New Hampshire": [43.452492, -71.563896],
    "New Jersey": [40.298904, -74.521011],
    "New Mexico": [34.840515, -106.248482],
    "New York": [42.165726, -74.948051],
    "North Carolina": [35.630066, -79.806419],
    "North Dakota": [47.528912, -99.784012],
    "Ohio": [40.388783, -82.764915],
    "Oklahoma": [35.565342, -96.928917],
    "Oregon": [44.572021, -122.070938],
    "Pennsylvania": [41.203323, -77.194527],
    "Rhode Island": [41.680893, -71.511780],
    "South Carolina": [33.856892, -80.945007],
    "South Dakota": [44.299782, -99.438828],
    "Tennessee": [35.747845, -86.692345],
    "Texas": [31.054487, -97.563461],
    "Utah": [40.150032, -111.862434],
    "Vermont": [44.045876, -72.710686],
    "Virginia": [37.769337, -78.169968],
    "Washington": [47.400902, -121.490494],
    "West Virginia": [38.491226, -80.954453],
    "Wisconsin": [44.268543, -89.616508],
    "Wyoming": [42.755966, -107.302490],
}

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


# --- Custom Filters ---
class LocalMetadataPostFilter(BaseNodePostprocessor):
    """
    Custom LlamaIndex post-processor for filtering retrieved nodes based on metadata.

    This allows for "Post-Retrieval Filtering", where we filter the results *after*
    semantic search but *before* sending them to the LLM. This is useful for
    complex logic that vector search alone cannot handle easily (e.g., fuzzy matching).

    Attributes:
        phase (Optional[str]): Comma-separated string of phases to keep (e.g., "PHASE2,PHASE3").
        sponsor (Optional[str]): Sponsor name to filter by (fuzzy match).
    """

    phase: Optional[str] = None
    sponsor: Optional[str] = None
    intervention: Optional[str] = None

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle=None
    ) -> List[NodeWithScore]:
        """
        Filters the list of nodes based on the configured criteria.

        Args:
            nodes (List[NodeWithScore]): The list of nodes retrieved from the index.
            query_bundle: The query bundle (unused here, but required by signature).

        Returns:
            List[NodeWithScore]: The filtered list of nodes.
        """
        new_nodes = []
        for node in nodes:
            keep = True
            metadata = node.node.metadata

            # Phase Filter (Loose match)
            # Checks if any of the requested phases are present in the node's phase string
            if self.phase:
                node_phase = str(metadata.get("phase", "")).upper().replace(" ", "")
                target_phases = [
                    p.strip().upper().replace(" ", "") for p in self.phase.split(",")
                ]
                if not any(tp in node_phase for tp in target_phases):
                    keep = False

            # Sponsor Filter (Fuzzy/Alias match)
            # Normalizes both the target and node sponsor names before comparing
            if self.sponsor and keep:
                node_sponsor = str(metadata.get("org", ""))
                norm_target = normalize_sponsor(self.sponsor).lower()
                norm_node = normalize_sponsor(node_sponsor).lower()
                if norm_target not in norm_node:
                    keep = False

            # Intervention Filter (Fuzzy match)
            if self.intervention and keep:
                node_intervention = str(metadata.get("intervention", "")).lower()
                target_intervention = self.intervention.lower()
                if target_intervention not in node_intervention:
                    keep = False

            if keep:
                new_nodes.append(node)
        return new_nodes



