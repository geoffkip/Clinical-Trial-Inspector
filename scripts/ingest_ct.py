"""
Data Ingestion Script for Clinical Trial Agent.

This script fetches clinical trial data from the ClinicalTrials.gov API (v2),
processes it into a rich text format, and ingests it into a local ChromaDB vector index
using LlamaIndex and PubMedBERT embeddings.

Features:
- **Pagination**: Fetches data in batches using the API's pagination tokens.
- **Robustness**: Implements retry logic for network errors.
- **Efficiency**: Uses batch insertion and reuses the existing index.
- **Progress Tracking**: Displays a progress bar using `tqdm`.
"""

import requests
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
import argparse
import time
from tqdm import tqdm
import os
import concurrent.futures

# LlamaIndex Imports
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()

# Disable LLM for ingestion (we only need embeddings, not generation)
Settings.llm = None


def clean_text(text: str) -> str:
    """
    Cleans raw text by removing HTML tags and normalizing whitespace.

    Args:
        text (str): The raw text string.

    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove multiple spaces/newlines and trim
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_trials_generator(
    years: int = 5, max_studies: int = 1000, status: list = None, phases: list = None
):
    """
    Yields batches of clinical trials from the ClinicalTrials.gov API.

    Handles pagination automatically and implements retry logic for API requests.

    Args:
        years (int): Number of years to look back for study start dates.
        max_studies (int): Maximum total number of studies to fetch (-1 for all).
        status (list): List of status strings to filter by (e.g., ["RECRUITING"]).
        phases (list): List of phase strings to filter by (e.g., ["PHASE2"]).

    Yields:
        list: A batch of study dictionaries (JSON objects).
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"

    # Calculate start date for filtering
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    print("📡 Connecting to CT.gov API...")
    print(f"🔎 Fetching trials starting after: {start_date}")
    if status:
        print(f"   Filters - Status: {status}")
    if phases:
        print(f"   Filters - Phases: {phases}")

    fetched_count = 0
    next_page_token = None

    # If max_studies is -1, fetch ALL studies (infinite limit)
    fetch_limit = float("inf") if max_studies == -1 else max_studies

    while fetched_count < fetch_limit:
        # Determine batch size (max 1000 per API limit)
        current_limit = 1000
        if max_studies != -1:
            current_limit = min(1000, max_studies - fetched_count)

        # --- Query Construction ---
        # Build the query term using the API's syntax
        query_parts = [f"AREA[StartDate]RANGE[{start_date},MAX]"]

        if status:
            status_str = " OR ".join(status)
            query_parts.append(f"AREA[OverallStatus]({status_str})")

        if phases:
            phase_str = " OR ".join(phases)
            query_parts.append(f"AREA[Phase]({phase_str})")

        full_query = " AND ".join(query_parts)

        params = {
            "query.term": full_query,
            "pageSize": current_limit,
            # Request specific fields to minimize payload size
            "fields": ",".join(
                [
                    "protocolSection.identificationModule.nctId",
                    "protocolSection.identificationModule.briefTitle",
                    "protocolSection.identificationModule.officialTitle",
                    "protocolSection.identificationModule.organization",
                    "protocolSection.statusModule.overallStatus",
                    "protocolSection.statusModule.startDateStruct",
                    "protocolSection.statusModule.completionDateStruct",
                    "protocolSection.designModule.phases",
                    "protocolSection.designModule.studyType",
                    "protocolSection.eligibilityModule.eligibilityCriteria",
                    "protocolSection.eligibilityModule.sex",
                    "protocolSection.eligibilityModule.stdAges",
                    "protocolSection.descriptionModule.briefSummary",
                    "protocolSection.conditionsModule.conditions",
                    "protocolSection.outcomesModule.primaryOutcomes",
                    "protocolSection.contactsLocationsModule.locations",
                    "protocolSection.armsInterventionsModule",
                ]
            ),
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        # --- Retry Logic ---
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.get(base_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    studies = data.get("studies", [])

                    if not studies:
                        return  # Stop generator if no studies returned

                    yield studies

                    fetched_count += len(studies)
                    next_page_token = data.get("nextPageToken")

                    if not next_page_token:
                        return  # Stop generator if no more pages

                    break  # Success, exit retry loop
                else:
                    print(f"❌ API Error: {response.status_code} - {response.text}")
                    if attempt < retries - 1:
                        time.sleep(2)
                    else:
                        return  # Stop generator on persistent error
            except Exception as e:
                print(f"❌ Request Error (Attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    return  # Stop generator


def process_study(study):
    """
    Processes a single study dictionary into a LlamaIndex Document.
    This function is designed to be run in parallel.
    """
    try:
        # Extract Modules
        protocol = study.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        design = protocol.get("designModule", {})
        eligibility = protocol.get("eligibilityModule", {})
        description = protocol.get("descriptionModule", {})
        conditions_module = protocol.get("conditionsModule", {})
        outcomes_module = protocol.get("outcomesModule", {})
        arms_interventions_module = protocol.get("armsInterventionsModule", {})
        locations_module = protocol.get("contactsLocationsModule", {})

        # Extract Fields
        nct_id = identification.get("nctId", "N/A")
        title = identification.get("briefTitle", "N/A")
        official_title = identification.get("officialTitle", "N/A")
        org = identification.get("organization", {}).get("fullName", "N/A")
        summary = clean_text(description.get("briefSummary", "N/A"))

        overall_status = status_module.get("overallStatus", "N/A")
        start_date = status_module.get("startDateStruct", {}).get("date", "N/A")
        completion_date = status_module.get("completionDateStruct", {}).get(
            "date", "N/A"
        )

        phases = ", ".join(design.get("phases", []))
        study_type = design.get("studyType", "N/A")

        criteria = clean_text(eligibility.get("eligibilityCriteria", "N/A"))
        gender = eligibility.get("sex", "N/A")
        ages = ", ".join(eligibility.get("stdAges", []))

        conditions = ", ".join(conditions_module.get("conditions", []))

        interventions = []
        for interv in arms_interventions_module.get("interventions", []):
            name = interv.get("name", "")
            type_ = interv.get("type", "")
            interventions.append(f"{type_}: {name}")
        interventions_str = "; ".join(interventions)

        primary_outcomes = []
        for outcome in outcomes_module.get("primaryOutcomes", []):
            measure = outcome.get("measure", "")
            desc = outcome.get("description", "")
            primary_outcomes.append(f"- {measure}: {desc}")
        outcomes_str = clean_text("\n".join(primary_outcomes))

        locations = []
        for loc in locations_module.get("locations", []):
            facility = loc.get("facility", "N/A")
            city = loc.get("city", "")
            country = loc.get("country", "")
            locations.append(f"{facility} ({city}, {country})")
        locations_str = "; ".join(locations[:5])  # Limit to 5 locations to save space

        # Construct Rich Page Content with Markdown Headers
        # This text is what gets embedded and searched
        page_content = (
            f"# {title}\n"
            f"**NCT ID:** {nct_id}\n"
            f"**Official Title:** {official_title}\n"
            f"**Sponsor:** {org}\n"
            f"**Status:** {overall_status}\n"
            f"**Phase:** {phases}\n"
            f"**Study Type:** {study_type}\n"
            f"**Start Date:** {start_date}\n"
            f"**Completion Date:** {completion_date}\n\n"
            f"## Summary\n{summary}\n\n"
            f"## Conditions\n{conditions}\n\n"
            f"## Interventions\n{interventions_str}\n\n"
            f"## Eligibility Criteria\n"
            f"**Gender:** {gender}\n"
            f"**Ages:** {ages}\n"
            f"**Criteria:**\n{criteria}\n\n"
            f"## Primary Outcomes\n{outcomes_str}\n\n"
            f"## Locations\n{locations_str}"
        )

        # Metadata for filtering (Structured Data)
        metadata = {
            "nct_id": nct_id,
            "title": title,
            "org": org,
            "status": overall_status,
            "phase": phases,
            "study_type": study_type,
            "start_year": (int(start_date.split("-")[0]) if start_date != "N/A" else 0),
            "condition": conditions,
            "intervention": interventions_str,
            "country": (
                locations[0].split(",")[-1].strip() if locations else "Unknown"
            ),
        }

        return Document(text=page_content, metadata=metadata, id_=nct_id)
    except Exception as e:
        print(
            f"⚠️ Error processing study {study.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'Unknown')}: {e}"
        )
        return None


def run_ingestion():
    """
    Main execution function for the ingestion script.
    Parses arguments, initializes the index, and runs the ingestion loop.
    """
    parser = argparse.ArgumentParser(description="Ingest Clinical Trials data.")
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Number of studies to ingest. Set to -1 for ALL.",
    )
    parser.add_argument(
        "--years", type=int, default=10, help="Number of years to look back."
    )
    parser.add_argument(
        "--status",
        type=str,
        default="COMPLETED",
        help="Comma-separated list of statuses (e.g., COMPLETED,RECRUITING).",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="PHASE1,PHASE2,PHASE3,PHASE4",
        help="Comma-separated list of phases (e.g., PHASE2,PHASE3).",
    )
    args = parser.parse_args()

    status_list = args.status.split(",") if args.status else []
    phase_list = args.phases.split(",") if args.phases else []

    print(f"⚙️ Configuration: Limit={args.limit}, Years={args.years}")
    print(f"   Status Filter: {status_list}")
    print(f"   Phase Filter: {phase_list}")

    # --- INITIALIZE LLAMAINDEX COMPONENTS ---
    print("🧠 Initializing LlamaIndex Embeddings (PubMedBERT)...")
    embed_model = HuggingFaceEmbedding(model_name="pritamdeka/S-PubMedBert-MS-MARCO")

    # Initialize ChromaDB (Persistent)
    print("🚀 Initializing ChromaDB...")

    # Determine the project root directory (one level up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    db_path = os.path.join(project_root, "ct_gov_index")

    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("clinical_trials")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Initialize Index ONCE
    # We pass the storage context to link it to the vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embed_model
    )

    total_ingested = 0

    # Progress Bar
    pbar = tqdm(
        total=args.limit if args.limit > 0 else float("inf"),
        desc="Ingesting Studies",
        unit="study",
    )

    # --- INGESTION LOOP ---
    # Use ProcessPoolExecutor for parallel processing of study data
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for batch_studies in fetch_trials_generator(
            years=args.years,
            max_studies=args.limit,
            status=status_list,
            phases=phase_list,
        ):
            # Parallelize the processing of the batch
            # map returns an iterator, so we convert to list to trigger execution
            documents_iter = executor.map(process_study, batch_studies)

            # Filter out None results (errors)
            documents = [doc for doc in documents_iter if doc is not None]

            if documents:
                # Overwrite Logic:
                # To avoid duplicates and ensure we update existing studies with new fields (like interventions),
                # we first delete any existing records with the same NCT IDs.
                doc_ids = [doc.id_ for doc in documents]
                try:
                    # Delete by metadata nct_id to be safe (covers all nodes for a doc)
                    chroma_collection.delete(where={"nct_id": {"$in": doc_ids}})
                except Exception:
                    # Ignore if collection is empty or other minor errors
                    pass

                # Efficient Batch Insertion
                # We convert documents to nodes and insert them into the index.
                # This handles embedding generation automatically.
                parser = Settings.node_parser
                nodes = parser.get_nodes_from_documents(documents)

                index.insert_nodes(nodes)

                total_ingested += len(documents)
                pbar.update(len(documents))

    pbar.close()
    print(f"🎉 Ingestion Complete! Total studies in DB: {total_ingested}")


if __name__ == "__main__":
    run_ingestion()
