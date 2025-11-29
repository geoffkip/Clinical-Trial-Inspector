"""
Script to remove duplicate NCT IDs from the ChromaDB collection.

This script scans the 'clinical_trials' collection, identifies records with duplicate 'nct_id' metadata,
and removes the extras, keeping the one with the most metadata (richness score).
"""

import chromadb
import os
from collections import defaultdict


def calculate_richness(metadata):
    """Calculates a 'richness' score for a metadata dict based on field count and content length."""
    score = 0
    if not metadata:
        return 0

    for key, value in metadata.items():
        # Check for non-empty values
        if value is not None and str(value).strip() != "":
            score += 10  # Base points for having a populated field

            # Bonus points for content length (e.g. longer descriptions are better)
            if isinstance(value, str):
                score += len(value) / 100.0

    return score


def remove_duplicates():
    # Determine the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    db_path = os.path.join(project_root, "ct_gov_index")

    if not os.path.exists(db_path):
        print(f"❌ Database directory '{db_path}' does not exist.")
        return

    print(f"📂 Loading database from {db_path}...")
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection("clinical_trials")

        print("🔍 Scanning for duplicates...")
        # Fetch all IDs and metadata
        data = collection.get(include=["metadatas"])

        ids = data["ids"]
        metadatas = data["metadatas"]

        if not ids:
            print("Database is empty.")
            return

        # Map NCT ID -> List of (Chroma ID, Metadata)
        nct_map = defaultdict(list)
        for i, meta in enumerate(metadatas):
            nct_id = meta.get("nct_id")
            if nct_id:
                nct_map[nct_id].append((ids[i], meta))

        # Identify duplicates
        duplicates = {k: v for k, v in nct_map.items() if len(v) > 1}

        if not duplicates:
            print("✅ No duplicates found. Database is clean.")
            return

        print(f"⚠️ Found {len(duplicates)} NCT IDs with duplicate records.")

        ids_to_delete = []
        for nct_id, records in duplicates.items():
            # Sort records by richness score (descending)
            # records is a list of tuples: (chroma_id, metadata)
            records.sort(key=lambda x: calculate_richness(x[1]), reverse=True)

            best_record = records[0]
            extras = records[1:]

            best_score = calculate_richness(best_record[1])

            print(
                f"   - {nct_id}: Found {len(records)} copies. "
                f"Keeping {best_record[0]} (Score: {best_score:.1f}). "
                f"Removing {len(extras)}."
            )

            ids_to_delete.extend([r[0] for r in extras])

        if ids_to_delete:
            print(f"🗑️ Deleting {len(ids_to_delete)} duplicate records...")
            collection.delete(ids=ids_to_delete)
            print("🎉 Deduplication complete!")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    remove_duplicates()
