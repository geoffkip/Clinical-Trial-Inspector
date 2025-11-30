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
    db_path = os.path.join(project_root, "ct_gov_lancedb")

    if not os.path.exists(db_path):
        print(f"❌ Database directory '{db_path}' does not exist.")
        return

    print(f"📂 Loading database from {db_path}...")
    try:
        import lancedb
        db = lancedb.connect(db_path)
        tbl = db.open_table("clinical_trials")

        print("🔍 Scanning for duplicates...")
        # Fetch all data
        df = tbl.to_pandas()

        if df.empty:
            print("Database is empty.")
            return

        # Check for duplicates
        if "nct_id" not in df.columns:
            print("❌ 'nct_id' column not found.")
            return

        # Check for duplicates based on NCT ID AND Text content
        # This prevents deleting valid chunks of the same document
        if "text" in df.columns:
            duplicates = df[df.duplicated(subset=["nct_id", "text"], keep=False)]
        else:
            # Fallback if text column is missing (unlikely in LanceDB)
            duplicates = df[df.duplicated(subset="nct_id", keep=False)]

        if duplicates.empty:
            print("✅ No duplicates found. Database is clean.")
            return

        print(f"⚠️ Found {len(duplicates)} duplicate records.")

        # Group by NCT ID
        grouped = duplicates.groupby("nct_id")
        
        ids_to_reinsert = []
        ids_to_delete = []

        for nct_id, group in grouped:
            # Calculate richness for each
            # We assume metadata columns are available or we use the whole row
            # Convert to dicts
            records = group.to_dict("records")
            
            # Calculate score
            # We need to handle nested metadata if present, but to_pandas flattens or keeps struct
            # For simplicity, we count non-null fields in the row
            def score_record(row):
                score = 0
                for k, v in row.items():
                    if v is not None and str(v).strip() != "":
                        score += 10
                        if isinstance(v, str):
                            score += len(v) / 100.0
                return score

            records.sort(key=score_record, reverse=True)
            best_record = records[0]
            
            print(f"   - {nct_id}: Found {len(records)} copies. Keeping best.")
            
            # We will delete ALL records for this NCT ID and re-insert the best one
            ids_to_delete.append(nct_id)
            ids_to_reinsert.append(best_record)

        if ids_to_delete:
            print(f"🗑️ Deleting duplicates for {len(ids_to_delete)} NCT IDs...")
            # Delete all with these IDs
            # Construct where clause: nct_id IN (...)
            # Process in batches to avoid huge query
            batch_size = 100
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i:i+batch_size]
                ids_str = ", ".join([f"'{id}'" for id in batch])
                tbl.delete(f"nct_id IN ({ids_str})")
            
            print(f"📥 Re-inserting {len(ids_to_reinsert)} best records...")
            tbl.add(ids_to_reinsert)
            
            print("🎉 Deduplication complete!")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    remove_duplicates()
