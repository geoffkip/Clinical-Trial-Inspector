"""
Script to remove duplicate records from the LanceDB database.

This script scans the 'clinical_trials' table, identifies records with duplicate content 
(same 'nct_id' AND same 'text'), and removes the extras.

It uses a safe "Fetch -> Dedupe -> Overwrite" strategy:
1. Identifies NCT IDs that have duplicates.
2. For each such NCT ID, fetches ALL its records (chunks).
3. Deduplicates these records in memory based on their text content.
4. Deletes ALL records for that NCT ID from the database.
5. Re-inserts the unique records.

This ensures that valid chunks of the same study are PRESERVED, while exact duplicates are removed.
"""

import os
import pandas as pd
import lancedb

import argparse

def calculate_richness(record):
    """Calculates a 'richness' score for a record based on metadata field count and content length."""
    score = 0
    if not record:
        return 0

    for key, value in record.items():
        if key == "vector": continue 
        
        # Handle nested metadata
        if key == "metadata" and isinstance(value, dict):
            score += calculate_richness(value) # Recurse
            continue

        # Check for non-empty values
        if value is not None and str(value).strip() != "":
            score += 10  # Base points for having a populated field

            # Bonus points for content length
            if isinstance(value, str):
                score += len(value) / 100.0

    return score

def remove_duplicates(dry_run=False):
    # Determine the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    db_path = os.path.join(project_root, "ct_gov_lancedb")

    if not os.path.exists(db_path):
        print(f"❌ Database directory '{db_path}' does not exist.")
        return

    print(f"📂 Loading database from {db_path}...")
    if dry_run:
        print("🧪 RUNNING IN DRY-RUN MODE (No changes will be made)")

    try:
        db = lancedb.connect(db_path)
        tbl = db.open_table("clinical_trials")

        print("🔍 Scanning for duplicates...")
        # Fetch all data
        df = tbl.to_pandas()

        if df.empty:
            print("Database is empty.")
            return

        # Create a working copy to flatten metadata for analysis
        working_df = df.copy()
        if "metadata" in working_df.columns:
             # Flatten metadata
             meta_df = pd.json_normalize(working_df["metadata"])
             # We drop the original metadata column from working_df and join the flattened one
             working_df = pd.concat([working_df.drop(columns=["metadata"]), meta_df], axis=1)

        if "nct_id" not in working_df.columns:
            print("❌ 'nct_id' column not found (checked metadata too).")
            return

        if "text" not in working_df.columns:
            print("❌ 'text' column not found. Cannot safely deduplicate chunks.")
            return

        # Identify duplicates based on (nct_id, text) using the flattened view
        duplicates_mask = working_df.duplicated(subset=["nct_id", "text"], keep=False)
        
        # We use the mask on working_df to find the IDs
        duplicates_working_df = working_df[duplicates_mask]

        if duplicates_working_df.empty:
            print("✅ No exact duplicates found. Database is clean.")
            return

        unique_duplicate_ids = duplicates_working_df["nct_id"].unique()
        print(f"⚠️ Found duplicates affecting {len(unique_duplicate_ids)} studies (NCT IDs).")

        total_deleted = 0
        total_reinserted = 0
        
        # Process each affected NCT ID
        for nct_id in unique_duplicate_ids:
            # Get indices from working_df where nct_id matches
            # This ensures we are looking at the right rows in the ORIGINAL df
            indices = working_df[working_df["nct_id"] == nct_id].index
            
            # Extract original records (preserving structure)
            study_records_df = df.loc[indices]
            original_count = len(study_records_df)
            
            unique_records = []
            seen_texts = set()
            
            records = study_records_df.to_dict("records")
            records.sort(key=calculate_richness, reverse=True)
            
            for record in records:
                text_content = record.get("text", "")
                if text_content not in seen_texts:
                    unique_records.append(record)
                    seen_texts.add(text_content)
            
            new_count = len(unique_records)
            
            if new_count < original_count:
                print(f"   - {nct_id}: Reducing {original_count} -> {new_count} records.")
                
                if not dry_run:
                    # Delete using the ID (LanceDB SQL filter)
                    # Note: In LanceDB SQL, if nct_id is in metadata struct, we access it via metadata.nct_id
                    # But wait, tbl.delete() takes a SQL string.
                    # If the schema has 'metadata' struct, we must use 'metadata.nct_id'.
                    # If it was flattened (unlikely for the table itself), we use 'nct_id'.
                    
                    # We check if 'nct_id' is a top-level column in the original DF
                    if "nct_id" in df.columns:
                        where_clause = f"nct_id = '{nct_id}'"
                    else:
                        where_clause = f"metadata.nct_id = '{nct_id}'"
                        
                    tbl.delete(where_clause)
                    
                    if unique_records:
                        tbl.add(unique_records)
                
                total_deleted += original_count
                total_reinserted += new_count
            else:
                print(f"   - {nct_id}: No reduction needed (false positive?).")

        if dry_run:
            print(f"\n🧪 DRY RUN COMPLETE.")
            print(f"   - WOULD remove {total_deleted - total_reinserted} duplicate records.")
            print(f"   - WOULD preserve {total_reinserted} unique chunks.")
        else:
            print(f"\n🎉 Deduplication complete!")
            print(f"   - Removed {total_deleted - total_reinserted} duplicate records.")
            print(f"   - Preserved {total_reinserted} unique chunks.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove duplicate records from LanceDB.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate the process without making changes.")
    args = parser.parse_args()
    
    remove_duplicates(dry_run=args.dry_run)
