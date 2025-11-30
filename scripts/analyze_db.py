"""
Database Analysis Script.

This script connects to the local ChromaDB vector store and performs a quick analysis
of the ingested clinical trial data. It prints statistics about:
- Top Sponsors
- Phase Distribution
- Status Distribution
- Top Medical Conditions
- Sample of Recent Studies

Usage:
    python scripts/analyze_db.py
    # OR
    cd scripts && python analyze_db.py
"""

import lancedb
import pandas as pd
import os


def analyze_db():
    """
    Connects to ChromaDB and prints summary statistics of the dataset.
    """
    # Determine the project root directory (one level up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    db_path = os.path.join(project_root, "ct_gov_lancedb")

    if not os.path.exists(db_path):
        print(f"❌ Database directory '{db_path}' does not exist.")
        print("   Please run 'python scripts/ingest_ct.py' first to ingest data.")
        return

    print(f"📂 Loading database from {db_path}...")
    try:
        db = lancedb.connect(db_path)
        
        # Check for table existence
        if "clinical_trials" not in db.table_names():
            print(f"❌ Table 'clinical_trials' not found. Available: {db.table_names()}")
            return

        tbl = db.open_table("clinical_trials")
        count = len(tbl)
        print(f"✅ Found 'clinical_trials' table with {count} documents.")

        # Fetch all data for analysis
        df = tbl.to_pandas()
        
        if df.empty:
            print("❌ No data found.")
            return
            
        # Handle metadata if nested (LlamaIndex might nest it)
        if "metadata" in df.columns:
             # Try to flatten if it's a struct/dict
             try:
                 meta_df = pd.json_normalize(df["metadata"])
                 # Merge with original df or just use meta_df for analysis
                 # We'll use meta_df for the metadata fields analysis
                 # But we might need 'text' from original
                 df = pd.concat([df.drop(columns=["metadata"]), meta_df], axis=1)
             except:
                 pass

        if "nct_id" in df.columns:
            unique_ncts = df["nct_id"].nunique()
            print(f"🔢 Unique NCT IDs: {unique_ncts}")
            if unique_ncts < count:
                print(f"⚠️ Warning: {count - unique_ncts} duplicate records found!")
        else:
            print("⚠️ 'nct_id' field not found in metadata.")

        # --- Analysis Sections ---

        print("\n📊 --- Top 10 Sponsors ---")
        if "org" in df.columns:
            print(df["org"].value_counts().head(10))
        else:
            print("⚠️ 'org' field not found in metadata.")

        print("\n📊 --- Phase Distribution ---")
        if "phase" in df.columns:
            print(df["phase"].value_counts())
        else:
            print("⚠️ 'phase' field not found in metadata.")

        print("\n📊 --- Status Distribution ---")
        if "status" in df.columns:
            print(df["status"].value_counts())
        else:
            print("⚠️ 'status' field not found in metadata.")

        print("\n📊 --- Top Conditions ---")
        if "condition" in df.columns:
            # Conditions are comma-separated strings, so we split and explode them
            all_conditions = []
            for conditions in df["condition"].dropna():
                all_conditions.extend([c.strip() for c in conditions.split(",")])
            print(pd.Series(all_conditions).value_counts().head(10))
        else:
            print("⚠️ 'condition' field not found in metadata.")

        print("\n📊 --- Top Interventions ---")
        if "intervention" in df.columns:
            # Interventions are semicolon-separated strings (from ingest_ct.py), so we split by "; "
            all_interventions = []
            for interventions in df["intervention"].dropna():
                # Split by semicolon and strip whitespace
                parts = [i.strip() for i in interventions.split(";") if i.strip()]
                all_interventions.extend(parts)
            
            if all_interventions:
                print(pd.Series(all_interventions).value_counts().head(20))
            else:
                print("No interventions found.")
        else:
            print("⚠️ 'intervention' field not found in metadata.")

        print("\n📝 --- Sample Studies (Most Recent Start Years) ---")
        if "start_year" in df.columns and "title" in df.columns:
            # Ensure start_year is numeric for sorting
            df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")
            top_recent = df.sort_values(by="start_year", ascending=False).head(5)
            for _, row in top_recent.iterrows():
                print(
                    f"- [{row.get('start_year', 'N/A')}] {row.get('title', 'N/A')} ({row.get('nct_id', 'N/A')})"
                )
                print(f"  Sponsor: {row.get('org', 'N/A')}")
                print(f"  Intervention: {row.get('intervention', 'N/A')}")

        print("\n📊 --- Intervention Check ---")
        if "intervention" in df.columns:
            non_empty = df[df["intervention"].str.len() > 0]
            print(f"Total records with interventions: {len(non_empty)}")
            if not non_empty.empty:
                print("Sample Intervention:", non_empty.iloc[0]["intervention"])
        else:
            print("⚠️ 'intervention' field not found.")

    except Exception as e:
        print(f"⚠️ Error analyzing DB: {e}")


if __name__ == "__main__":
    analyze_db()
