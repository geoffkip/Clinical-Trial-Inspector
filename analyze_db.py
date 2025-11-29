import chromadb
import pandas as pd
import os

def analyze_db():
    db_path = "./ct_gov_index"
    if not os.path.exists(db_path):
        print(f"❌ Database directory '{db_path}' does not exist.")
        return

    print(f"📂 Loading database from {db_path}...")
    try:
        client = chromadb.PersistentClient(path=db_path)
        # Check for collection
        collections = client.list_collections()
        # In Chroma 0.6.0+, list_collections returns a list of strings (collection names)
        # In older versions, it returned objects with a .name attribute
        col_names = [c if isinstance(c, str) else c.name for c in collections]
        
        if "clinical_trials" not in col_names:
             print(f"❌ Collection 'clinical_trials' not found. Available: {col_names}")
             return

        collection = client.get_collection("clinical_trials")
        count = collection.count()
        print(f"✅ Found 'clinical_trials' collection with {count} documents.")

        # Fetch all metadata
        # Chroma get() without ids returns all if limit is not set, but let's be safe with a large limit or loop if needed.
        # For this analysis, assuming < 10k studies, we can fetch all.
        data = collection.get(include=['metadatas'])
        
        if not data['metadatas']:
            print("❌ No metadata found.")
            return

        df = pd.DataFrame(data['metadatas'])
        
        print("\n📊 --- Top 10 Sponsors ---")
        if 'org' in df.columns:
            print(df['org'].value_counts().head(10))
        else:
            print("⚠️ 'org' field not found in metadata.")

        print("\n📊 --- Phase Distribution ---")
        if 'phase' in df.columns:
            print(df['phase'].value_counts())
        else:
            print("⚠️ 'phase' field not found in metadata.")

        print("\n📊 --- Status Distribution ---")
        if 'status' in df.columns:
            print(df['status'].value_counts())
        else:
            print("⚠️ 'status' field not found in metadata.")
            
        print("\n📊 --- Top Conditions ---")
        if 'condition' in df.columns:
            # Conditions might be comma separated, let's split them
            all_conditions = []
            for conditions in df['condition'].dropna():
                all_conditions.extend([c.strip() for c in conditions.split(',')])
            print(pd.Series(all_conditions).value_counts().head(10))
        else:
            print("⚠️ 'condition' field not found in metadata.")

        print("\n📝 --- Sample Studies (Most Recent Start Years) ---")
        if 'start_year' in df.columns and 'title' in df.columns:
            # Ensure start_year is numeric
            df['start_year'] = pd.to_numeric(df['start_year'], errors='coerce')
            top_recent = df.sort_values(by='start_year', ascending=False).head(5)
            for _, row in top_recent.iterrows():
                print(f"- [{row.get('start_year', 'N/A')}] {row.get('title', 'N/A')} ({row.get('nct_id', 'N/A')})")
                print(f"  Sponsor: {row.get('org', 'N/A')}")
        
    except Exception as e:
        print(f"⚠️ Error analyzing DB: {e}")

if __name__ == "__main__":
    analyze_db()
