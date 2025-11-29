import chromadb
from dotenv import load_dotenv
import os
import re
from tqdm import tqdm

load_dotenv()

# List of US States for extraction
US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", 
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", 
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", 
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", 
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", 
    "Wisconsin", "Wyoming", "District of Columbia"
]

def update_states():
    db_path = "./ct_gov_index"
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("clinical_trials")
    
    print("Fetching all documents...")
    results = collection.get(include=["metadatas", "documents"])
    ids = results["ids"]
    metadatas = results["metadatas"]
    documents = results["documents"]
    
    updates_ids = []
    updates_metadatas = []
    
    print(f"Processing {len(ids)} documents...")
    
    for i, doc_id in enumerate(tqdm(ids)):
        meta = metadatas[i]
        text = documents[i]
        
        # Only process if country is United States (or similar)
        country = meta.get("country", "")
        if "United States" not in country:
            continue
            
        # Extract State from text
        # Look for "Locations" section
        state_found = None
        
        # Strategy 1: Look for "City, State, United States" pattern in Locations section
        if "## Locations" in text:
            loc_section = text.split("## Locations")[1]
            for state in US_STATES:
                if state in loc_section:
                    state_found = state
                    break
        
        # Strategy 2: Fallback to searching entire text if not found in section
        if not state_found:
             for state in US_STATES:
                # Simple check: "State, United States"
                if f"{state}, United States" in text:
                    state_found = state
                    break
        
        if state_found:
            meta["state"] = state_found
            updates_ids.append(doc_id)
            updates_metadatas.append(meta)
            
    if updates_ids:
        print(f"Updating {len(updates_ids)} documents with State info...")
        # Update in batches of 1000
        batch_size = 1000
        for i in range(0, len(updates_ids), batch_size):
            batch_ids = updates_ids[i:i+batch_size]
            batch_metas = updates_metadatas[i:i+batch_size]
            collection.update(ids=batch_ids, metadatas=batch_metas)
        print("Update complete! ✅")
    else:
        print("No updates needed.")

if __name__ == "__main__":
    update_states()
