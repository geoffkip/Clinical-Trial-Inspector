import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.tools import search_trials
from modules.utils import load_environment

# Mark as integration test since it loads the DB
@pytest.mark.integration
def test_hybrid_search_integration():
    """
    Integration test for Hybrid Search.
    Verifies that the search_trials tool can retrieve results using the hybrid retriever.
    """
    load_environment()
    
    # Test 1: Dynamic ID Search
    # First, find a valid ID from a broad search
    print("\n🔍 Finding a valid ID for testing...")
    broad_results = search_trials.invoke({"query": "cancer"})
    
    # Extract an ID from the results
    import re
    match = re.search(r"ID: (NCT\d+)", broad_results)
    if not match:
        pytest.skip("Could not find any studies in DB to test against.")
    
    target_id = match.group(1)
    print(f"🎯 Found target ID: {target_id}. Now testing exact search...")
    
    # Now search for that specific ID
    results_id = search_trials.invoke({"query": target_id})
    
    assert "Found" in results_id
    assert target_id in results_id, f"Hybrid search failed to retrieve exact ID {target_id}"

    # Extract sponsor from the first result to ensure we test with valid data
    # Result format: "**Title** ... - Sponsor: SponsorName ..."
    sponsor_match = re.search(r"Sponsor: (.*?)\n", broad_results)
    if not sponsor_match:
        print("⚠️ Could not extract sponsor from results. Skipping hybrid test.")
        return

    target_sponsor = sponsor_match.group(1).strip()
    # Normalize it to get the simple name if possible, or just use it
    # But search_trials expects a simple name to map to variations.
    # If we pass the full name, get_sponsor_variations might return None if not mapped.
    # So let's try to find a mapped sponsor if possible, or just skip if not mapped.
    
    from modules.utils import normalize_sponsor
    simple_sponsor = normalize_sponsor(target_sponsor)
    
    # If normalization didn't change it, it might not be in our alias list.
    # But we can still try to search with it.
    
    print(f"\n🔍 Testing Hybrid Search with dynamic sponsor: '{simple_sponsor}' (Original: {target_sponsor})")
    
    # Use a generic query that likely matches the study, or just "study"
    results_hybrid = search_trials.invoke({"query": "study", "sponsor": simple_sponsor})
    
    assert "Found" in results_hybrid, f"Should find results for valid sponsor {simple_sponsor}"
    assert target_sponsor in results_hybrid or simple_sponsor in results_hybrid
