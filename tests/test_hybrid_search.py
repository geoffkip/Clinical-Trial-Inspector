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

    # Test 2: Specific Drug Name + Sponsor (Hybrid)
    query_drug = "Teclistamab"
    sponsor = "Janssen"
    print(f"\n🔍 Testing Hybrid Search: {query_drug} + {sponsor}")
    results_hybrid = search_trials.invoke({"query": query_drug, "sponsor": sponsor})
    
    assert "Found" in results_hybrid, "Should find results for valid drug/sponsor"
    # Check for presence of key terms in the output
    assert "Janssen" in results_hybrid or "Johnson & Johnson" in results_hybrid
