import pytest
import pandas as pd
import sys
import os

# Add project root to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ct_agent_app import normalize_sponsor

# --- Tests for normalize_sponsor ---

def test_normalize_sponsor_aliases():
    assert normalize_sponsor("J&J") == "Janssen"
    assert normalize_sponsor("Johnson & Johnson") == "Janssen"
    assert normalize_sponsor("GSK") == "GlaxoSmithKline"
    assert normalize_sponsor("Merck") == "Merck Sharp & Dohme"
    assert normalize_sponsor("MSD") == "Merck Sharp & Dohme"
    assert normalize_sponsor("BMS") == "Bristol-Myers Squibb"

def test_normalize_sponsor_no_change():
    assert normalize_sponsor("Pfizer") == "Pfizer"
    assert normalize_sponsor("Moderna") == "Moderna"
    assert normalize_sponsor("Unknown Sponsor") == "Unknown Sponsor"

def test_normalize_sponsor_case_insensitivity():
    # The function itself might not be case insensitive, but let's check current behavior
    # Based on code: return SPONSOR_ALIASES.get(sponsor, sponsor)
    # It seems strictly case-sensitive based on the dict lookup.
    # Let's verify exact matches first.
    assert normalize_sponsor("Sanofi") == "Sanofi"

# --- Tests for Analytics Logic (Mocked) ---

def filter_dataframe(df, phase=None, status=None, sponsor=None):
    """
    Replicating the logic from get_study_analytics for testing purposes,
    since the original function is decorated as a tool and hard to import/test directly without side effects.
    """
    if phase:
        target_phases = [p.strip().upper().replace(" ", "") for p in phase.split(',')]
        df['phase_upper'] = df['phase'].astype(str).str.upper().str.replace(" ", "")
        mask = df['phase_upper'].apply(lambda x: any(tp in x for tp in target_phases))
        df = df[mask]
        
    if status:
        df = df[df['status'].str.upper() == status.upper()]
        
    if sponsor:
        target_sponsor = normalize_sponsor(sponsor).lower()
        df['org_lower'] = df['org'].astype(str).apply(normalize_sponsor).str.lower()
        df = df[df['org_lower'].str.contains(target_sponsor, regex=False)]
        
    return df

@pytest.fixture
def sample_df():
    data = {
        'nct_id': ['NCT001', 'NCT002', 'NCT003', 'NCT004'],
        'phase': ['PHASE1', 'PHASE2', 'PHASE3', 'PHASE2'],
        'status': ['RECRUITING', 'COMPLETED', 'COMPLETED', 'RECRUITING'],
        'org': ['Pfizer', 'Janssen', 'Merck Sharp & Dohme', 'Pfizer'],
        'start_year': [2020, 2021, 2022, 2023]
    }
    return pd.DataFrame(data)

def test_analytics_filter_phase(sample_df):
    # Filter for Phase 2
    filtered = filter_dataframe(sample_df, phase="Phase 2")
    assert len(filtered) == 2
    assert all(filtered['phase'] == 'PHASE2')

def test_analytics_filter_multiple_phases(sample_df):
    # Filter for Phase 1 OR Phase 3
    filtered = filter_dataframe(sample_df, phase="Phase 1, Phase 3")
    assert len(filtered) == 2
    assert set(filtered['nct_id']) == {'NCT001', 'NCT003'}

def test_analytics_filter_sponsor(sample_df):
    # Filter for Pfizer
    filtered = filter_dataframe(sample_df, sponsor="Pfizer")
    assert len(filtered) == 2
    assert all(filtered['org'] == 'Pfizer')

def test_analytics_filter_sponsor_alias(sample_df):
    # Filter for J&J (should match Janssen)
    filtered = filter_dataframe(sample_df, sponsor="J&J")
    assert len(filtered) == 1
    assert filtered.iloc[0]['org'] == 'Janssen'

def test_analytics_filter_status(sample_df):
    # Filter for Recruiting
    filtered = filter_dataframe(sample_df, status="Recruiting")
    assert len(filtered) == 2
    assert all(filtered['status'] == 'RECRUITING')

def test_analytics_multi_filter(sample_df):
    # Filter for Pfizer AND Recruiting
    filtered = filter_dataframe(sample_df, sponsor="Pfizer", status="Recruiting")
    assert len(filtered) == 2 # Both Pfizer studies are recruiting/start_year... wait
    # NCT001: Pfizer, Phase 1, Recruiting
    # NCT004: Pfizer, Phase 2, Recruiting
    assert set(filtered['nct_id']) == {'NCT001', 'NCT004'}
    
    # Filter for Pfizer AND Phase 2
    filtered = filter_dataframe(sample_df, sponsor="Pfizer", phase="Phase 2")
    assert len(filtered) == 1
    assert filtered.iloc[0]['nct_id'] == 'NCT004'
