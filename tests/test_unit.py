import pytest
import pandas as pd
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.utils import normalize_sponsor  # noqa: E402
from modules.tools import expand_query  # noqa: E402
from modules.graph_viz import build_graph  # noqa: E402
from llama_index.core.schema import NodeWithScore, TextNode  # noqa: E402

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


# --- Tests for Analytics Logic (Mocked) ---


def filter_dataframe(df, phase=None, status=None, sponsor=None, intervention=None):
    """
    Replicating the logic from get_study_analytics for testing purposes.
    """
    if phase:
        target_phases = [p.strip().upper().replace(" ", "") for p in phase.split(",")]
        df["phase_upper"] = df["phase"].astype(str).str.upper().str.replace(" ", "")
        mask = df["phase_upper"].apply(lambda x: any(tp in x for tp in target_phases))
        df = df[mask]

    if status:
        df = df[df["status"].str.upper() == status.upper()]

    if sponsor:
        target_sponsor = normalize_sponsor(sponsor).lower()
        df["org_lower"] = df["org"].astype(str).apply(normalize_sponsor).str.lower()
        df = df[df["org_lower"].str.contains(target_sponsor, regex=False)]

    if intervention:
        target_intervention = intervention.lower()
        df["intervention_lower"] = df["intervention"].astype(str).str.lower()
        df = df[df["intervention_lower"].str.contains(target_intervention, regex=False)]

    return df


@pytest.fixture
def sample_df():
    data = {
        "nct_id": ["NCT001", "NCT002", "NCT003", "NCT004"],
        "phase": ["PHASE1", "PHASE2", "PHASE3", "PHASE2"],
        "status": ["RECRUITING", "COMPLETED", "COMPLETED", "RECRUITING"],
        "org": ["Pfizer", "Janssen", "Merck Sharp & Dohme", "Pfizer"],
        "intervention": ["Drug A", "Drug B", "Keytruda", "Drug A + Drug C"],
        "start_year": [2020, 2021, 2022, 2023],
        "title": [
            "Study of Drug A",
            "Study of Drug B",
            "Keytruda Trial",
            "Combo Study",
        ],
        "condition": ["Cancer", "Diabetes", "Lung Cancer", "Cancer"],
    }
    return pd.DataFrame(data)


def test_analytics_filter_intervention(sample_df):
    # Filter for Keytruda
    filtered = filter_dataframe(sample_df, intervention="Keytruda")
    assert len(filtered) == 1
    assert filtered.iloc[0]["nct_id"] == "NCT003"


def test_analytics_filter_intervention_partial(sample_df):
    # Filter for "Drug A" (should match NCT001 and NCT004)
    filtered = filter_dataframe(sample_df, intervention="Drug A")
    assert len(filtered) == 2
    assert set(filtered["nct_id"]) == {"NCT001", "NCT004"}


# --- Tests for Query Expansion ---


@patch("modules.tools.Settings")
def test_expand_query(mock_settings):
    # Mock LLM response
    mock_response = MagicMock()
    mock_response.text = "Expanded Query: cancer OR carcinoma OR tumor"
    mock_settings.llm.complete.return_value = mock_response

    query = "cancer"
    expanded = expand_query(query)

    assert "cancer OR carcinoma OR tumor" in expanded
    mock_settings.llm.complete.assert_called_once()


def test_expand_query_skip_long():
    long_query = "this is a very long query that should definitely be skipped because it has too many words"
    assert expand_query(long_query) == long_query





# --- Tests for Graph Visualization ---


def test_build_graph():
    data = [
        {"nct_id": "NCT1", "title": "Study 1", "org": "Pfizer", "condition": "Cancer"},
        {
            "nct_id": "NCT2",
            "title": "Study 2",
            "org": "Merck",
            "condition": "Cancer, Diabetes",
        },
    ]

    nodes, edges, config = build_graph(data)

    # Check Nodes
    # 2 Studies + 2 Sponsors + 2 Conditions (Cancer, Diabetes) = 6 Nodes
    assert len(nodes) == 6

    node_ids = [n.id for n in nodes]
    assert "NCT1" in node_ids
    assert "Pfizer" in node_ids
    assert "Cancer" in node_ids

    # Check Edges
    # NCT1 -> Pfizer, NCT1 -> Cancer (2 edges)
    # NCT2 -> Merck, NCT2 -> Cancer, NCT2 -> Diabetes (3 edges)
    assert len(edges) == 5
