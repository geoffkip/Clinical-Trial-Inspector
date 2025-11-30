"""
Clinical Trial Inspector Agent Application.

This is the main Streamlit application script. It orchestrates:
1.  **LLM & Agents**: Initializes Google Gemini and the LangChain agent.
2.  **RAG Pipeline**: Loads the LlamaIndex vector store for semantic retrieval.
3.  **User Interface**: Renders the Streamlit UI with tabs for Chat, Analytics, and Raw Data.
4.  **Visualization**: Handles dynamic chart generation using Altair.
"""

import streamlit as st
import pandas as pd
import os
import altair as alt
import logging
from dotenv import load_dotenv

# Suppress logging
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# Module Imports
from modules.utils import load_index, setup_llama_index, COUNTRY_COORDINATES, STATE_COORDINATES

# ... (imports)
from modules.tools import (
    search_trials,
    find_similar_studies,
    get_study_analytics,
    compare_studies,
    get_study_details,
    fetch_study_analytics_data,
)
from modules.cohort_tools import get_cohort_sql
from modules.graph_viz import build_graph
from streamlit_agraph import agraph
from streamlit_option_menu import option_menu
import folium
from streamlit_folium import st_folium

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

# --- App Configuration ---
st.set_page_config(
    page_title="Clinical Trial Inspector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Sidebar Width ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 200px;
        max-width: 250px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧬 Clinical Trial Inspector Agent")

# 1. Setup LLM & LlamaIndex Settings
# We use Google Gemini-2.5-Flash for fast and accurate responses.
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Please set GOOGLE_API_KEY in .env")
    st.stop()

# Ensure LlamaIndex settings (Embeddings, LLM) are applied on every run
setup_llama_index()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 2. Load LlamaIndex (Cached)
# The index is loaded once and cached to avoid reloading on every interaction.
index = load_index()


# 3. Define Agent (Cached)
@st.cache_resource
def get_agent():
    """Initializes and caches the LangChain agent."""
    tools = [
        search_trials,
        find_similar_studies,
        get_study_analytics,
        compare_studies,
        get_study_details,
        get_cohort_sql,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Clinical Trial Expert Assistant. "
                "Your goal is to help researchers and analysts understand clinical trial data. "
                "You have access to a local database of clinical trials (embedded from ClinicalTrials.gov). "
                "Use the available tools to search for studies, find similar studies, and generate analytics. "
                "When asked about 'trends', 'counts', 'how many', or 'most common', ALWAYS use the `get_study_analytics` tool. "
                "Do NOT use `search_trials` for counting questions like 'How many studies...'. "
                "When asked to 'find studies', 'search', or 'list', use `search_trials`. "
                "ALWAYS report the exact 'Total Potential Matches' and 'Relevant Matches' counts returned by the tool. "
                "Do not summarize '50 potential matches' as just '3 studies'. "
                "When asked to 'compare' multiple studies or answer complex multi-part questions, use `compare_studies`. "
                "If the user asks for a specific study by ID (e.g., NCT12345678), `search_trials` handles that automatically. "
                "However, if the user asks for specific **details**, **criteria**, **summary**, or **protocol** of a single study, "
                "you MUST use the `get_study_details` tool to fetch the full content. "
                "If the user asks to **generate SQL**, **build a cohort**, or **translate criteria to code** for a study, "
                "use the `get_cohort_sql` tool. "
                "When reporting 'similar studies', ALWAYS include the similarity score provided by the tool "
                "and DO NOT include the study that was used as the query (the reference study). "
                "Provide concise, evidence-based answers citing specific studies when possible.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


agent_executor = get_agent()

# --- Sidebar ---
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=50
    )
    st.title("Clinical Trial Agent")
    
    page = option_menu(
        "Main Menu",
        ["Chat Assistant", "Analytics Dashboard", "Knowledge Graph", "Study Map", "Raw Data"],
        icons=["chat-dots", "graph-up", "diagram-3", "map", "database"],
        menu_icon="cast",
        default_index=0,
    )


# --- Helper Functions ---
def generate_dashboard_analytics():
    """Callback to generate analytics and update session state."""
    # Map UI selection to tool arguments
    group_map = {
        "Phase": "phase",
        "Status": "status",
        "Sponsor": "sponsor",
        "Start Year": "start_year",
        "Intervention": "intervention",
        "Study Type": "study_type",
    }

    # Get values from session state
    # We use .get() to avoid KeyErrors if the widget hasn't initialized yet (though it should have)
    g_by = st.session_state.get("dash_group_by", "Sponsor")
    p_filter = st.session_state.get("dash_phase", "")
    s_filter = st.session_state.get("dash_sponsor", "")

    with st.spinner(f"Analyzing studies by {g_by}..."):
        # Call the tool directly
        result = get_study_analytics.invoke(
            {
                "query": "overall",
                "group_by": group_map.get(g_by, "sponsor"),
                "phase": p_filter if p_filter else None,
                "sponsor": s_filter if s_filter else None,
            }
        )

        # The tool sets session state 'inline_chart_data'
        if "inline_chart_data" in st.session_state:
            st.session_state["dashboard_data"] = st.session_state["inline_chart_data"]
        else:
            st.warning(result)


# --- PAGE 1: CHAT ---
if page == "Chat Assistant":
    st.header("💬 Chat Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Render chart if present in message history (persisted charts)
            if "chart_data" in message:
                chart_data = message["chart_data"]
                st.caption(chart_data["title"])
                chart = (
                    alt.Chart(pd.DataFrame(chart_data["data"]))
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            chart_data["x"], sort="-y", axis=alt.Axis(labelLimit=200)
                        ),
                        y=alt.Y(chart_data["y"], title="Count"),
                        tooltip=[chart_data["x"], chart_data["y"]],
                    )
                    .interactive()
                )
                st.altair_chart(chart, theme="streamlit", width="stretch")

    # Chat Input
    if prompt := st.chat_input("Ask about clinical trials..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing clinical trials..."):
                try:
                    # Clear previous inline chart data to avoid stale charts
                    if "inline_chart_data" in st.session_state:
                        del st.session_state["inline_chart_data"]

                    # Construct chat history for the agent context
                    chat_history = []
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))

                    # Invoke Agent
                    response = agent_executor.invoke(
                        {"input": prompt, "chat_history": chat_history}
                    )
                    output = response["output"]
                    st.markdown(output)

                    # Check for inline chart data (set by tools)
                    chart_data = None
                    if "inline_chart_data" in st.session_state:
                        chart_data = st.session_state["inline_chart_data"]
                        st.caption(chart_data["title"])
                        if chart_data["type"] == "bar":
                            # Use Altair for better charts
                            chart = (
                                alt.Chart(pd.DataFrame(chart_data["data"]))
                                .mark_bar()
                                .encode(
                                    x=alt.X(
                                        chart_data["x"],
                                        sort="-y",
                                        axis=alt.Axis(labelLimit=200),
                                    ),
                                    y=alt.Y(chart_data["y"], title="Count"),
                                    tooltip=[chart_data["x"], chart_data["y"]],
                                )
                                .interactive()
                            )
                            st.altair_chart(chart, theme="streamlit", width="stretch")

                        # Clean up session state
                        del st.session_state["inline_chart_data"]

                    # Save message with chart data if present
                    msg_obj = {"role": "assistant", "content": output}
                    if chart_data:
                        msg_obj["chart_data"] = chart_data
                    st.session_state.messages.append(msg_obj)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

# --- PAGE 2: ANALYTICS DASHBOARD ---
if page == "Analytics Dashboard":
    st.header("📊 Global Analytics")
    st.write(
        "Analyze trends across the entire clinical trial dataset (60,000+ studies)."
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Configuration")
        group_by = st.selectbox(
            "Group By",
            ["Phase", "Status", "Sponsor", "Start Year", "Intervention", "Study Type"],
            index=2,
            key="dash_group_by",
        )

        # Optional Filters
        st.markdown("---")
        st.markdown("**Filters (Optional)**")
        filter_phase = st.text_input("Phase (e.g., Phase 2)", key="dash_phase")
        filter_sponsor = st.text_input("Sponsor (e.g., Pfizer)", key="dash_sponsor")

        st.button(
            "Generate Analytics", type="primary", on_click=generate_dashboard_analytics
        )

    with col2:
        # Always render if data exists in session state
        if "dashboard_data" in st.session_state:
            c_data = st.session_state["dashboard_data"]
            st.subheader(c_data["title"])

            # Altair Chart Rendering
            if (
                c_data["x"] == "start_year" or group_by == "Start Year"
            ):  # Check both key and UI selection
                # Line chart for years
                chart = (
                    alt.Chart(pd.DataFrame(c_data["data"]))
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            c_data["x"], axis=alt.Axis(format="d"), title="Year"
                        ),  # 'd' for integer year
                        y=alt.Y(c_data["y"], title="Count"),
                        tooltip=[c_data["x"], c_data["y"]],
                    )
                    .interactive()
                )
            else:
                # Bar chart for others
                chart = (
                    alt.Chart(pd.DataFrame(c_data["data"]))
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            c_data["x"],
                            sort="-y",
                            axis=alt.Axis(labelLimit=200),
                        ),
                        y=alt.Y(c_data["y"], title="Count"),
                        tooltip=[c_data["x"], c_data["y"]],
                    )
                    .interactive()
                )

            st.altair_chart(chart, theme="streamlit", width="stretch")

            # Show raw table
            with st.expander("View Source Data"):
                st.dataframe(pd.DataFrame(c_data["data"]))

# --- PAGE 3: KNOWLEDGE GRAPH ---
if page == "Knowledge Graph":
    st.header("🕸️ Interactive Knowledge Graph")
    st.write("Visualize connections between Studies, Sponsors, and Conditions.")

    col_g1, col_g2 = st.columns([1, 3])

    with col_g1:
        st.subheader("Graph Settings")
        graph_query = st.text_input("Search Topic", value="Cancer")
        limit = st.slider("Max Nodes", 10, 100, 50)

        if st.button("Build Graph"):
            with st.spinner("Fetching data and building graph..."):
                # Use retriever to get relevant nodes
                retriever = index.as_retriever(similarity_top_k=limit)
                nodes = retriever.retrieve(graph_query)
                data = [n.metadata for n in nodes]

                # Build Graph
                g_nodes, g_edges, g_config = build_graph(data)

                st.session_state["graph_data"] = {
                    "nodes": g_nodes,
                    "edges": g_edges,
                    "config": g_config,
                }

    with col_g2:
        if "graph_data" in st.session_state:
            g_data = st.session_state["graph_data"]
            st.success(
                f"Graph built with {len(g_data['nodes'])} nodes and {len(g_data['edges'])} edges."
            )
            agraph(
                nodes=g_data["nodes"], edges=g_data["edges"], config=g_data["config"]
            )
        else:
            st.info("Enter a topic and click 'Build Graph' to visualize connections.")

# --- PAGE# --- Study Map Tab ---
elif page == "Study Map":
    st.header("🌍 Global Clinical Trial Map")
    st.markdown("Visualize the geographic distribution of clinical trials.")

    # Sidebar Filters for Map
    st.sidebar.markdown("### 🗺️ Map Filters")
    map_region = st.sidebar.radio("Region", ["World", "USA"], index=0)
    
    map_phase = st.sidebar.multiselect(
        "Phase", ["PHASE1", "PHASE2", "PHASE3", "PHASE4"], default=["PHASE2", "PHASE3"]
    )
    map_status = st.sidebar.selectbox(
        "Status", ["RECRUITING", "COMPLETED", "ACTIVE_NOT_RECRUITING"], index=0
    )
    map_sponsor = st.sidebar.text_input("Sponsor (Optional)", "")
    map_year = st.sidebar.number_input("Start Year (>=)", min_value=2000, value=2020)
    map_type = st.sidebar.selectbox(
        "Study Type", ["Interventional", "Observational", "All"], index=0
    )

    # Convert filters to arguments
    phase_str = ",".join(map_phase) if map_phase else None
    type_arg = map_type if map_type != "All" else None

    if st.button("Update Map"):
        with st.spinner("Aggregating geographic data..."):
            # Determine grouping based on Region
            group_by_field = "state" if map_region == "USA" else "country"
            
            # Call analytics logic directly
            summary = fetch_study_analytics_data(
                query="overall",
                group_by=group_by_field,
                phase=phase_str,
                status=map_status,
                sponsor=map_sponsor,
                start_year=map_year,
                study_type=type_arg,
            )
            
            # Retrieve data from session state
            chart_data = st.session_state.get("inline_chart_data", {})
            data_records = chart_data.get("data", [])
            
            if not data_records:
                st.warning("No data found for these filters.")
                st.session_state["map_data"] = None
                st.session_state["map_region"] = map_region # Store region too
            else:
                # Store in session state for persistence
                st.session_state["map_data"] = data_records
                st.session_state["map_region"] = map_region

    # Render Map (Outside Button Block)
    if st.session_state.get("map_data"):
        data_records = st.session_state["map_data"]
        region_mode = st.session_state.get("map_region", "World")
        df_map = pd.DataFrame(data_records)
        
        # Configure Map Center/Zoom
        if region_mode == "USA":
            m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
            coord_map = STATE_COORDINATES
        else:
            m = folium.Map(location=[20, 0], zoom_start=2)
            coord_map = COUNTRY_COORDINATES
        
        # Add CircleMarkers
        for _, row in df_map.iterrows():
            loc_name = row["category"]
            count = row["count"]
            
            # Clean name if needed (strip trailing parenthesis)
            loc_clean = loc_name.rstrip(")")
            coords = coord_map.get(loc_clean)
            
            if coords:
                folium.CircleMarker(
                    location=coords,
                    radius=min(max(count / 5, 3), 20),  # Adjust scale
                    popup=f"{loc_clean}: {count} trials",
                    color="blue" if region_mode == "USA" else "crimson",
                    fill=True,
                    fill_color="blue" if region_mode == "USA" else "crimson",
                ).add_to(m)
        
        st_folium(m, width=800, height=500)
        
        # Show data table
        st.subheader(f"{region_mode} Data")
        st.dataframe(df_map)

# --- PAGE 4: RAW DATA ---
if page == "Raw Data":
    st.header("📂 Raw Data Explorer")
    st.write("View and filter the underlying dataset.")

    # Load a sample or full dataset? Full might be slow.
    # We load a sample (top 100) to avoid performance issues.
    col_raw_1, col_raw_2 = st.columns([1, 1])

    with col_raw_1:
        if st.button("Load Sample Data (Top 100)"):
            with st.spinner("Fetching data..."):
                retriever = index.as_retriever(similarity_top_k=100)
                nodes = retriever.retrieve("clinical trial")
                data = [n.metadata for n in nodes]
                df_raw = pd.DataFrame(data)

                # Format Year to remove commas (e.g., 2,023 -> 2023)
                if "start_year" in df_raw.columns:
                    df_raw["start_year"] = (
                        pd.to_numeric(df_raw["start_year"], errors="coerce")
                        .astype("Int64")
                        .astype(str)
                        .str.replace(",", "")
                    )

                # Store in session state to persist the table
                st.session_state["sample_data"] = df_raw

    with col_raw_2:
        # Download Full Dataset Logic
        if st.button("Prepare Full Download (CSV)"):
            with st.spinner("Fetching all records from database..."):
                try:
                    # Access the underlying ChromaDB collection directly for speed
                    collection = index.vector_store._collection

                    # Fetch all metadata
                    all_data = collection.get(include=["metadatas"])

                    if all_data and all_data["metadatas"]:
                        df_full = pd.DataFrame(all_data["metadatas"])

                        # Convert to CSV
                        csv = df_full.to_csv(index=False).encode("utf-8")
                        st.session_state["full_csv"] = csv
                        st.success(f"Ready! Fetched {len(df_full)} records.")
                    else:
                        st.warning("No data found in database.")
                except Exception as e:
                    st.error(f"Error fetching data: {e}")

        if "full_csv" in st.session_state:
            st.download_button(
                label="⬇️ Download Full CSV",
                data=st.session_state["full_csv"],
                file_name="clinical_trials_full.csv",
                mime="text/csv",
            )

    # Display Sample Data Table (Full Width)
    if "sample_data" in st.session_state:
        st.markdown("### Sample Data (Top 100)")
        st.dataframe(
            st.session_state["sample_data"],
            column_config={
                "nct_id": "NCT ID",
                "title": "Study Title",
                "start_year": st.column_config.TextColumn(
                    "Start Year"
                ),  # Force text to avoid commas
                "url": st.column_config.LinkColumn("URL"),
            },
            width="stretch",
            hide_index=True,
        )
