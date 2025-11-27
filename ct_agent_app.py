import streamlit as st
import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="Clinical Trial Inspector", layout="wide")
st.title("🧬 Clinical Trial Inspector Agent")

# 1. Setup LLM (Gemini) & Embeddings (Local)
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Google API Key not found.")
    st.stop()

# LLM for Thinking
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Embeddings for Searching (Must match ingestion)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load Vector DB
persist_dir = "./ct_gov_local_db" # Make sure this matches the new folder name
if not os.path.exists(persist_dir):
    st.error("Database not found! Run 'python ingest_gemini.py' first.")
    st.stop()

vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# 3. Define Metadata for Self-Querying
metadata_field_info = [
    AttributeInfo(name="year", description="The year the study started (Integer).", type="integer"),
    AttributeInfo(name="phase", description="The phase (String): PHASE1, PHASE2, PHASE3, PHASE4, NA.", type="string"),
    AttributeInfo(name="sponsor", description="The sponsor organization (String).", type="string"),
    AttributeInfo(name="title", description="The title of the study (String).", type="string"),
    AttributeInfo(name="nct_id", description="The NCT ID of the study (String).", type="string"),
]
document_content_description = "Clinical trial eligibility criteria"

# 4. Initialize Self-Query Retriever
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True 
)

# 5. Create Tool & Agent
tool = create_retriever_tool(
    retriever,
    "search_trials",
    "Searches for clinical trials. Returns criteria and IDs."
)
tools = [tool]

system_prompt = """
You are an expert Clinical Research Assistant.
1. Use 'search_trials' to find real data. 
2. If asked to COMPARE, retrieve data for all mentioned studies, then synthesize.
3. Always cite the NCT ID.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if user_input := st.chat_input("Ex: Find Phase 3 Pfizer studies started after 2022"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                response = agent_executor.invoke({
                    "input": user_input, 
                    "chat_history": st.session_state.messages
                })
                output = response["output"]
                st.markdown(output)
                st.session_state.messages.append(AIMessage(content=output))
            except Exception as e:
                st.error(f"Error: {e}")