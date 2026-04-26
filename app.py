from typing import Any

import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.agents import create_pandas_dataframe_agent
from typing import Any
# Page config
st.set_page_config(page_title="PMO AI Assistant", page_icon="🤖", layout="wide")

# Step 1-5: Load datasets
@st.cache_data
def load_data():
    initiative_df = pd.read_csv("Initiative Jira Data.csv")
    feature_df = pd.read_csv("Feature-Initiative Jira.csv")
    epic_df = pd.read_csv("Epic.csv")
    cost_df = pd.read_csv("Revised_Raw_Cost_Data.csv")
    risk_df = pd.read_csv("Revised_Raw_Risk_Data.csv")
    loader = PyPDFLoader("RACI.pdf")
    docs = loader.load()
    return initiative_df, feature_df, epic_df, cost_df, risk_df,docs
initiative_df, feature_df, epic_df, cost_df, risk_df, docs = load_data()
all_data = {
    "initiatives": initiative_df,
    "features": feature_df,
    "epics": epic_df,
    "costs": cost_df,
    "risks": risk_df
}

@st.cache_resource
def load_vector_store(docs):
    embeddings = OpenAIEmbeddings(
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )
    return vectorstore.as_retriever()
retriever = load_vector_store(docs)

@st.cache_resource
def load_llm():
    return ChatOpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        temperature=0.7
    )

# UI Header
st.title("🤖 PMO AI Assistant")
st.markdown("**Ask questions about project health, risks, costs, and delivery**")

persona = st.selectbox(
    "Select Stakeholder Persona",
    ["Director", "Project Manager", "CIO"]
)
if persona == "Director":
    
    col1, col2, col3, col4 = st.columns(4)

    # On track initiatives
    on_track_count = len(
        initiative_df[
            initiative_df["Status"].isin(["On Track", "Completed"])
        ]
    )

    # Budget variance
    budget_variance = (
        cost_df["Actual_Cost_USD"].sum() -
        cost_df["Planned_Budget_USD"].sum()
    )

    # Critical risks
    critical_risks = len(
        risk_df[
            (risk_df["Impact"].isin(["High", "Critical"])) &
            (risk_df["Status"] != "Closed")
        ]
    )

    col1.metric("Total Initiatives", len(initiative_df))
    col2.metric("On Track", on_track_count)
    col3.metric("Budget Variance", f"${budget_variance:,.0f}")
    col4.metric("Critical Risks", critical_risks)
elif persona == "Project Manager":
    
    col1, col2, col3 = st.columns(3)

    open_epics = len(
        epic_df[
            epic_df["Status"] != "Done"
        ]
    )

    open_risks = len(
        risk_df[
            risk_df["Status"] != "Closed"
        ]
    )

    blocked_features = len(
        feature_df[
            feature_df["Status"] == "Blocked"
        ]
    )

    col1.metric("Open Epics", open_epics)
    col2.metric("Open Risks", open_risks)
    col3.metric("Blocked Features", blocked_features)
elif persona == "CIO":
    col1, col2 = st.columns(2)

    investment_at_risk = cost_df[
        cost_df["Actual_Cost_USD"] >
        cost_df["Planned_Budget_USD"]
    ]["Actual_Cost_USD"].sum()

    strategic_delays = len(
        initiative_df[
            initiative_df["Status"] == "Delayed"
        ]
    )

    col1.metric(
        "Investment at Risk",
        f"${investment_at_risk:,.0f}"
    )
    col2.metric(
        "Strategic Delays",
        strategic_delays
    )   
@st.cache_resource
def load_pmo_agent():
    return create_pandas_dataframe_agent(
        load_llm(),
        [initiative_df, feature_df, epic_df, cost_df, risk_df],
        verbose=True,
        allow_dangerous_code=True
    )
pmo_agent = load_pmo_agent()
# Sidebar with data overview
with st.sidebar:
    st.header("📊 Project Data Overview")
    st.metric("Initiatives", len(initiative_df))
    st.metric("Features", len(feature_df))
    st.metric("Epics", len(epic_df))
    st.metric("Risks", len(risk_df))
    
    st.subheader("💰 Cost Summary")
    total_budget = cost_df['Planned_Budget_USD'].sum()
    total_actual = cost_df['Actual_Cost_USD'].sum()
    st.metric("Total Budget", f"${total_budget:,.0f}")
    st.metric("Total Actual", f"${total_actual:,.0f}")
    st.metric("Variance", f"${total_budget - total_actual:,.0f}")
    
    st.subheader("⚠️ Risk Summary")
    risk_status = risk_df['Status'].value_counts()
    for status, count in risk_status.items():
        st.write(f"- {status}: {count}")

# Example questions
st.subheader("💡 Example Questions")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Why is Hypercare high risk?"):
        st.session_state.question = "Why is Hypercare high risk?"
with col2:
    if st.button("Which initiatives are over budget?"):
        st.session_state.question = "Which initiatives are over budget?"
with col3:
    if st.button("What should leadership focus on today?"):
        st.session_state.question = "What should leadership focus on today?"
with col4:
    if st.button("Summarize project health"):
        st.session_state.question = "Summarize overall project health"

# Initialize chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
question = st.chat_input("Ask about project health, risks, costs, or delivery...")

# Use button question if set
if "question" in st.session_state and st.session_state.question:
    question = st.session_state.question
    st.session_state.question = None

if question:
    api_key = st.secrets.get("OPENAI_API_KEY", None)

    if not api_key:
        st.warning("⚠️ OpenAI API key not configured.")
        st.info("Add OPENAI_API_KEY in secrets.toml")

    else:
        llm = load_llm()

        with st.spinner("🤔 Analyzing project data..."):

            try:
                data_response = pmo_agent.invoke(
                    {"input": str(question)}
                )

                structured_output = data_response["output"]

            except Exception:
                structured_output = "Unable to analyze structured PMO data for this query."
                data_response = {"output": structured_output}

            rag_docs = retriever.invoke(str(question))

            rag_context = "\n".join(
                [doc.page_content for doc in rag_docs]
            )

            response = llm.invoke(
                f"""
You are an intelligent PMO AI Assistant.

Selected Persona: {persona}

Response Rules:

Director:
- Keep concise
- Highlight decisions
- Focus on portfolio impact
- show only top 3 priorities
- Prioritize strategic insights

Project Manager:
- Show blockers
- Show execution risks
- Provide mitigation actions
- Show blockers

CIO:
- Focus on strategic impact
- Budget exposure
- Governance risks

Chat History:
{st.session_state.chat_history}

RACI Context:
{rag_context}

Structured Data Insights:
{structured_output}

User Question:
{question}

Answer using actual project data.
"""
            )
        
         # MOVE THESE HERE
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.write(response.content)

        st.session_state.chat_history.append({
            "user": question,
            "assistant": response.content
        })

        with st.expander("📊 Data Analysis Used"):
            st.write(structured_output)
# Footer
st.markdown("---")
st.caption("🚀 PMO AI Assistant | Deploy on Streamlit Cloud")
