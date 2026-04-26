import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

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
# Step 6: Convert datasets into summary context
def create_context(initiative_df, cost_df, risk_df, feature_df, epic_df):
    """Create a summary context from all datasets"""
    
    # Initiative summary
    initiative_summary = initiative_df[['Issue key', 'Summary', 'Status', 'Priority']].head(10).to_string()
    
    # Cost summary
    cost_summary = cost_df.head(20).to_string()
    
    # Risk summary
    risk_summary = risk_df.head(20).to_string()
    
    # Feature count per initiative
    feature_counts = feature_df['Parent key'].value_counts().to_string()
    
    # Epic summary
    epic_summary = epic_df[['Issue key', 'Summary', 'Status']].head(10).to_string()
    
    context = f"""=== INITIATIVES ===
{initiative_summary}

=== COSTS (Budget vs Actual) ===
{cost_summary}

=== RISKS ===
{risk_summary}

=== FEATURES PER INITIATIVE ===
{feature_counts}

=== EPICS ===
{epic_summary}"""
    
    return context

context = create_context(initiative_df, cost_df, risk_df, feature_df, epic_df)

# UI Header
st.title("🤖 PMO AI Assistant")
st.markdown("**Ask questions about project health, risks, costs, and delivery**")

persona = st.selectbox(
    "Select Stakeholder Persona",
    ["Director", "Project Manager", "CIO", "Engineering Manager"]
)

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

            rag_docs = retriever.invoke(str(question))

            rag_context = "\n".join(
                [doc.page_content for doc in rag_docs]
            )

            response = llm.invoke(
                f"""
You are a PMO AI assistant helping leaders understand risks, delays, costs, and stakeholder communication.

Use previous chat history to maintain conversation continuity.

Chat History:
{st.session_state.chat_history}

{persona}

RACI Context:
{rag_context}

Project Data:
{context}

User Question:
{question}

Provide:
1. Root cause
2. Business impact
3. Recommended action
4. Tailor tone based on persona
"""
            )

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                st.write(response.content)

            st.session_state.chat_history.append({
            st.session_state.chat_history.append({
                "user": question,
                "assistant": response.content
            })

            with st.expander("📊 Data Sources"):
                st.text(context[:2000] + "..." if len(context) > 2000 else context)

# Footer
st.markdown("---")
st.caption("🚀 PMO AI Assistant | Deploy on Streamlit Cloud")
