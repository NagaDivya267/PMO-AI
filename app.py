import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI


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
    return initiative_df, feature_df, epic_df, cost_df, risk_df

initiative_df, feature_df, epic_df, cost_df, risk_df = load_data()

# Step 6: Convert datasets into summary context
def create_context(initiative_df, cost_df, risk_df, feature_df, epic_df):
    """Create a summary context from all datasets"""
    
    # Initiative summary
    initiative_summary = initiative_df[['Issue key', 'Summary', 'Status', 'Priority']].head(10).to_string()
    
    # Cost summary
    cost_summary = cost_df.to_string()
    
    # Risk summary
    risk_summary = risk_df.to_string()
    
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

# Chat input
question = st.chat_input("Ask about project health, risks, costs, or delivery...")

# Use button question if set
if "question" in st.session_state and st.session_state.question:
    question = st.session_state.question
    st.session_state.question = None

if question:
    # Step 7: OpenAI integration
    
    # Note: Replace with your actual API key or use st.secrets
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    
    if not api_key:
        st.warning("⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY in secrets.toml or environment variables.")
        st.info("Add to your `.streamlit/secrets.toml`:")
        st.code("OPENAI_API_KEY = 'your-api-key-here'", lang="toml")
        
        # Show context for debugging
        st.subheader("📋 Project Data Context")
        st.text_area("Context sent to AI", context, height=300)
        st.subheader("❓ Your Question")
        st.write(question)
    else:
        # Initialize LangChain LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.7
        )

        with st.spinner("🤔 Analyzing project data..."):
            response = llm.invoke(
                f"""
You are a PMO AI assistant helping leaders understand risks, delays and cost overruns.

Provide concise actionable insights based on project data.

Focus on:
- Key risks
- Budget overruns
- Delivery delays
- Leadership recommendations

Project Data:
{context}

User Question:
{question}
"""
            )

            st.subheader("💬 AI Response")
            st.write(response.content)

            with st.expander("📊 Data Sources"):
                st.text(context[:2000] + "..." if len(context) > 2000 else context)

# Footer
st.markdown("---")
st.caption("🚀 PMO AI Assistant | Deploy on Streamlit Cloud")
