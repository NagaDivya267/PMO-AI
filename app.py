import os
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.agents import create_pandas_dataframe_agent

# Page config
st.set_page_config(page_title="PMO AI Assistant", page_icon="🤖", layout="wide")

@st.cache_data
def load_data():
    dataframes = {}

    for file in os.listdir("."):
        if file.endswith(".csv"):
            df_name = file.replace(".csv", "").replace(" ", "_").lower()
            dataframes[df_name] = pd.read_csv(file)

    loader = PyPDFLoader("RACI.pdf")
    docs = loader.load()

    return dataframes, docs


dataframes, docs = load_data()

def get_schema_info(dataframes):
    schema_info = {}

    for name, df in dataframes.items():
        schema_info[name] = {
            "columns": list(df.columns),
            "rows": len(df)
        }

    return schema_info


schema_info = get_schema_info(dataframes)


def detect_relationships(dataframes):
    relationships = []

    names = list(dataframes.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            df1 = dataframes[names[i]]
            df2 = dataframes[names[j]]

            common_cols = list(
                set(df1.columns).intersection(set(df2.columns))
            )

            if common_cols:
                relationships.append(
                    f"{names[i]} ↔ {names[j]} via {common_cols}"
                )

    if not relationships:
        relationships.append(
        "No direct relationships detected. Use column similarity and business context."
    )
    return relationships

relationships = detect_relationships(dataframes)
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
        dataframes["initiative_jira_data"][
            dataframes["initiative_jira_data"]["Status"].isin(["On Track", "Completed"])
        ]
    )

    # Budget variance
    budget_variance = (
        dataframes["revised_raw_cost_data"]["Actual_Cost_USD"].sum() -
        dataframes["revised_raw_cost_data"]["Planned_Budget_USD"].sum()
    )

    # Critical risks
    critical_risks = len(
        dataframes["revised_raw_risk_data"][
            dataframes["revised_raw_risk_data"]["Impact"].isin(["High", "Critical"]) &
            dataframes["revised_raw_risk_data"]["Status"] != "Closed"
        ]
    )

    col1.metric("Total Initiatives", len(dataframes["initiative_jira_data"]))
    col2.metric("On Track", on_track_count)
    col3.metric("Budget Variance", f"${budget_variance:,.0f}")
    col4.metric("Critical Risks", critical_risks)
elif persona == "Project Manager":

    col1, col2, col3 = st.columns(3)

    open_epics = len(
        dataframes["epic"][
            dataframes["epic"]["Status"] != "Done"
        ]
    )

    open_risks = len(
        dataframes["revised_raw_risk_data"][
            dataframes["revised_raw_risk_data"]["Status"] != "Closed"
        ]
    )

    blocked_features = len(
        dataframes["feature-initiative_jira"][
            dataframes["feature-initiative_jira"]["Status"] == "Blocked"
        ]
    )

    col1.metric("Open Epics", open_epics)
    col2.metric("Open Risks", open_risks)
    col3.metric("Blocked Features", blocked_features)
    
elif persona == "CIO":

    col1, col2 = st.columns(2)

    investment_at_risk = dataframes["revised_raw_cost_data"][
        dataframes["revised_raw_cost_data"]["Actual_Cost_USD"] >
        dataframes["revised_raw_cost_data"]["Planned_Budget_USD"]
    ]["Actual_Cost_USD"].sum()

    strategic_delays = len(
        dataframes["initiative_jira_data"][
            dataframes["initiative_jira_data"]["Status"] == "Delayed"
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
    agents = {}

    for name, df in dataframes.items():
        agents[name] = create_pandas_dataframe_agent(
            load_llm(),
            df,
            verbose=True,
            allow_dangerous_code=True
        )

    return agents

pmo_agents = load_pmo_agent()
# Sidebar with data overview
with st.sidebar:
    st.sidebar.write("Loaded Tables:")
    st.sidebar.write(list(dataframes.keys()))
    st.metric("Initiatives", len(dataframes["initiative_jira_data"]))
    st.metric("Features", len(dataframes["feature-initiative_jira"]))
    st.metric("Epics", len(dataframes["epic"]))
    st.metric("Risks", len(dataframes["revised_raw_risk_data"]))
    st.subheader("💰 Cost Summary")
    total_budget = dataframes["revised_raw_cost_data"]['Planned_Budget_USD'].sum()
    total_actual = dataframes["revised_raw_cost_data"]['Actual_Cost_USD'].sum()
    st.metric("Total Budget", f"${total_budget:,.0f}")
    st.metric("Total Actual", f"${total_actual:,.0f}")
    st.metric("Variance", f"${total_budget - total_actual:,.0f}")
    
    st.subheader("⚠️ Risk Summary")
    risk_status = dataframes["revised_raw_risk_data"]['Status'].value_counts()
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
                # Dynamically select best table based on schema similarity
                question_lower = str(question).lower()

                best_table = None
                best_score = 0

                for table_name, df in dataframes.items():

                    score = 0

                    for col in df.columns:
                        if col.lower() in question_lower:
                            score += 1

                    if score > best_score:
                        best_score = score
                        best_table = table_name

                if best_table:
                    data_response = pmo_agents[best_table].invoke(
                        {"input": str(question)}
                    )
                    structured_output = data_response["output"]

                else:
                    structured_output = "No relevant dataset identified."

            except Exception as e:
                structured_output = f"Agent Error: {str(e)}"
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
- Show only top 3 priorities

Project Manager:
- Show blockers
- Show execution risks
- Provide mitigation actions

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

Available Tables:
{list(dataframes.keys())}

Schema Information:
{schema_info}

Detected Relationships:
{relationships}

Use relationships between datasets dynamically.
Join tables when common columns exist.

User Question:
{question}

Answer using actual project data.
"""
            )

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
