import os
import pandas as pd
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate


# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(page_title="PMO Agentic AI", page_icon="🤖", layout="wide")
st.title("🤖 Agentic PMO Assistant")
st.write("Ask about risks, costs, blockers, initiatives, or responsibilities.")


# -------------------------------------------------------
# Load Data
# -------------------------------------------------------
@st.cache_data
def load_data():
    dfs = {}
    for file in os.listdir("."):
        if file.endswith(".csv"):
            name = file.replace(".csv", "").replace(" ", "_").lower()
            dfs[name] = pd.read_csv(file)
    docs = PyPDFLoader("RACI.pdf").load()
    return dfs, docs

dataframes, docs = load_data()


# -------------------------------------------------------
# Schema Summary for Routing
# -------------------------------------------------------
def build_schema_summary(dfs):
    summary = []
    for name, df in dfs.items():
        summary.append(f"{name}: {list(df.columns)}")
    return "\n".join(summary)

schema_summary = build_schema_summary(dataframes)


# -------------------------------------------------------
# LLMs
# -------------------------------------------------------
llm_router = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], temperature=0)
llm_agent = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], temperature=0.4)


# -------------------------------------------------------
# Vector Store (RACI)
# -------------------------------------------------------
@st.cache_resource
def build_raci_vectorstore(docs):
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vs = FAISS.from_documents(docs, embeddings)
    return vs.as_retriever()

retriever = build_raci_vectorstore(docs)


# -------------------------------------------------------
# Persona Configuration
# -------------------------------------------------------
PERSONA_CONFIG = {
    "Director": {
        "detail_level": "summary",
        "filters": {"Status": ["Delayed", "Blocked", "At Risk"]},
        "focus_tables": ["initiative_jira_data", "revised_raw_cost_data"]
    },
    "Project Manager": {
        "detail_level": "detailed",
        "filters": {"Status": ["Open", "Blocked", "In Progress"]},
        "focus_tables": ["epic", "feature-initiative_jira", "revised_raw_risk_data"]
    },
    "CIO": {
        "detail_level": "executive",
        "filters": {},
        "focus_tables": ["initiative_jira_data", "revised_raw_cost_data"]
    }
}

def get_persona_context(persona, dfs):
    cfg = PERSONA_CONFIG[persona]
    summary = []

    for table in cfg["focus_tables"]:
        if table not in dfs:
            continue
        df = dfs[table].copy()

        for col, values in cfg["filters"].items():
            if col in df.columns:
                df = df[df[col].isin(values)]

        summary.append(f"{table}: {len(df)} records")

    return "\n".join(summary)


# -------------------------------------------------------
# Table Router
# -------------------------------------------------------
def select_relevant_tables(question: str):
    msg = f"""
Question: "{question}"

Available tables and schemas:
{schema_summary}

Return ONLY a JSON list of table names necessary to answer this.
"""
    response = llm_router.invoke(msg)

    import json
    try:
        return json.loads(response.content)
    except:
        return []


# -------------------------------------------------------
# PMO Tools
# -------------------------------------------------------
@tool
def get_initiative_status(initiative_name: str):
    df = dataframes["initiative_jira_data"]
    match = df[df["Initiative_Name"].str.contains(initiative_name, case=False, na=False)]
    return match.to_dict(orient="records")


@tool
def get_risks_by_severity(severity: str):
    df = dataframes["revised_raw_risk_data"]
    match = df[df["Impact"].str.lower() == severity.lower()]
    return match.to_dict(orient="records")


@tool
def get_budget_variance(initiative_id: str = None):
    df = dataframes["revised_raw_cost_data"]
    if initiative_id:
        df = df[df["Initiative_ID"] == initiative_id]

    variance = df["Actual_Cost_USD"].sum() - df["Planned_Budget_USD"].sum()
    over = df[df["Actual_Cost_USD"] > df["Planned_Budget_USD"]]

    return {
        "variance": variance,
        "over_budget_items": over.to_dict(orient="records")
    }


@tool
def search_raci(query: str):
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs[:3]])


tools = [get_initiative_status, get_risks_by_severity, get_budget_variance, search_raci]


# -------------------------------------------------------
# Memory
# -------------------------------------------------------
memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], temperature=0),
    max_token_limit=1000,
    return_messages=True
)


# -------------------------------------------------------
# Agent Prompt
# -------------------------------------------------------
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an enterprise-grade Agentic PMO AI.

You MUST:
- Decide which tools to call.
- Use data instead of guessing.
- Consider the persona.
- Use the routed tables and RACI context.

Persona Context:
{persona_context}

Conversation History:
{history}
"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])


# -------------------------------------------------------
# Agent Executor
# -------------------------------------------------------
agent = create_openai_tools_agent(llm_agent, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)


# -------------------------------------------------------
# Unified Query Function
# -------------------------------------------------------
def run_pmo_agent(question: str, persona: str):
    persona_context = get_persona_context(persona, dataframes)
    routed_tables = select_relevant_tables(question)
    raci_docs = retriever.invoke(question)
    raci_text = "\n".join([d.page_content for d in raci_docs])

    full_input = f"""
User asked: {question}

Persona: {persona}
Persona Context:
{persona_context}

Suggested Relevant Tables:
{routed_tables}

RACI Context:
{raci_text}
"""

    result = agent_executor.invoke({"input": full_input})
    return result["output"]


# -------------------------------------------------------
# UI
# -------------------------------------------------------
persona = st.selectbox("Select Persona", ["Director", "Project Manager", "CIO"])
question = st.text_input("Ask a PMO question (risk, blockers, cost, etc.)")

if question:
    with st.spinner("Analyzing project portfolio..."):
        answer = run_pmo_agent(question, persona)

    st.subheader("📌 Answer")
    st.write(answer)
