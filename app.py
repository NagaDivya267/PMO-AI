import os
import pandas as pd
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


# -------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Agentic PMO AI", page_icon="🤖", layout="wide")
st.title("🤖 Agentic PMO Assistant")


# -------------------------------------------------------
# LOAD DATAFILES
# -------------------------------------------------------
@st.cache_data
def load_data():
    dfs = {}
    for name in os.listdir("."):
        if name.endswith(".csv"):
            df_name = name.replace(".csv", "").replace(" ", "_").lower()
            dfs[df_name] = pd.read_csv(name)

    docs = PyPDFLoader("RACI.pdf").load()
    return dfs, docs

dataframes, docs = load_data()


# -------------------------------------------------------
# BUILD RACI VECTOR STORE
# -------------------------------------------------------
@st.cache_resource
def load_vectorstore(docs):
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vs = FAISS.from_documents(docs, embeddings)
    return vs.as_retriever()

retriever = load_vectorstore(docs)


# -------------------------------------------------------
# PERSONA CONFIGURATION
# -------------------------------------------------------
PERSONA_CONFIG = {
    "Director": {
        "detail_level": "summary",
        "focus_tables": ["initiative_jira_data", "revised_raw_cost_data"],
        "filters": {"Status": ["Delayed", "At Risk", "Blocked"]}
    },
    "Project Manager": {
        "detail_level": "detailed",
        "focus_tables": ["epic", "feature-initiative_jira", "revised_raw_risk_data"],
        "filters": {"Status": ["Open", "In Progress", "Blocked"]}
    },
    "CIO": {
        "detail_level": "executive",
        "focus_tables": ["initiative_jira_data", "revised_raw_cost_data"],
        "filters": {}
    }
}


def get_persona_context(persona, dfs):
    cfg = PERSONA_CONFIG[persona]
    results = []

    for table in cfg["focus_tables"]:
        if table not in dfs:
            continue

        df = dfs[table].copy()
        for col, values in cfg["filters"].items():
            if col in df.columns:
                df = df[df[col].isin(values)]
        results.append(f"{table}: {len(df)} relevant records")

    return "\n".join(results)


# -------------------------------------------------------
# TABLE ROUTER USING LLM
# -------------------------------------------------------
router_llm = ChatOpenAI(
    temperature=0,
    model="gpt-4.1",
    api_key=st.secrets["OPENAI_API_KEY"]
)

schema_summary = "\n".join(
    f"{name}: {list(df.columns)}"
    for name, df in dataframes.items()
)


def select_tables(question: str):
    msg = f"""
Question: "{question}"

Available datasets:
{schema_summary}

Return only a JSON list of the table names relevant.
"""
    resp = router_llm.invoke(msg)

    import json
    try:
        return json.loads(resp.content)
    except:
        return []


# -------------------------------------------------------
# PMO TOOLS
# -------------------------------------------------------
@tool
def get_initiative_status(name: str):
    df = dataframes["initiative_jira_data"]
    match = df[df["Initiative_Name"].str.contains(name, case=False, na=False)]
    return match.to_dict(orient="records")


@tool
def get_risks_by_severity(level: str):
    df = dataframes["revised_raw_risk_data"]
    m = df[df["Impact"].str.lower() == level.lower()]
    return m.to_dict(orient="records")


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


TOOLS = {
    "get_initiative_status": get_initiative_status,
    "get_risks_by_severity": get_risks_by_severity,
    "get_budget_variance": get_budget_variance,
    "search_raci": search_raci
}

# LLM with tool-calling enabled
agent_llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.4,
    api_key=st.secrets["OPENAI_API_KEY"]
).bind_tools(list(TOOLS.values()))


# -------------------------------------------------------
# PMO AGENT FUNCTION
# -------------------------------------------------------
def run_pmo_agent(question: str, persona: str):
    persona_context = get_persona_context(persona, dataframes)
    routed_tables = select_tables(question)
    raci_docs = retriever.invoke(question)
    raci_text = "\n".join([d.page_content for d in raci_docs])

    system_msg = f"""
You are an enterprise PMO AI with access to structured datasets and tools.

Persona: {persona}
Persona Context:
{persona_context}

Relevant Tables Suggested: {routed_tables}

RACI Context:
{raci_text}
"""

    # Invoke initial LLM to decide whether to call tools
    result = agent_llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=question)
    ])

    # Check for tool invocation
    tool_calls = result.additional_kwargs.get("tool_calls")
    if tool_calls:
        tc = tool_calls[0]
        tool_name = tc["function"]["name"]
        tool_args = tc["function"]["arguments"]

        # Execute tool
        tool_fn = TOOLS[tool_name]
        tool_output = tool_fn.run(tool_args)

        # Summarize result for user
        final = agent_llm.invoke([
            SystemMessage(content="Summarize the tool result clearly:"),
            HumanMessage(content=str(tool_output))
        ])

        return final.content

    # If no tool call, return direct response
    return result.content


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
persona = st.selectbox("Choose persona:", ["Director", "Project Manager", "CIO"])
question = st.text_input("Ask about risk, cost, blockers or initiatives:")

if question:
    with st.spinner("Analyzing PMO data..."):
        answer = run_pmo_agent(question, persona)
    st.subheader("🔍 AI Response")
    st.write(answer)
