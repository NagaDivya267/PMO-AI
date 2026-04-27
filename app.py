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

Return only a JSON list of table names that are relevant.
"""

    response = router_llm.invoke(msg)

    import json
    try:
        return json.loads(response.content)
    except:
        return []


# -------------------------------------------------------
# PMO TOOLS (STRICTLY TYPED — FIXED)
# -------------------------------------------------------
@tool
def get_initiative_status(name: str) -> list:
    """Return initiative status by name."""
    df = dataframes["initiative_jira_data"]
    if "Initiative_Name" not in df.columns:
        return []
    match = df[df["Initiative_Name"].str.contains(name, case=False, na=False)]
    return match.to_dict(orient="records")


@tool
def get_risks_by_severity(level: str) -> list:
    """Return risks filtered by severity."""
    df = dataframes["revised_raw_risk_data"]
    if "Impact" not in df.columns:
        return []
    match = df[df["Impact"].str.lower() == level.lower()]
    return match.to_dict(orient="records")


@tool
def get_budget_variance(initiative_id: str | None = None) -> dict:
    """Return cost variance and over-budget items."""
    df = dataframes["revised_raw_cost_data"]

    if initiative_id:
        if "Initiative_ID" in df.columns:
            df = df[df["Initiative_ID"] == initiative_id]

    actual = float(df["Actual_Cost_USD"].sum())
    planned = float(df["Planned_Budget_USD"].sum())
    variance = actual - planned

    if "Actual_Cost_USD" in df.columns and "Planned_Budget_USD" in df.columns:
        over = df[df["Actual_Cost_USD"] > df["Planned_Budget_USD"]]
        over_list = over.to_dict(orient="records")
    else:
        over_list = []

    return {"variance": variance, "over_budget_items": over_list}


@tool
def search_raci(query: str) -> str:
    """Search role responsibilities in RACI PDF."""
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs[:3]])


TOOLS = {
    "get_initiative_status": get_initiative_status,
    "get_risks_by_severity": get_risks_by_severity,
    "get_budget_variance": get_budget_variance,
    "search_raci": search_raci
}


# -------------------------------------------------------
# MAIN AGENT LLM (TOOL CALLING)
# -------------------------------------------------------
agent_llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.4,
    api_key=st.secrets["OPENAI_API_KEY"]
).bind_tools(list(TOOLS.values()))


# -------------------------------------------------------
# AGENT LOGIC
# -------------------------------------------------------
def run_pmo_agent(question: str, persona: str):
    persona_context = get_persona_context(persona, dataframes)
    routed = select_tables(question)
    raci_docs = retriever.invoke(question)
    raci_text = "\n".join([d.page_content for d in raci_docs])

    system_prompt = f"""
You are an enterprise PMO AI assistant.

Persona: {persona}
Persona Context:
{persona_context}

Relevant Tables Suggested:
{routed}

RACI Context:
{raci_text}

Use tools to answer questions when data lookup is required.
"""

    # Invoke LLM (may generate a tool call)
    result = agent_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    tool_calls = result.additional_kwargs.get("tool_calls")

    if tool_calls:
        call = tool_calls[0]
        tool_name = call["function"]["name"]
        tool_args = call["function"]["arguments"]

        tool_fn = TOOLS[tool_name]
        output = tool_fn.run(tool_args)

        final = agent_llm.invoke([
            SystemMessage(content="Explain the tool result clearly:"),
            HumanMessage(content=str(output))
        ])
        return final.content

    return result.content


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
persona = st.selectbox("Select Persona:", ["Director", "Project Manager", "CIO"])
question = st.text_input("Ask a PMO question (risks, blockers, costs, initiatives)...")

if question:
    with st.spinner("Analyzing PMO data..."):
        answer = run_pmo_agent(question, persona)
    st.subheader("📌 Response")
    st.write(answer)
