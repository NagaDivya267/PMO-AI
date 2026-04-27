import os
import pandas as pd
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage


# -------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Agentic PMO Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Agentic PMO Assistant")


# -------------------------------------------------------
# LOAD ALL PROJECT DATA
# -------------------------------------------------------
@st.cache_data
def load_data():
    dfs = {}
    for fname in os.listdir("."):
        if fname.endswith(".csv"):
            dfname = fname.replace(".csv", "").replace(" ", "_").lower()
            dfs[dfname] = pd.read_csv(fname)

    docs = PyPDFLoader("RACI.pdf").load()
    return dfs, docs

dataframes, docs = load_data()


# -------------------------------------------------------
# LOAD RACI VECTOR STORE
# -------------------------------------------------------
@st.cache_resource
def load_vectorstore(documents):
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    store = FAISS.from_documents(documents, embeddings)
    return store.as_retriever()

retriever = load_vectorstore(docs)


# -------------------------------------------------------
# PERSONA CONFIG
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


def get_persona_context(persona: str, dfs: dict) -> str:
    cfg = PERSONA_CONFIG[persona]
    result = []

    for table in cfg["focus_tables"]:
        if table not in dfs:
            continue

        df = dfs[table].copy()
        for col, values in cfg["filters"].items():
            if col in df.columns:
                df = df[df[col].isin(values)]

        result.append(f"{table}: {len(df)} relevant items")

    return "\n".join(result)


# -------------------------------------------------------
# LLM FOR ROUTING DATASET SELECTION
# -------------------------------------------------------
router_llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    api_key=st.secrets["OPENAI_API_KEY"]
)

schema_summary = "\n".join(
    f"{name}: {list(df.columns)}"
    for name, df in dataframes.items()
)


def select_tables(question: str):
    msg = f"""
Question: {question}

Here are available datasets and columns:
{schema_summary}

Return a JSON list of the table names required to answer the question.
"""

    response = router_llm.invoke(msg)

    import json
    try:
        return json.loads(response.content)
    except:
        return []


# -------------------------------------------------------
# PMO TOOLS (STRICTLY TYPED + SERIALIZABLE)
# -------------------------------------------------------
@tool
def get_initiative_status(name: str) -> list:
    """Return matching initiatives from initiative_jira_data."""
    df = dataframes.get("initiative_jira_data", pd.DataFrame())
    if "Initiative_Name" not in df.columns:
        return []
    match = df[df["Initiative_Name"].str.contains(name, case=False, na=False)]
    return match.to_dict(orient="records")


@tool
def get_risks_by_severity(level: str) -> list:
    """Return risks filtered by severity."""
    df = dataframes.get("revised_raw_risk_data", pd.DataFrame())
    if "Impact" not in df.columns:
        return []
    match = df[df["Impact"].str.lower() == level.lower()]
    return match.to_dict(orient="records")


@tool
def get_budget_variance(initiative_id: str | None = None) -> dict:
    """Return total variance and list of over-budget items."""
    df = dataframes.get("revised_raw_cost_data", pd.DataFrame())

    if initiative_id and "Initiative_ID" in df.columns:
        df = df[df["Initiative_ID"] == initiative_id]

    if "Actual_Cost_USD" not in df.columns or "Planned_Budget_USD" not in df.columns:
        return {"variance": 0.0, "over_budget_items": []}

    actual = float(df["Actual_Cost_USD"].sum())
    planned = float(df["Planned_Budget_USD"].sum())
    variance = actual - planned

    over = df[df["Actual_Cost_USD"] > df["Planned_Budget_USD"]]
    over_items = over.to_dict(orient="records")

    return {"variance": variance, "over_budget_items": over_items}


@tool
def search_raci(query: str) -> str:
    """Search the RACI PDF for responsibility insights."""
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs[:3]])


TOOLS = {
    "get_initiative_status": get_initiative_status,
    "get_risks_by_severity": get_risks_by_severity,
    "get_budget_variance": get_budget_variance,
    "search_raci": search_raci,
}


# -------------------------------------------------------
# MAIN AGENT LLM (TOOL CALLING ENABLED)
# -------------------------------------------------------
agent_llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.4,
    api_key=st.secrets["OPENAI_API_KEY"],
    response_format="text"
).bind_tools(list(TOOLS.values()))


# -------------------------------------------------------
# MAIN AGENT PIPELINE (PATCHED + GUARANTEED OUTPUT)
# -------------------------------------------------------
def run_pmo_agent(question: str, persona: str) -> str:
    persona_ctx = get_persona_context(persona, dataframes)
    routed_tables = select_tables(question)

    raci_docs = retriever.invoke(question)
    raci_context = "\n".join([doc.page_content for doc in raci_docs])

    system_prompt = f"""
You are an enterprise PMO AI assistant.

Persona: {persona}
Persona Context:
{persona_ctx}

Likely Relevant Tables:
{routed_tables}

RACI Context:
{raci_context}

Rules:
- Call tools when data lookup is needed.
- Otherwise answer using available context.
- Always respond in a clear, readable summary.
"""

    # Request initial reasoning
    result = agent_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    tool_calls = result.additional_kwargs.get("tool_calls")

    # -------------------------------------------------------
    # If the model wants to call a tool
    # -------------------------------------------------------
    if tool_calls:
        call = tool_calls[0]
        tool_name = call["function"]["name"]
        raw_args = call["function"]["arguments"]

        # Convert args safely
        import json
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except:
                args = {"name": raw_args}
        else:
            args = raw_args

        tool_fn = TOOLS[tool_name]
        tool_output = tool_fn.invoke(args)

        # Summarize the tool output
        summarized = agent_llm.invoke([
            SystemMessage(content="Summarize the PMO data clearly:"),
            HumanMessage(content=str(tool_output))
        ])
        return summarized.content

    # -------------------------------------------------------
    # Otherwise return direct text answer
    # -------------------------------------------------------
    if isinstance(result.content, list):
        return "\n".join([m.get("text", "") for m in result.content])

    return result.content


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
persona = st.selectbox("Select Persona:", ["Director", "Project Manager", "CIO"])
question = st.text_input("Ask a PMO question (risks, blockers, costs, initiatives)…")

if question:
    with st.spinner("Analyzing PMO data…"):
        answer = run_pmo_agent(question, persona)

    st.subheader("📌 Response")
    st.write(answer)
