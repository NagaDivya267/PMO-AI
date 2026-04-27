import os
import pandas as pd
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# -------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Agentic PMO AI", page_icon="🤖", layout="wide")
st.title("🤖 Agentic PMO Assistant (Strict Tools Version)")


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
# VECTOR STORE (RACI)
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
        "filters": {"Status": ["Delayed", "At Risk", "Blocked"]},
    },
    "Project Manager": {
        "detail_level": "detailed",
        "focus_tables": [
            "epic",
            "feature-initiative_jira",
            "revised_raw_risk_data",
        ],
        "filters": {"Status": ["Open", "In Progress", "Blocked"]},
    },
    "CIO": {
        "detail_level": "executive",
        "focus_tables": ["initiative_jira_data", "revised_raw_cost_data"],
        "filters": {},
    },
}


def get_persona_context(persona: str, dfs: dict) -> str:
    """Summaries of the most relevant tables per persona."""
    cfg = PERSONA_CONFIG[persona]
    result = []

    for table in cfg["focus_tables"]:
        if table not in dfs:
            continue

        df = dfs[table].copy()
        for col, values in cfg["filters"].items():
            if col in df.columns:
                df = df[df[col].isin(values)]

        result.append(f"{table}: {len(df)} records relevant")

    return "\n".join(result)


# -------------------------------------------------------
# LLM FOR ROUTING DATASET SELECTION
# -------------------------------------------------------
router_llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    api_key=st.secrets["OPENAI_API_KEY"],
)

schema_summary = "\n".join(
    f"{name}: {list(df.columns)}" for name, df in dataframes.items()
)


def select_tables(question: str):
    msg = f"""
Question: {question}

Available datasets and columns:
{schema_summary}

Return ONLY a JSON list of table names relevant to the question.
"""

    response = router_llm.invoke(msg)
    import json

    try:
        return json.loads(response.content)
    except:
        return []


# -------------------------------------------------------
# STRICT PMO TOOLS (StructuredTool + Pydantic)
# -------------------------------------------------------

# --- 1. Initiative Status Tool ---
class InitiativeStatusInput(BaseModel):
    name: str = Field(..., description="The initiative name to search for")


def get_initiative_status_func(name: str) -> list:
    df = dataframes.get("initiative_jira_data", pd.DataFrame())
    if "Initiative_Name" not in df.columns:
        return []
    match = df[df["Initiative_Name"].str.contains(name, case=False, na=False)]
    return match.to_dict(orient="records")


get_initiative_status = StructuredTool.from_function(
    name="get_initiative_status",
    description="Look up initiatives by partial or full name.",
    func=get_initiative_status_func,
    args_schema=InitiativeStatusInput,
    strict=True,
)


# --- 2. Risk Severity Tool ---
class RiskSeverityInput(BaseModel):
    level: str = Field(..., description="Risk level: Critical, High, Medium, Low")


def get_risks_by_severity_func(level: str) -> list:
    df = dataframes.get("revised_raw_risk_data", pd.DataFrame())
    if "Impact" not in df.columns:
        return []
    match = df[df["Impact"].str.lower() == level.lower()]
    return match.to_dict(orient="records")


get_risks_by_severity = StructuredTool.from_function(
    name="get_risks_by_severity",
    description="Return all risks matching a severity level.",
    func=get_risks_by_severity_func,
    args_schema=RiskSeverityInput,
    strict=True,
)


# --- 3. Budget Variance Tool ---
class BudgetVarianceInput(BaseModel):
    initiative_id: str | None = Field(
        None, description="Optional initiative ID to filter cost data"
    )


def get_budget_variance_func(initiative_id: str | None) -> dict:
    df = dataframes.get("revised_raw_cost_data", pd.DataFrame())

    if initiative_id and "Initiative_ID" in df.columns:
        df = df[df["Initiative_ID"] == initiative_id]

    if (
        "Actual_Cost_USD" not in df.columns
        or "Planned_Budget_USD" not in df.columns
    ):
        return {"variance": 0.0, "over_budget_items": []}

    actual = float(df["Actual_Cost_USD"].sum())
    planned = float(df["Planned_Budget_USD"].sum())
    variance = actual - planned

    over = df[df["Actual_Cost_USD"] > df["Planned_Budget_USD"]]
    return {
        "variance": variance,
        "over_budget_items": over.to_dict(orient="records"),
    }


get_budget_variance = StructuredTool.from_function(
    name="get_budget_variance",
    description="Calculate budget variance and list all over‑budget items.",
    func=get_budget_variance_func,
    args_schema=BudgetVarianceInput,
    strict=True,
)


# --- 4. RACI Search Tool ---
class RACIQueryInput(BaseModel):
    query: str = Field(..., description="Query text for RACI search")


def search_raci_func(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs[:3]])


search_raci = StructuredTool.from_function(
    name="search_raci",
    description="Search role or responsibility information from the RACI document.",
    func=search_raci_func,
    args_schema=RACIQueryInput,
    strict=True,
)


# ALL TOOLS
TOOLS = [
    get_initiative_status,
    get_risks_by_severity,
    get_budget_variance,
    search_raci,
]


# -------------------------------------------------------
# MAIN AGENT LLM (STRICT TOOL CALLING)
# -------------------------------------------------------
agent_llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.4,
    api_key=st.secrets["OPENAI_API_KEY"],
).bind_tools(TOOLS)


# -------------------------------------------------------
# MAIN AGENT PIPELINE
# -------------------------------------------------------
def run_pmo_agent(question: str, persona: str) -> str:
    persona_ctx = get_persona_context(persona, dataframes)
    routed = select_tables(question)

    raci_docs = retriever.invoke(question)
    raci_context = "\n".join([doc.page_content for doc in raci_docs])

    system_prompt = f"""
You are an enterprise PMO AI system.

Persona: {persona}

Persona Context:
{persona_ctx}

Likely Relevant Tables:
{routed}

RACI Context:
{raci_context}

Rules:
- If the answer requires data lookup, call a tool.
- If tool output is returned, summarize it clearly.
- Keep responses concise and business‑focused.
"""

    # Step 1: Ask LLM what to do
    result = agent_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    # Step 2: If tool call exists, execute
    tool_calls = result.additional_kwargs.get("tool_calls")

    if tool_calls:
        call = tool_calls[0]
        tool_name = call["function"]["name"]
        raw_args = call["function"]["arguments"]

        import json
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except:
                args = {"name": raw_args}
        else:
            args = raw_args

        tool_fn = next(t for t in TOOLS if t.name == tool_name)
        tool_output = tool_fn.invoke(args)

        # Step 3: Summarize tool results
        summary = agent_llm.invoke([
            SystemMessage(content="Summarize the PMO insights clearly:"),
            HumanMessage(content=str(tool_output))
        ])
        return summary.content

    # Step 4: If no tool used, return text
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
