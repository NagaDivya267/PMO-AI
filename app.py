import os
import pandas as pd
import numpy as np
import streamlit as st

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import altair as alt
import numpy as np
from langchain_openai import ChatOpenAI
from datetime import datetime


# -------------------------------------------------------------------
# STREAMLIT LAYOUT SETUP
# -------------------------------------------------------------------
st.set_page_config(page_title="Agentic PMO AI", page_icon="🤖", layout="wide")


# -------------------------------------------------------------------
# LOAD ALL DATAFRAMES & RACI PDF
# -------------------------------------------------------------------
@st.cache_data
def load_all_data():
    dfs = {}
    for fname in os.listdir("."):
        if fname.endswith(".csv"):
            key = fname.replace(".csv", "").replace(" ", "_").lower()
            try:
                dfs[key] = pd.read_csv(fname)
            except:
                dfs[key] = pd.read_csv(fname, encoding="latin-1")

    # Load RACI
    docs = PyPDFLoader("RACI.pdf").load()
    return dfs, docs


dataframes, docs = load_all_data()


# -------------------------------------------------------------------
# VECTORSTORE FOR RACI SEARCH
# -------------------------------------------------------------------
@st.cache_resource
def load_raci_retriever():
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vs = FAISS.from_documents(docs, embeddings)
    return vs.as_retriever()


raci_retriever = load_raci_retriever()


# -------------------------------------------------------------------
# SCHEMA NORMALIZATION
# -------------------------------------------------------------------
def normalize_col(col: str):
    return (
        col.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def normalize_dataframe(df: pd.DataFrame):
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]
    return df


normalized_dfs = {name: normalize_dataframe(df) for name, df in dataframes.items()}


# -------------------------------------------------------------------
# RELATIONSHIP ENGINE
# -------------------------------------------------------------------
def detect_relationships(dfs: dict):
    """
    Automatically infer relationships between tables using:
    - identical column names
    - fuzzy key similarity
    - overlapping values
    Produces a relationship graph.
    """
    relationships = {}
    table_names = list(dfs.keys())

    for t1 in table_names:
        relationships[t1] = {"related": {}}
        df1 = dfs[t1]

        for t2 in table_names:
            if t1 == t2:
                continue
            df2 = dfs[t2]

            # 1) Shared columns
            shared_cols = list(set(df1.columns).intersection(set(df2.columns)))

            # 2) Overlap scoring for join keys
            join_candidates = []
            for col in shared_cols:
                # Values must be non-null strings or IDs
                vals1 = df1[col].dropna().astype(str).unique()
                vals2 = df2[col].dropna().astype(str).unique()

                if len(vals1) == 0 or len(vals2) == 0:
                    continue

                overlap = len(set(vals1).intersection(vals2))
                if overlap > 0:
                    join_candidates.append((col, overlap))

            if join_candidates:
                join_candidates = sorted(join_candidates, key=lambda x: -x[1])
                relationships[t1]["related"][t2] = {
                    "shared_columns": shared_cols,
                    "join_candidates": join_candidates,
                }

    return relationships


relationship_graph = detect_relationships(normalized_dfs)
# -------------------------------------------------------------------
# PART 2 — STRICT Pydantic Tools + Auto‑Join Analyzer
# -------------------------------------------------------------------
# Utility: Auto‑detect join keys between two tables
def find_join_keys(table1: str, table2: str) -> list:
    rel = relationship_graph.get(table1, {}).get("related", {}).get(table2)
    if not rel:
        return []
    # Return candidate join keys sorted by overlap score
    return [col for col, overlap in rel["join_candidates"]]


# Utility: Join two tables using best join key
def auto_join(df1: pd.DataFrame, df2: pd.DataFrame, key: str):
    try:
        return df1.merge(df2, on=key, how="left")
    except:
        return df1  # fail-safe, never crash agent


# -------------------------------------------------------------------
# TOOL 1: Initiative Status Lookup
# -------------------------------------------------------------------
class InitiativeStatusInput(BaseModel):
    name: str = Field(..., description="Name or partial name of an initiative")


def initiative_status_func(name: str) -> list:
    results = []
    for tname, df in normalized_dfs.items():
        if "initiative" in tname:
            if "initiative_name" in df.columns:
                match = df[df["initiative_name"].str.contains(name, case=False, na=False)]
                if not match.empty:
                    results.extend(match.to_dict(orient="records"))
    return results


initiative_status = StructuredTool.from_function(
    name="initiative_status",
    description="Lookup initiative details by matching partial name.",
    func=initiative_status_func,
    args_schema=InitiativeStatusInput,
    strict=True,
)


# -------------------------------------------------------------------
# TOOL 2: Risk Lookup by Severity
# -------------------------------------------------------------------
class RiskSeverityInput(BaseModel):
    level: str = Field(..., description="Risk level filter: critical, high, etc.")


def risk_severity_func(level: str) -> list:
    results = []
    for tname, df in normalized_dfs.items():
        if "risk" in tname:
            if "impact" in df.columns:
                match = df[df["impact"].str.lower() == level.lower()]
                if not match.empty:
                    results.extend(match.to_dict(orient="records"))
    return results


risk_by_severity = StructuredTool.from_function(
    name="risk_by_severity",
    description="Return all risks matching a severity level.",
    func=risk_severity_func,
    args_schema=RiskSeverityInput,
    strict=True,
)


# -------------------------------------------------------------------
# TOOL 3: Budget Variance Lookup
# -------------------------------------------------------------------
class BudgetVarianceInput(BaseModel):
    key: str | None = Field(None, description="Optional initiative key (Issue key)")


def budget_variance_func(key: str | None) -> dict:
    costs = None
    for tname, df in normalized_dfs.items():
        if "cost" in tname:
            costs = df.copy()

    if costs is None:
        return {"variance": 0.0, "items": []}

    if key:
        # auto-detect columns resembling issue_key
        candidates = [c for c in costs.columns if "key" in c]
        if candidates:
            col = candidates[0]
            costs = costs[costs[col] == key]

    if "actual_cost_usd" not in costs.columns or "planned_budget_usd" not in costs.columns:
        return {"variance": 0.0, "items": []}

    actual = float(costs["actual_cost_usd"].sum())
    planned = float(costs["planned_budget_usd"].sum())
    variance = actual - planned

    return {
        "variance": variance,
        "items": costs.to_dict(orient="records")
    }


budget_variance = StructuredTool.from_function(
    name="budget_variance",
    description="Calculate budget variance for an initiative or for the full portfolio.",
    func=budget_variance_func,
    args_schema=BudgetVarianceInput,
    strict=True,
)


# -------------------------------------------------------------------
# TOOL 4: RACI Search Tool
# -------------------------------------------------------------------
class RACIQueryInput(BaseModel):
    text: str = Field(..., description="Text to search inside the RACI document")


def raci_query_func(text: str) -> str:
    docs = raci_retriever.invoke(text)
    return "\n".join([d.page_content for d in docs[:3]])


raci_search = StructuredTool.from_function(
    name="raci_search",
    description="Search the RACI document for stakeholder or responsibility guidance.",
    func=raci_query_func,
    args_schema=RACIQueryInput,
    strict=True,
)


# -------------------------------------------------------------------
# TOOL 5: Cross-Table Initiative Analyzer (AUTO-JOIN)
# -------------------------------------------------------------------
class InitiativeAnalyzeInput(BaseModel):
    name: str = Field(..., description="Initiative name to analyze")


def initiative_analyze_func(name: str) -> dict:
    """
    Automatically:
    - Find all initiative entries across tables
    - Detect join keys via relationship_graph
    - Join with risk, cost, features, epics, etc.
    """
    output = {
        "initiative_rows": [],
        "risks": [],
        "costs": [],
        "features": [],
        "epics": [],
        "joins_used": []
    }

    # Find initiative row
    initiative_df = None
    for tname, df in normalized_dfs.items():
        if "initiative" in tname and "initiative_name" in df.columns:
            match = df[df["initiative_name"].str.contains(name, case=False, na=False)]
            if not match.empty:
                initiative_df = match
                output["initiative_rows"] = match.to_dict(orient="records")
                break

    if initiative_df is None:
        return output  # initiative not found anywhere

    # Normalize the first row's keys for join detection
    found_table = tname

    # For each related table, auto-join via relationship_graph
    related_tables = relationship_graph[found_table]["related"]

    for t2, rel_info in related_tables.items():
        join_keys = [key for key, score in rel_info["join_candidates"]]
        if not join_keys:
            continue

        df_target = normalized_dfs[t2]
        join_key = join_keys[0]  # best match
        if join_key in initiative_df.columns and join_key in df_target.columns:
            merged = initiative_df.merge(df_target, on=join_key, how="left")

            if "risk" in t2:
                output["risks"] = merged.to_dict(orient="records")
            if "cost" in t2:
                output["costs"] = merged.to_dict(orient="records")
            if "feature" in t2:
                output["features"] = merged.to_dict(orient="records")
            if "epic" in t2:
                output["epics"] = merged.to_dict(orient="records")

            output["joins_used"].append({t2: join_key})

    return output


initiative_analyze = StructuredTool.from_function(
    name="initiative_analyze",
    description="Deep-dive analyze an initiative across all related PMO tables automatically.",
    func=initiative_analyze_func,
    args_schema=InitiativeAnalyzeInput,
    strict=True,
)


# -------------------------------------------------------------------
# TOOL REGISTRY
# -------------------------------------------------------------------
TOOLS = [
    initiative_status,
    risk_by_severity,
    budget_variance,
    raci_search,
    initiative_analyze,
]

# -------------------------------------------------------------------
# PART 3 — Persona Engine, KPI Extractor, Dashboard Visuals
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# KPI Helper Functions
# -------------------------------------------------------------------
def numeric_safe(series):
    """Convert a series to numeric, coercing errors."""
    return pd.to_numeric(series, errors="coerce")


def count_if(df, col, values):
    if col not in df.columns:
        return 0
    return df[df[col].isin(values)].shape[0]


# -------------------------------------------------------------------
# Persona KPI Logic
# -------------------------------------------------------------------
def compute_director_kpis(dfs):
    """High-level portfolio KPIs for Directors."""
    kpis = {}

    # Total initiatives
    for name, df in dfs.items():
        if "initiative" in name and "initiative_name" in df.columns:
            kpis["total_initiatives"] = df.shape[0]
            break

    # At-risk initiatives (using risk table)
    risky = 0
    for name, df in dfs.items():
        if "risk" in name and "impact" in df.columns:
            risky = count_if(df, "impact", ["critical", "high"])
            break
    kpis["high_risk_items"] = risky

    # Over-budget items
    for name, df in dfs.items():
        if "cost" in name:
            if "planned_budget_usd" in df.columns and "actual_cost_usd" in df.columns:
                over = df[df["actual_cost_usd"] > df["planned_budget_usd"]]
                kpis["over_budget_items"] = over.shape[0]
                kpis["budget_variance"] = float(
                    numeric_safe(df["actual_cost_usd"]).sum() - 
                    numeric_safe(df["planned_budget_usd"]).sum()
                )
                break

    # Strategic delays (date-based)
    delays = 0
    for name, df in dfs.items():
        if "initiative" in name and "end_date" in df.columns:
            # end_date < today but not finished
            df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
            df["is_delayed"] = df["end_date"] < pd.Timestamp.now()
            delays = df["is_delayed"].sum()
            break
    kpis["strategic_delays"] = delays

    return kpis


def compute_pm_kpis(dfs):
    """Project Manager metrics — execution-level detail."""
    kpis = {}

    # Epics
    for name, df in dfs.items():
        if "epic" in name and "status" in df.columns:
            kpis["open_epics"] = df[df["status"].str.lower() != "done"].shape[0]

    # Blocked features
    for name, df in dfs.items():
        if "feature" in name and "status" in df.columns:
            kpis["blocked_features"] = df[df["status"].str.lower() == "blocked"].shape[0]

    # Open risks
    for name, df in dfs.items():
        if "risk" in name and "status" in df.columns:
            kpis["open_risks"] = df[df["status"].str.lower() != "closed"].shape[0]

    return kpis


def compute_cio_kpis(dfs):
    """CIO-level strategic KPIs."""
    kpis = {}

    # Investment and financials
    for name, df in dfs.items():
        if "cost" in name:
            if "planned_budget_usd" in df.columns and "actual_cost_usd" in df.columns:
                actual = numeric_safe(df["actual_cost_usd"]).sum()
                planned = numeric_safe(df["planned_budget_usd"]).sum()
                variance = actual - planned

                kpis["total_budget"] = float(planned)
                kpis["total_actual"] = float(actual)
                kpis["variance"] = float(variance)

                over = df[df["actual_cost_usd"] > df["planned_budget_usd"]]
                kpis["investment_at_risk"] = numeric_safe(over["actual_cost_usd"]).sum()

    # Strategic delays
    for name, df in dfs.items():
        if "initiative" in name and "status" in df.columns:
            delays = df[df["status"].str.lower().isin(["delayed"])].shape[0]
            kpis["strategic_delays"] = delays

    return kpis


# -------------------------------------------------------------------
# KPI DISPATCHER
# -------------------------------------------------------------------
def get_kpis_for_persona(persona, dfs):
    if persona == "Director":
        return compute_director_kpis(dfs)
    elif persona == "Project Manager":
        return compute_pm_kpis(dfs)
    elif persona == "CIO":
        return compute_cio_kpis(dfs)
    return {}


# -------------------------------------------------------------------
# Visual Charts
# -------------------------------------------------------------------
def plot_timeline(df, title="Project Timeline"):
    if "start_date" not in df.columns or "end_date" not in df.columns:
        return None

    df = df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    df = df.dropna(subset=["start_date", "end_date"])
    if df.empty:
        return None

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="start_date:T",
            x2="end_date:T",
            y="initiative_name:N",
            color=alt.Color("status:N", legend=None),
        )
        .properties(title=title, height=300)
    )
    return chart


def plot_risk_heatmap(df, title="Risk Heatmap"):
    if "impact" not in df.columns or "likelihood" not in df.columns:
        return None

    df = df.copy()
    df["impact"] = df["impact"].astype(str)
    df["likelihood"] = df["likelihood"].astype(str)

    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("likelihood:N"),
            y=alt.Y("impact:N"),
            color=alt.Color("count()", scale=alt.Scale(scheme="reds")),
            tooltip=["impact", "likelihood", "count()"],
        )
        .properties(height=300, title=title)
    )
    return chart


def plot_budget_variance(df, title="Budget Variance"):
    if "planned_budget_usd" not in df.columns or "actual_cost_usd" not in df.columns:
        return None

    df = df.copy()
    df["planned_budget_usd"] = numeric_safe(df["planned_budget_usd"])
    df["actual_cost_usd"] = numeric_safe(df["actual_cost_usd"])

    melted = df.melt(
        value_vars=["planned_budget_usd", "actual_cost_usd"],
        var_name="type",
        value_name="amount",
    )

    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x="type:N",
            y="amount:Q",
            color="type:N",
            tooltip=["type", "amount"],
        )
        .properties(height=300, title=title)
    )
    return chart
# -------------------------------------------------------------------
# PART 4 — Multi‑Tool Agent Pipeline + Chat History + Reasoning Layer
# -------------------------------------------------------------------
# -----------------------------------------------------------
# Chat History Initialization
# -----------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def append_history(user_msg, assistant_msg):
    st.session_state.chat_history.append(
        {
            "user": user_msg,
            "assistant": assistant_msg,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def format_history():
    """Convert previous turns into a compact summary for prompting."""
    formatted = []
    for turn in st.session_state.chat_history[-6:]:  # limit clutter
        formatted.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")
    return "\n\n".join(formatted)


# -----------------------------------------------------------
# Conditional RACI Retrieval
# -----------------------------------------------------------
def maybe_get_raci(question: str):
    """Only query RACI when question is communication/governance related."""
    keywords = ["communicat", "stakeholder", "escalat", "email", "responsib"]
    if any(k in question.lower() for k in keywords):
        docs = raci_retriever.invoke(question)
        return "\n".join([d.page_content for d in docs[:3]])
    return "RACI not required."


# -----------------------------------------------------------
# Persona‑Aware System Prompt
# -----------------------------------------------------------
def build_system_prompt(persona: str, question: str):
    history = format_history()
    raci_context = maybe_get_raci(question)

    return f"""
You are an enterprise‑grade Agentic PMO AI system.

Persona: {persona}

Persona Expectations:
- Director: Top‑level portfolio summary, risks, cost variance, strategic delays. 
- Project Manager: Blockers, epics, feature execution, upcoming deadlines, risks.
- CIO: Strategic exposure, investment risk, governance, overall portfolio performance.

You MUST:
- Use available PMO tools when data lookup or cross‑analysis is required.
- Use multiple tools if necessary — do not restrict to only one.
- Use the relationship graph to determine how tables are connected.
- Join tables using the relationships, not assumptions.
- If RACI_context is “RACI not required”, ignore all RACI logic.
- Always produce a clear, concise business‑level answer.
- If tools return structured data, summarize the insights.
- If multiple tools are called, combine outputs into one narrative.

Relationship Graph (auto‑generated):
{relationship_graph}

RACI Context:
{raci_context}

Chat History:
{history}

Now answer the user’s question intelligently.
"""


# -----------------------------------------------------------
# LLM Setup (Strict Tool Calling)
# -----------------------------------------------------------
agent_llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.4,
    api_key=st.secrets["OPENAI_API_KEY"]
).bind_tools(TOOLS)


# -----------------------------------------------------------
# Multi‑Tool Execution Engine
# -----------------------------------------------------------
def run_agent(question: str, persona: str):
    system_msg = build_system_prompt(persona, question)

    # Step 1 — Ask the LLM how to proceed
    result = agent_llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question}
    ])

    tool_calls = result.additional_kwargs.get("tool_calls", [])

    # ----------------------------------------
    # Case A — No tool calls, return plain text
    # ----------------------------------------
    if not tool_calls:
        answer = result.content
        append_history(question, answer)
        return answer

    # ----------------------------------------
    # Case B — MULTIPLE tool calls (your fix)
    # ----------------------------------------
    all_outputs = []
    import json

    for call in tool_calls:
        tool_name = call["function"]["name"]
        raw_args = call["function"]["arguments"]

        # Parse arguments safely
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except:
                args = {}
        else:
            args = raw_args

        # Fetch the tool object
        tool_obj = next(t for t in TOOLS if t.name == tool_name)

        # Execute tool with safety
        try:
            output = tool_obj.invoke(args)
        except Exception as e:
            output = {"error": str(e)}

        all_outputs.append({"tool": tool_name, "output": output})

    # ----------------------------------------
    # Step 2 — Combine tool outputs into one answer
    # ----------------------------------------
    final_answer = agent_llm.invoke([
        {"role": "system", "content": "You are a PMO summarization engine."},
        {"role": "user", "content":
            f"Combine these tool outputs into a single business answer:\n\n{all_outputs}"}
    ]).content

    append_history(question, final_answer)
    return final_answer
# -------------------------------------------------------------------
# PART 5 — Final Streamlit UI (Persona KPIs + Visuals + Chatbot)
# -------------------------------------------------------------------

# Title
st.title("🤖 Agentic PMO Assistant")


# -----------------------------------------------------------
# Persona Selector
# -----------------------------------------------------------
persona = st.selectbox(
    "Select Persona:",
    ["Director", "Project Manager", "CIO"],
    index=0
)


# -----------------------------------------------------------
# KPI Dashboard
# -----------------------------------------------------------
st.markdown("## 📊 Key Performance Indicators")

kpis = get_kpis_for_persona(persona, normalized_dfs)

kpi_cols = st.columns(len(kpis))

for idx, (label, value) in enumerate(kpis.items()):
    kpi_cols[idx].metric(label.replace("_", " ").title(), f"{value:,}")


# -----------------------------------------------------------
# Visuals Section
# -----------------------------------------------------------
st.markdown("## 📈 Portfolio Visualizations")

# Timeline chart
for name, df in normalized_dfs.items():
    if "initiative" in name and "start_date" in df.columns and "end_date" in df.columns:
        chart = plot_timeline(df)
        if chart:
            st.altair_chart(chart, use_container_width=True)
        break

# Risk heatmap
for name, df in normalized_dfs.items():
    if "risk" in name and "impact" in df.columns and "likelihood" in df.columns:
        heatmap = plot_risk_heatmap(df)
        if heatmap:
            st.altair_chart(heatmap, use_container_width=True)
        break

# Budget variance plot
for name, df in normalized_dfs.items():
    if "cost" in name and "planned_budget_usd" in df.columns:
        budget_plot = plot_budget_variance(df)
        if budget_plot:
            st.altair_chart(budget_plot, use_container_width=True)
        break


# -----------------------------------------------------------
# Chat Interface
# -----------------------------------------------------------
st.markdown("## 💬 PMO Chatbot")

question = st.text_input(
    "Ask a PMO question (risks, blockers, costs, initiatives, governance, strategy)..."
)

if question:
    with st.spinner("Analyzing…"):
        answer = run_agent(question, persona)

    st.markdown("### 🧠 Answer")
    st.write(answer)

    # Show recent history
    with st.expander("📝 Conversation History"):
        for turn in st.session_state.chat_history[-8:]:
            st.markdown(f"**You:** {turn['user']}")
            st.markdown(f"**Assistant:** {turn['assistant']}")
            st.markdown("---")


