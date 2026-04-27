import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# -------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Agentic PMO AI", page_icon="🤖", layout="wide")
st.title("🤖 Enterprise PMO Assistant")

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
@st.cache_data
def load_data():
    dfs = {}
    for fname in os.listdir("."):
        if fname.endswith(".csv"):
            name = fname.replace(".csv", "").replace(" ", "_").lower()
            dfs[name] = pd.read_csv(fname)
    docs = PyPDFLoader("RACI.pdf").load()
    return dfs, docs

dataframes, docs = load_data()

# -------------------------------------------------------
# RACI VECTORSTORE
# -------------------------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vs = FAISS.from_documents(docs, embeddings)
    return vs.as_retriever()

retriever = load_vectorstore()

# -------------------------------------------------------
# NORMALIZE COLUMNS
# -------------------------------------------------------
def normalize_col(col):
    return col.strip().lower().replace(" ", "_").replace("-", "_")

normalized_dfs = {name: df.rename(columns={c: normalize_col(c) for c in df.columns}) 
                  for name, df in dataframes.items()}

# -------------------------------------------------------
# RELATIONSHIP DETECTION (FIXED .unique() ISSUE)
# -------------------------------------------------------
def detect_relationships(dfs):
    relationships = {}
    table_names = list(dfs.keys())
    
    for t1 in table_names:
        relationships[t1] = {"related": {}}
        df1 = dfs[t1]
        
        for t2 in table_names:
            if t1 == t2:
                continue
            df2 = dfs[t2]
            
            shared_cols = list(set(df1.columns).intersection(set(df2.columns)))
            join_candidates = []
            
            for col in shared_cols:
                # FIX: Use .unique() properly
                vals1 = set(df1[col].dropna().astype(str))
                vals2 = set(df2[col].dropna().astype(str))
                
                if len(vals1) == 0 or len(vals2) == 0:
                    continue
                    
                overlap = len(vals1.intersection(vals2))
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

# -------------------------------------------------------
# PERSONA KPIS
# -------------------------------------------------------
def get_persona_kpis(persona, dfs):
    kpis = {}
    
    if persona == "Director":
        # Portfolio metrics
        for name, df in dfs.items():
            if "initiative" in name and "status" in df.columns:
                kpis["total_initiatives"] = len(df)
                kpis["at_risk"] = len(df[df["status"].str.lower().isin(["delayed", "at risk"])])
            if "cost" in name and "actual_cost_usd" in df.columns:
                kpis["budget_variance"] = f"${(df['actual_cost_usd'].sum() - df.get('planned_budget_usd', df['actual_cost_usd']).sum()):,.0f}"
    
    elif persona == "Project Manager":
        # Execution metrics
        for name, df in dfs.items():
            if "epic" in name and "status" in df.columns:
                kpis["open_epics"] = len(df[df["status"].str.lower() != "done"])
            if "feature" in name and "status" in df.columns:
                kpis["blocked_features"] = len(df[df["status"].str.lower() == "blocked"])
    
    elif persona == "CIO":
        # Strategic metrics
        for name, df in dfs.items():
            if "cost" in name and "actual_cost_usd" in df.columns:
                kpis["total_investment"] = f"${df['actual_cost_usd'].sum():,.0f}"
            if "initiative" in name and "status" in df.columns:
                kpis["strategic_delays"] = len(df[df["status"].str.lower() == "delayed"])
    
    return kpis

# -------------------------------------------------------
# STRICT TOOLS WITH DOCSTRINGS
# -------------------------------------------------------

class InitiativeInput(BaseModel):
    name: str = Field(..., description="Initiative name to search")

def get_initiative_status_func(name: str) -> list:
    """Find initiatives matching the given name."""
    results = []
    for tname, df in normalized_dfs.items():
        if "initiative" in tname:
            for col in df.columns:
                if "name" in col:
                    matches = df[df[col].str.contains(name, case=False, na=False)]
                    if not matches.empty:
                        results.extend(matches.to_dict(orient="records"))
    return results

get_initiative_status = StructuredTool.from_function(
    name="get_initiative_status",
    description="Look up initiative details by name",
    func=get_initiative_status_func,
    args_schema=InitiativeInput,
    strict=True
)

class RiskInput(BaseModel):
    level: str = Field(..., description="Risk level: critical, high, medium, low")

def get_risks_func(level: str) -> list:
    """Get risks by severity level."""
    results = []
    for tname, df in normalized_dfs.items():
        if "risk" in tname and "impact" in df.columns:
            matches = df[df["impact"].str.lower() == level.lower()]
            if not matches.empty:
                results.extend(matches.to_dict(orient="records"))
    return results

get_risks_by_severity = StructuredTool.from_function(
    name="get_risks_by_severity",
    description="Get risks filtered by severity level",
    func=get_risks_func,
    args_schema=RiskInput,
    strict=True
)

class RACIInput(BaseModel):
    query: str = Field(..., description="Query for RACI document")

def search_raci_func(query: str) -> str:
    """Search RACI document for responsibilities."""
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs[:3]])

search_raci = StructuredTool.from_function(
    name="search_raci",
    description="Search RACI document for role responsibilities",
    func=search_raci_func,
    args_schema=RACIInput,
    strict=True
)

TOOLS = [get_initiative_status, get_risks_by_severity, search_raci]

# -------------------------------------------------------
# AGENT LLM
# -------------------------------------------------------
agent_llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.4,
    api_key=st.secrets["OPENAI_API_KEY"]
).bind_tools(TOOLS)

# -------------------------------------------------------
# AGENT FUNCTION
# -------------------------------------------------------
def run_agent(question: str, persona: str) -> str:
    """Run the PMO agent with multi-tool support."""
    
    system_prompt = f"""
You are a PMO AI assistant for a {persona}.

Available data relationships:
{relationship_graph}

Use tools when data lookup is needed. Answer clearly and concisely.
"""
    
    result = agent_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])
    
    tool_calls = result.additional_kwargs.get("tool_calls", [])
    
    if tool_calls:
        all_outputs = []
        for call in tool_calls:
            tool_name = call["function"]["name"]
            args = call["function"]["arguments"]
            
            import json
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}
            
            tool_obj = next(t for t in TOOLS if t.name == tool_name)
            try:
                output = tool_obj.invoke(args)
                all_outputs.append({"tool": tool_name, "output": output})
            except Exception as e:
                all_outputs.append({"tool": tool_name, "output": f"Error: {e}"})
        
        summary = agent_llm.invoke([
            SystemMessage(content="Summarize these results clearly:"),
            HumanMessage(content=str(all_outputs))
        ])
        return summary.content
    
    return result.content

# -------------------------------------------------------
# UI
# -------------------------------------------------------
persona = st.selectbox("Select Persona:", ["Director", "Project Manager", "CIO"])

# KPI Dashboard
kpis = get_persona_kpis(persona, normalized_dfs)
if kpis:
    cols = st.columns(len(kpis))
    for i, (key, value) in enumerate(kpis.items()):
        cols[i].metric(key.replace("_", " ").title(), value)

# Chat Interface
question = st.text_input("Ask about risks, costs, initiatives, or governance:")

if question:
    with st.spinner("Analyzing..."):
        answer = run_agent(question, persona)
    
    st.subheader("📌 Response")
    st.write(answer)
