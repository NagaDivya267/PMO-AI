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
st.set_page_config(page_title="Enterprise PMO AI", page_icon="🤖", layout="wide")

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
# NORMALIZE COLUMNS (FIX UNIQUE ISSUE)
# -------------------------------------------------------
def normalize_col(col):
    return col.strip().lower().replace(" ", "_").replace("-", "_")

normalized_dfs = {name: df.rename(columns={c: normalize_col(c) for c in df.columns}) 
                  for name, df in dataframes.items()}

# -------------------------------------------------------
# RELATIONSHIP DETECTION (FIXED)
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
                # FIX: Use set instead of .unique()
                vals1 = set(df1[col].dropna().astype(str).tolist())
                vals2 = set(df2[col].dropna().astype(str).tolist())
                
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
# PERSONA KPI LOGIC (FIXED MUTATIONS)
# -------------------------------------------------------
def get_persona_kpis(persona, dfs):
    kpis = {}
    
    try:
        if persona == "Director":
            for name, df in dfs.items():
                if "initiative" in name:
                    kpis["total_initiatives"] = len(df)
                    if "status" in df.columns:
                        kpis["at_risk"] = len(df[df["status"].str.lower().str.contains("risk|delay", na=False)])
                    
                    # FIX: No mutation of original dataframe
                    if "end_date" in df.columns:
                        temp_df = df.copy()
                        temp_df["end_date"] = pd.to_datetime(temp_df["end_date"], errors="coerce")
                        temp_df["is_delayed"] = temp_df["end_date"] < pd.Timestamp.now()
                        kpis["strategic_delays"] = temp_df["is_delayed"].sum()
                
                if "cost" in name:
                    if "actual_cost_usd" in df.columns:
                        kpis["total_spend"] = f"${df['actual_cost_usd'].sum():,.0f}"
                        if "planned_budget_usd" in df.columns:
                            variance = df['actual_cost_usd'].sum() - df['planned_budget_usd'].sum()
                            kpis["budget_variance"] = f"${variance:,.0f}"
        
        elif persona == "Project Manager":
            for name, df in dfs.items():
                if "epic" in name and "status" in df.columns:
                    kpis["open_epics"] = len(df[df["status"].str.lower() != "done"])
                if "feature" in name and "status" in df.columns:
                    kpis["blocked_features"] = len(df[df["status"].str.lower() == "blocked"])
                if "risk" in name and "status" in df.columns:
                    kpis["open_risks"] = len(df[df["status"].str.lower() != "closed"])
        
        elif persona == "CIO":
            for name, df in dfs.items():
                if "cost" in name and "actual_cost_usd" in df.columns:
                    kpis["total_investment"] = f"${df['actual_cost_usd'].sum():,.0f}"
                if "initiative" in name and "status" in df.columns:
                    kpis["strategic_delays"] = len(df[df["status"].str.lower() == "delayed"])
    
    except Exception as e:
        st.warning(f"KPI calculation error: {e}")
    
    return kpis

# -------------------------------------------------------
# STRICT TOOLS WITH DOCSTRINGS (FIXED)
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

# Cross-table analyzer
class AnalyzeInput(BaseModel):
    name: str = Field(..., description="Initiative name to analyze across tables")

def analyze_initiative_func(name: str) -> dict:
    """Analyze initiative across all related PMO datasets."""
    result = {"initiative": [], "risks": [], "costs": [], "features": []}
    
    # Find initiative
    for tname, df in normalized_dfs.items():
        if "initiative" in tname:
            for col in df.columns:
                if "name" in col:
                    matches = df[df[col].str.contains(name, case=False, na=False)]
                    if not matches.empty:
                        result["initiative"] = matches.to_dict(orient="records")
                        
                        # Auto-join with other tables using relationship graph
                        if tname in relationship_graph:
                            for related_table, rel_info in relationship_graph[tname]["related"].items():
                                if rel_info["join_candidates"]:
                                    join_key = rel_info["join_candidates"][0][0]  # Best join key
                                    related_df = normalized_dfs[related_table]
                                    
                                    if join_key in related_df.columns:
                                        for _, row in matches.iterrows():
                                            key_val = row[join_key]
                                            joined = related_df[related_df[join_key] == key_val]
                                            
                                            if "risk" in related_table:
                                                result["risks"].extend(joined.to_dict(orient="records"))
                                            elif "cost" in related_table:
                                                result["costs"].extend(joined.to_dict(orient="records"))
                                            elif "feature" in related_table:
                                                result["features"].extend(joined.to_dict(orient="records"))
                        break
            if result["initiative"]:
                break
    
    return result

analyze_initiative = StructuredTool.from_function(
    name="analyze_initiative",
    description="Deep analyze initiative across risk, cost, and feature data",
    func=analyze_initiative_func,
    args_schema=AnalyzeInput,
    strict=True
)

TOOLS = [get_initiative_status, get_risks_by_severity, search_raci, analyze_initiative]

# -------------------------------------------------------
# AGENT LLM (FIXED)
# -------------------------------------------------------
agent_llm = ChatOpenAI(
    model="gpt-4o-mini",  # More reliable than gpt-4.1
    temperature=0.4,
    api_key=st.secrets["OPENAI_API_KEY"]
).bind_tools(TOOLS)

# -------------------------------------------------------
# CHAT HISTORY (FIXED)
# -------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------------------------------
# AGENT FUNCTION (FIXED MULTI-TOOL + SAFE LOOKUP)
# -------------------------------------------------------
def run_agent(question: str, persona: str) -> str:
    """Run the PMO agent with multi-tool support."""
    
    # Lightweight relationship summary (not full graph)
    relationship_summary = {k: list(v["related"].keys()) for k, v in relationship_graph.items()}
    
    system_prompt = f"""
You are a PMO AI assistant for a {persona}.

Available data relationships: {relationship_summary}

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
            
            # SAFE TOOL LOOKUP (FIX)
            tool_obj = next((t for t in TOOLS if t.name == tool_name), None)
            if tool_obj is None:
                all_outputs.append({"tool": tool_name, "output": "Invalid tool requested"})
                continue
                
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
# UI WITH TABS (IMPROVED UX)
# -------------------------------------------------------
st.title("🤖 Enterprise PMO Assistant")

# Persona selector
persona = st.selectbox("Select Persona:", ["Director", "Project Manager", "CIO"])

# TABS FOR BETTER UX
tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Chat Assistant"])

with tab1:
    st.subheader(f"{persona} Dashboard")
    
    # KPI Dashboard (FIXED CRASH)
    kpis = get_persona_kpis(persona, normalized_dfs)
    
    if kpis:
        cols = st.columns(len(kpis))
        for i, (key, value) in enumerate(kpis.items()):
            cols[i].metric(key.replace("_", " ").title(), value)
    else:
        st.info("No KPI data available for this persona.")
    
    # Visualizations
    if normalized_dfs:
        # Risk heatmap
        for name, df in normalized_dfs.items():
            if "risk" in name and "impact" in df.columns and "likelihood" in df.columns:
                fig = px.density_heatmap(df, x="likelihood", y="impact", 
                                       title="Risk Heatmap", color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)
                break
        
        # Budget overview
        for name, df in normalized_dfs.items():
            if "cost" in name and "actual_cost_usd" in df.columns:
                if "planned_budget_usd" in df.columns:
                    budget_data = pd.DataFrame({
                        "Type": ["Planned", "Actual"],
                        "Amount": [df["planned_budget_usd"].sum(), df["actual_cost_usd"].sum()]
                    })
                    fig = px.bar(budget_data, x="Type", y="Amount", title="Budget vs Actual")
                    st.plotly_chart(fig, use_container_width=True)
                break

with tab2:
    st.subheader("💬 PMO Chat Assistant")
    
    # Chat history display
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.write(turn["assistant"])
    
    # CHAT INPUT (BETTER UX)
    question = st.chat_input("Ask about risks, costs, initiatives, or governance...")
    
    if question:
        with st.spinner("Analyzing..."):
            answer = run_agent(question, persona)
        
        # Save to history
        st.session_state.chat_history.append({
            "user": question,
            "assistant": answer
        })
        
        # Display new answer
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
        
        st.rerun()
