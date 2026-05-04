import os
import pandas as pd
import streamlit as st
import plotly.express as px

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Enterprise PMO Assistant",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    dfs = {}

    for file in os.listdir("."):
        if file.endswith(".csv"):
            table_name = file.replace(".csv", "").replace(" ", "_").lower()

            try:
                dfs[table_name] = pd.read_csv(file)
            except Exception:
                pass

    all_docs = []

    pdf_files = [
        "RACI.pdf",
        "Operational_Decision_RACI_PMO.pdf"
    ]

    for pdf in pdf_files:
        if os.path.exists(pdf):
            try:
                loaded_docs = PyPDFLoader(pdf).load()

                # tag document source
                for doc in loaded_docs:
                    doc.metadata["source_type"] = pdf

                all_docs.extend(loaded_docs)

            except Exception:
                pass

    return dfs, all_docs


dataframes, docs = load_data()


# ---------------------------------------------------
# NORMALIZE COLUMN NAMES
# ---------------------------------------------------
def normalize_col(col):
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_dataframe(df):
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]
    return df


normalized_dfs = {
    name: normalize_dataframe(df)
    for name, df in dataframes.items()
}


# ---------------------------------------------------
# IDENTIFY DATASETS
# ---------------------------------------------------
# IDENTIFY DATASETS (robust)
initiative_df = None
feature_df = None
epic_df = None
cost_df = None
risk_df = None

for name, df in normalized_dfs.items():
    lower_name = name.lower()

    if any(k in lower_name for k in ["feature", "story"]):
        feature_df = df

    elif "epic" in lower_name:
        epic_df = df

    elif any(k in lower_name for k in ["cost", "budget"]):
        cost_df = df

    elif "risk" in lower_name:
        risk_df = df

    elif any(k in lower_name for k in ["initiative", "project"]):
        initiative_df = df


# 🔥 FALLBACK FIX (IMPORTANT)
if feature_df is None:
    # fallback to initiative_df so dashboard never breaks
    feature_df = initiative_df


# ---------------------------------------------------
# RELATIONSHIP DETECTOR
# ---------------------------------------------------
def detect_relationships(normalized_dfs):
    relationships = []
    dataset_names = list(normalized_dfs.keys())

    for i in range(len(dataset_names)):
        for j in range(i + 1, len(dataset_names)):
            df1_name = dataset_names[i]
            df2_name = dataset_names[j]
            df1 = normalized_dfs[df1_name]
            df2 = normalized_dfs[df2_name]

            common_cols = list(
                set(df1.columns).intersection(
                    set(df2.columns)
                )
            )

            for col in common_cols:
                try:
                    # Handle duplicate column names
                    if isinstance(df1[col], pd.DataFrame):
                        series1 = df1[col].iloc[:, 0]
                    else:
                        series1 = df1[col]

                    if isinstance(df2[col], pd.DataFrame):
                        series2 = df2[col].iloc[:, 0]
                    else:
                        series2 = df2[col]

                    vals1 = set(
                        series1.dropna()
                        .astype(str)
                        .unique()
                    )
                    vals2 = set(
                        series2.dropna()
                        .astype(str)
                        .unique()
                    )

                    overlap = vals1.intersection(vals2)

                    if overlap:
                        relationships.append({
                            "table_1": df1_name,
                            "table_2": df2_name,
                            "column": col,
                            "matches": len(overlap)
                        })

                except Exception:
                    continue

    return relationships


# Detect table relationships
relationship_graph = detect_relationships(normalized_dfs)


# ---------------------------------------------------
# API KEY CHECK
# ---------------------------------------------------
def get_api_key():
    """Safely get API key — returns None if missing or placeholder."""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key and not key.startswith("sk-placeholder"):
            return key
    except Exception:
        pass
    return None


# ---------------------------------------------------
# VECTOR STORE (LAZY — only built on first chat use)
# ---------------------------------------------------
@st.cache_resource
def load_vectorstore(_api_key):
    if not docs or not _api_key:
        return None

    try:
        embeddings = OpenAIEmbeddings(api_key=_api_key)
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore.as_retriever()
    except Exception:
        return None


def get_retriever():
    """Lazy retriever — only initializes when called."""
    api_key = get_api_key()
    if not api_key:
        return None
    return load_vectorstore(api_key)


# ---------------------------------------------------
# LLM (LAZY — only created on first chat use)
# ---------------------------------------------------
@st.cache_resource
def load_llm(_api_key):
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=_api_key
    )


def get_llm():
    """Lazy LLM — only initializes when called."""
    api_key = get_api_key()
    if not api_key:
        return None
    return load_llm(api_key)


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def numeric_safe(series):
    return pd.to_numeric(series, errors="coerce")


def find_column(df, possible_cols):
    if df is None:
        return None

    for col in df.columns:
        for keyword in possible_cols:
            if keyword in col.lower():
                return col

    return None
# ---------------------------------------------------
# ROLE BASED DATA FILTER
# ---------------------------------------------------
def apply_role_filter(persona):
    filtered = {}

    try:
        # -----------------------------
        # INITIATIVE FILTER
        # -----------------------------
        if initiative_df is not None:
            df = initiative_df.copy()

            owner_col = find_column(df, ["owner", "assignee"])
            dept_col = find_column(df, ["department", "domain"])

            if persona == "Project Manager":
                if owner_col:
                    filtered["initiative"] = df[
                        df[owner_col].astype(str).str.contains("pm", case=False, na=False)
                    ]
                else:
                    filtered["initiative"] = df.head(3)

            elif persona == "Director":
                if dept_col:
                    filtered["initiative"] = df[
                        df[dept_col].astype(str).str.contains("claims", case=False, na=False)
                    ]
                else:
                    filtered["initiative"] = df.head(6)

            else:  # CIO
                filtered["initiative"] = df

        # -----------------------------
        # COST FILTER
        # -----------------------------
        if cost_df is not None:
            filtered["cost"] = cost_df

        # -----------------------------
        # RISK FILTER
        # -----------------------------
        if risk_df is not None:
            filtered["risk"] = risk_df

        # -----------------------------
        # EPIC FILTER
        # -----------------------------
        if epic_df is not None:
            filtered["epic"] = epic_df

        return filtered

    except Exception:
        return {
            "initiative": initiative_df,
            "cost": cost_df,
            "risk": risk_df,
            "epic": epic_df
        }
# ---------------------------------------------------
# INTENT CLASSIFIER (CONVERSATION-AWARE)
# ---------------------------------------------------
def classify_user_intent(question, chat_history=None):
    history_block = ""
    if chat_history:
        recent = chat_history[-4:]
        lines = []
        for msg in recent:
            lines.append(f"User: {msg['user']}")
            lines.append(f"Assistant: {msg['assistant'][:300]}")
        history_block = "\n".join(lines)

    prompt = f"""Classify this PMO question into ONE category.
Use conversation history to understand follow-up context.

Categories:
- cost (budget, spending, CPI, cost variance, over budget)
- risk (risks, impact, probability, mitigation)
- schedule (SPI, delays, timeline, completion, progress)
- ownership (RACI, who is responsible, accountability)
- portfolio (overall status, multiple projects overview)
- delivery (blockers, epic progress, sprint, velocity)

Conversation history:
{history_block}

Current question: {question}

Return ONLY the category name, nothing else."""

    try:
        llm = get_llm()
        if not llm:
            return "portfolio"
        response = llm.invoke(prompt)
        return response.content.strip().lower()
    except Exception:
        return "portfolio"


# ---------------------------------------------------
# DATA PLANNER
# ---------------------------------------------------
def determine_required_data(intent):
    data_sources = {
        "cost": ["initiative", "cost"],
        "risk": ["initiative", "risk"],
        "schedule": ["initiative", "epic", "feature"],
        "ownership": ["raci"],
        "delivery": ["initiative", "epic", "risk"],
        "portfolio": ["initiative", "cost", "risk", "epic"]
    }

    return data_sources.get(
        intent,
        ["initiative"]
    )


# ---------------------------------------------------
# DIRECTOR DASHBOARD
# ---------------------------------------------------
def show_director_dashboard():
    st.subheader("Director Portfolio Dashboard")

    if filtered_initiative_df is None:
        st.warning("Initiative dataset missing")
        return

    status_col = find_column(
        
        filtered_initiative_df,
        ["status", "state"]
    )

    completion_col = find_column(
        filtered_initiative_df,
        ["completion", "progress"]
    )

    total_projects = len(filtered_initiative_df)

    delayed_projects = 0
    if status_col:
        delayed_projects = len(
            filtered_initiative_df[
                filtered_initiative_df[status_col]
                .astype(str)
                .str.lower()
                .isin(["delayed"])
            ]
        )

    avg_completion = 0
    if completion_col:
        avg_completion = filtered_initiative_df[
            completion_col
        ].mean()

    open_risks = len(filtered_risk_df) if filtered_risk_df is not None else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Initiatives", total_projects)
    c2.metric("Delayed Projects", delayed_projects)
    c3.metric("Portfolio Completion", f"{avg_completion:.1f}%")
    c4.metric("Open Risks", open_risks)

    if status_col:
        fig = px.pie(
            filtered_initiative_df,
            names=status_col,
            title="Portfolio Status Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    if filtered_risk_df is not None:
        impact_col = find_column(filtered_risk_df, ["impact"])
        probability_col = find_column(
            filtered_risk_df,
            ["probability", "likelihood"]
        )

        if impact_col and probability_col:
            heatmap = filtered_risk_df.groupby(
                [impact_col, probability_col]
            ).size().reset_index(name="count")

            fig2 = px.density_heatmap(
                heatmap,
                x=probability_col,
                y=impact_col,
                z="count",
                text_auto=True,
                title="Risk Heatmap"
            )

            st.plotly_chart(fig2, use_container_width=True)
# ---------------------------------------------------
# PM DASHBOARD
# ---------------------------------------------------
def show_pm_dashboard():
    st.subheader("Project Manager Dashboard")

    # -----------------------------
    # Dataset validation
    # -----------------------------
    if filtered_epic_df is None:
        st.warning("Epic dataset missing")
        return

    # Column detection
    epic_name_col = find_column(
        filtered_epic_df,
        ["epic_name", "summary", "name"]
    )

    epic_status_col = find_column(
        filtered_epic_df,
        ["status", "state"]
    )

    epic_id_col = find_column(
        filtered_epic_df,
        ["epic_id", "id", "issue_key"]
    )

    cost_epic_col = None
    actual_cost_col = None

    if filtered_cost_df is not None:
        cost_epic_col = find_column(
            filtered_cost_df,
            ["epic_id", "initiative_id", "id"]
        )

        actual_cost_col = find_column(
            filtered_cost_df,
            ["actual_cost"]
        )
    # -----------------------------
    # KPI Calculations
    # -----------------------------
    total_epics = len(filtered_epic_df)

    blocked_items = 0
    completed_items = 0
    in_progress_items = 0

    if epic_status_col:
        blocked_items = len(
            filtered_epic_df[
                filtered_epic_df[epic_status_col]
                .astype(str)
                .str.lower()
                .isin(["blocked"])
            ]
        )

        completed_items = len(
            filtered_epic_df[
                filtered_epic_df[epic_status_col]
                .astype(str)
                .str.lower()
                .isin(["done", "completed"])
            ]
        )

        in_progress_items = len(
            filtered_epic_df[
                filtered_epic_df[epic_status_col]
                .astype(str)
                .str.lower()
                .isin(["in progress"])
            ]
        )

    total_epic_cost = 0

    if filtered_cost_df is not None and actual_cost_col:
        total_epic_cost = numeric_safe(
            filtered_cost_df[actual_cost_col]
        ).sum()

    open_risks = len(filtered_risk_df) if filtered_risk_df is not None else 0

    # -----------------------------
    # KPI Row
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Epics", total_epics)
    c2.metric("Blocked Items", blocked_items)
    c3.metric("Open Risks", open_risks)
    c4.metric(
        "Epic Cost",
        f"${total_epic_cost/1_000_000:.1f}M"
    )

    # -----------------------------
    # Status Pie Chart
    # -----------------------------
    if epic_status_col:
        fig1 = px.pie(
            filtered_epic_df,
            names=epic_status_col,
            title="Epic Status Distribution"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    # -----------------------------
    # Risk Heat Map
    # -----------------------------
    if filtered_risk_df is not None:
        impact_col = find_column(
            filtered_risk_df,
            ["impact"]
        )

        probability_col = find_column(
            filtered_risk_df,
            ["probability", "likelihood"]
        )

        if impact_col and probability_col:
            heatmap_df = filtered_risk_df.groupby(
                [impact_col, probability_col]
            ).size().reset_index(name="count")

            fig2 = px.density_heatmap(
                heatmap_df,
                x=probability_col,
                y=impact_col,
                z="count",
                text_auto=True,
                title="Risk Heat Map"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

    # -----------------------------
    # Epic-wise Cost Chart
    # -----------------------------
    if (
        filtered_cost_df is not None
        and epic_id_col
        and cost_epic_col
        and actual_cost_col
    ):
        try:
            epic_cost_df = filtered_epic_df.merge(
                filtered_cost_df,
                left_on=epic_id_col,
                right_on=cost_epic_col,
                how="left"
            )

            if epic_name_col:
                cost_summary = epic_cost_df.groupby(
                    epic_name_col
                )[actual_cost_col].sum().reset_index()

                fig3 = px.bar(
                    cost_summary,
                    x=epic_name_col,
                    y=actual_cost_col,
                    title="Epic Wise Cost Distribution"
                )

                st.plotly_chart(
                    fig3,
                    use_container_width=True
                )
        except:
            st.warning(
                "Unable to generate epic cost chart"
            )

    # -----------------------------
    # Epic Table
    # -----------------------------
    st.subheader("Epic Data Table")
    st.dataframe(
        filtered_epic_df.head(20),
        use_container_width=True
    )
# ---------------------------------------------------
# CIO DASHBOARD
# ---------------------------------------------------
def show_cio_dashboard():
    st.subheader("CIO Strategic Dashboard")

    if filtered_initiative_df is None or filtered_cost_df is None:
        st.warning("Required datasets missing")
        return

    # Column detection
    completion_col = find_column(filtered_initiative_df, ["completion"])
    planned_budget_col = find_column(filtered_cost_df, ["planned"])
    actual_cost_col = find_column(filtered_cost_df, ["actual"])
    planned_completion_col = find_column(filtered_cost_df, ["planned_completion"])

    # KPI calculations
    portfolio_completion = (
        filtered_initiative_df[completion_col].mean()
        if completion_col else 0
    )

    planned_budget = (
        numeric_safe(filtered_cost_df[planned_budget_col]).sum()
        if planned_budget_col else 0
    )

    actual_cost = (
        numeric_safe(filtered_cost_df[actual_cost_col]).sum()
        if actual_cost_col else 0
    )

    forecast_cost = (
        actual_cost / (portfolio_completion / 100)
        if portfolio_completion > 0 else actual_cost
    )

    spi = (
        portfolio_completion / 100 /
        (filtered_cost_df[planned_completion_col].mean() / 100)
        if planned_completion_col else 0
    )

    cpi = (
        planned_budget / actual_cost
        if actual_cost > 0 else 0
    )

    open_risks = len(filtered_risk_df) if filtered_risk_df is not None else 0

    confidence = (
        "Low" if spi < 0.8
        else "Medium" if spi < 1
        else "High"
    )

    # KPI ROW 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Completion %", f"{portfolio_completion:.1f}%")
    c2.metric("Planned Budget", f"{planned_budget/1_000_000:.1f}M")
    c3.metric("Total Actual Cost", f"{actual_cost/1_000_000:.1f}M")
    c4.metric("Cost Performance Index", f"{cpi:.2f}")

    # KPI ROW 2
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Forecast Cost", f"{forecast_cost/1_000_000:.1f}M")
    c6.metric("Schedule Performance Index", f"{spi:.2f}")
    c7.metric("Open Risks", open_risks)
    c8.metric("Delivery Confidence", confidence)

    # Risk Heatmap
    if filtered_risk_df is not None:
        impact_col = find_column(filtered_risk_df, ["impact"])
        probability_col = find_column(filtered_risk_df, ["probability", "likelihood"])

        if impact_col and probability_col:
            heatmap = filtered_risk_df.groupby(
                [impact_col, probability_col]
            ).size().reset_index(name="count")

            fig = px.density_heatmap(
                heatmap,
                x=probability_col,
                y=impact_col,
                z="count",
                text_auto=True,
                title="Risk Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Budget Chart
    initiative_name_col = find_column(filtered_cost_df, ["initiative", "name"])

    if initiative_name_col and planned_budget_col and actual_cost_col:
        budget_df = filtered_cost_df.groupby(
            initiative_name_col
        )[[planned_budget_col, actual_cost_col]].sum().reset_index()

        fig = px.bar(
            budget_df,
            x=initiative_name_col,
            y=[planned_budget_col, actual_cost_col],
            barmode="group",
            title="Planned vs Actual Cost"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Completion Chart
    if completion_col:
        fig = px.bar(
            filtered_initiative_df,
            x=initiative_name_col if initiative_name_col else filtered_initiative_df.index,
            y=completion_col,
            title="Completion % by Initiative"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Initiative Details")
    st.dataframe(filtered_initiative_df.head(10))
# ---------------------------------------------------
# CHATBOT ENGINE (FULLY CONVERSATIONAL)
# ---------------------------------------------------
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


PERSONA_STYLE = {
    "Project Manager": "Focus on execution, blockers, and delivery.",
    "Director": "Focus on risks, dependencies, and escalations.",
    "CIO": "Focus on strategy, financial impact, and decisions."
}


def build_system_prompt(persona):
    """Persistent system prompt that defines conversational agent behavior."""
    return f"""You are an intelligent, conversational PMO AI Assistant for an enterprise portfolio.

Persona: {persona}
Communication Style: {PERSONA_STYLE.get(persona, "")}

CRITICAL RULES — FOLLOW EXACTLY:

1. ANSWER WHAT THE USER ASKED — not what you think they should know.
   Read the question carefully. If they ask for SPI, give SPI. If they ask
   about 2 specific initiatives, answer ONLY for those 2.

2. USE CONVERSATION HISTORY — resolve pronouns and references:
   - "these" / "those" / "them" = initiatives mentioned in your last answer
   - "tell me more" = expand on the same topic
   - "what about risks for those?" = risks for previously discussed items

3. USE ONLY REAL DATA — never invent numbers. Pull values directly from
   the data tables provided. Show your calculation when computing metrics
   like SPI, CPI, variance, etc.

4. FORMULAS (use these exactly):
   - SPI = Completion_Percentage / Planned_Completion_Percentage
   - CPI = Planned_Budget_USD / Actual_Cost_USD
   - Budget Variance = Planned_Budget_USD - Actual_Cost_USD

5. STRUCTURE your response as:
   - **Direct Answer:** (answer the exact question first)
   - **Key Insights:** (additional relevant context)
   - **Recommended Actions:** (1-3 actionable next steps)

6. NEVER say "Project A" or "Project B" — use real initiative names and IDs.

7. DO NOT repeat information already given in prior messages.

8. Use markdown formatting (headers, bullets, bold) for readability.
"""


def clean_dataframe(df, max_cols=15):
    """Remove unnamed/empty columns and limit column count for LLM context."""
    if df is None:
        return None
    cleaned = df.copy()
    unnamed_cols = [c for c in cleaned.columns if "unnamed" in str(c).lower()]
    if unnamed_cols:
        cleaned = cleaned.drop(columns=unnamed_cols)
    cleaned = cleaned.dropna(axis=1, how="all")
    if len(cleaned.columns) > max_cols:
        cleaned = cleaned.iloc[:, :max_cols]
    return cleaned


def build_merged_context():
    """Build clean, merged data context for the LLM."""
    if filtered_initiative_df is None:
        return "No initiative data.", "", ""

    clean_init = clean_dataframe(filtered_initiative_df)

    init_id_col = find_column(clean_init, ["initiative_id", "issue_key", "id"])

    merged = clean_init.copy()

    if filtered_cost_df is not None and init_id_col:
        clean_cost = clean_dataframe(filtered_cost_df)
        cost_id_col = find_column(clean_cost, ["initiative_id", "id"])

        if cost_id_col:
            try:
                cost_cols_to_use = [cost_id_col]
                for c in clean_cost.columns:
                    if c != cost_id_col and c not in merged.columns:
                        cost_cols_to_use.append(c)

                merged = merged.merge(
                    clean_cost[cost_cols_to_use],
                    left_on=init_id_col,
                    right_on=cost_id_col,
                    how="left"
                )
            except Exception:
                pass

    merged = merged.dropna(how="all")
    merged_context = merged.head(20).to_string(index=False)

    if merged_context.count("NaN") > 50:
        init_context = clean_init.head(20).to_string(index=False)
        cost_context = ""
        if filtered_cost_df is not None:
            cost_context = clean_dataframe(filtered_cost_df).head(10).to_string(index=False)
        merged_context = f"Initiatives:\n{init_context}\n\nCost Data:\n{cost_context}"

    risk_context = ""
    if filtered_risk_df is not None:
        risk_context = clean_dataframe(filtered_risk_df).head(15).to_string(index=False)

    epic_context = ""
    if filtered_epic_df is not None:
        epic_context = clean_dataframe(filtered_epic_df).head(15).to_string(index=False)

    return merged_context, risk_context, epic_context


def run_agent(question, persona, chat_history=None):
    try:
        if filtered_initiative_df is None:
            return "No initiative dataset found."

        # -----------------------------
        # BUILD MERGED DATA CONTEXT
        # -----------------------------
        merged_context, risk_context, epic_context = build_merged_context()

        # -----------------------------
        # RAG CONTEXT (your documents)
        # -----------------------------
        rag_context = ""
        retriever = get_retriever()

        if retriever:
            try:
                rag_docs = retriever.invoke(question)
                rag_context = "\n".join([d.page_content for d in rag_docs[:3]])
            except Exception:
                pass

        # -----------------------------
        # BUILD CONVERSATIONAL MESSAGE CHAIN
        # -----------------------------
        llm = get_llm()
        if not llm:
            return "⚠️ OpenAI API key not configured. Please add OPENAI_API_KEY to .streamlit/secrets.toml"

        messages = [
            SystemMessage(content=build_system_prompt(persona))
        ]

        if chat_history:
            for msg in chat_history:
                messages.append(HumanMessage(content=msg["user"]))
                messages.append(AIMessage(content=msg["assistant"]))

        user_content = f"""{question}

--- Portfolio Data (Initiatives + Cost merged) ---
{merged_context}

--- Risk Register ---
{risk_context if risk_context else "No risk data available."}

--- Epics ---
{epic_context if epic_context else "No epic data available."}

--- RACI / Documents ---
{rag_context if rag_context else "No RACI document context available."}

IMPORTANT: Use the data above to answer precisely. For SPI calculations,
use: SPI = Actual_Completion / Planned_Completion (from the merged data).
For CPI: CPI = Planned_Budget / Actual_Cost. Always reference real initiative
names and IDs from the data.
"""
        messages.append(HumanMessage(content=user_content))

        response = llm.invoke(messages)

        return response.content

    except Exception as e:
        return f"AI analysis failed: {str(e)}"
# ---------------------------------------------------
# PROACTIVE ALERT ENGINE
# ---------------------------------------------------
def generate_persona_alerts(persona):
    alerts = []

    try:
        # -----------------------------
        # Cost Alerts
        # -----------------------------
        if filtered_cost_df is not None:
            planned_col = find_column(
                filtered_cost_df,
                ["planned_budget", "planned"]
            )

            actual_col = find_column(
                filtered_cost_df,
                ["actual_cost", "actual"]
            )

            if planned_col and actual_col:
                planned_total =(
                    numeric_safe(filtered_cost_df.get(planned_col))
                ).sum()

                actual_total = numeric_safe(
                    filtered_cost_df[actual_col]
                ).sum()

                if actual_total > planned_total:
                    alerts.append(
                        f"🔴 Budget exceeded by ${actual_total-planned_total:,.0f}"
                    )

                elif actual_total < planned_total:
                    alerts.append(
                        f"🟢 Cost savings identified: ${planned_total-actual_total:,.0f}"
                    )

        # -----------------------------
        # Delay Alerts
        # -----------------------------
        if filtered_initiative_df is not None:
            status_col = find_column(
                filtered_initiative_df,
                ["status"]
            )

            if status_col:
                delayed_count = len(
                    filtered_initiative_df[
                        filtered_initiative_df[status_col]
                        .astype(str)
                        .str.lower()
                        .isin(["delayed"])
                    ]
                )

                if delayed_count > 0:
                    alerts.append(
                        f"🟡 {delayed_count} initiatives delayed"
                    )

        # -----------------------------
        # Risk Alerts
        # -----------------------------
        if filtered_risk_df is not None:
            impact_col = find_column(
                filtered_risk_df,
                ["impact"]
            )

            if impact_col:
                high_risk_count = len(
                    filtered_risk_df[
                        filtered_risk_df[impact_col]
                        .astype(str)
                        .str.lower()
                        .str.contains(
                            "high|critical",
                            na=False
                        )
                    ]
                )

                if high_risk_count > 0:
                    alerts.append(
                        f"🔴 {high_risk_count} critical risks require attention"
                    )

        # -----------------------------
        # CIO Strategic Alert
        # -----------------------------
        if persona == "CIO":
            alerts.append(
                "🟢 Market opportunity detected: competitors lag in AI claims automation"
            )

        # Persona filtering
        if persona == "Project Manager":
            return alerts[:2]

        elif persona == "Director":
            return alerts[:3]

        elif persona == "CIO":
            return alerts[:4]

        return alerts

    except Exception:
        return ["No alerts available"]
# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("🤖 Enterprise PMO Assistant")

# ---------------------------------------------------
# SIMULATED LOGIN + DEMO ROLE SWITCH
# ---------------------------------------------------

st.sidebar.subheader("User Context")

# Simulated logged-in user
user_email = "divya@company.com"

# Default role (from login system in real world)
default_role = "Director"
# Demo mode toggle
demo_mode = st.sidebar.checkbox("Enable Demo Role Switch", value=True)

if demo_mode:
    persona = st.sidebar.selectbox(
        "Switch Role (Demo Mode)",
        ["Director", "Project Manager", "CIO"],
        index=0
    )
else:
    persona = default_role

# Display login info in main UI
st.markdown(f"""
### 👤 Logged in as: {user_email}  
**Role:** {persona}
""")
# Apply role-based filtering
filtered_data = apply_role_filter(persona)

filtered_initiative_df = filtered_data.get("initiative")
filtered_cost_df = filtered_data.get("cost")
filtered_risk_df = filtered_data.get("risk")
filtered_epic_df = filtered_data.get("epic")
tab1, tab2 = st.tabs([
    "📊 Dashboard",
    "💬 Chat Assistant"
])


with tab1:
    if persona == "Director":
        show_director_dashboard()

    elif persona == "Project Manager":
        show_pm_dashboard()

    elif persona == "CIO":
        show_cio_dashboard()
with tab2:
    st.subheader("Alert Automation")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("⏰ Schedule Daily Alerts"):
            st.success("Alerts scheduled daily at 9:00 AM for selected persona")

    with c2:
        run_now = st.button("▶ Run Alerts Now")

    # -----------------------------
    # ALERTS SECTION
    # -----------------------------
    st.subheader("AI Notification Center")

    if run_now:
        alerts = generate_persona_alerts(persona)
    else:
        alerts = []

    if alerts:
        for alert in alerts:
            if "🔴" in alert:
                st.error(alert)
            elif "🟡" in alert:
                st.warning(alert)
            else:
                st.success(alert)

        # Simulated delivery
        st.markdown("### Alert Delivery")

        if persona == "Project Manager":
            st.write("📩 Sent via Teams to Project Manager")

        elif persona == "Director":
            st.write("📩 Sent via Email to Director")

        elif persona == "CIO":
            st.write("📩 Sent via Executive Dashboard Notification")

    else:
        st.info("No alerts triggered. Click 'Run Alerts Now' to simulate.")

    # -----------------------------
    # CHAT SECTION (FULLY CONVERSATIONAL)
    # -----------------------------
    st.subheader("PMO Chat Assistant")

    chat_header_col1, chat_header_col2 = st.columns([4, 1])
    with chat_header_col1:
        st.caption(
            "Fully conversational — ask follow-ups and I'll remember context."
        )
    with chat_header_col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["assistant"])

    question = st.chat_input(
        "Ask project, budget, risk or delivery questions..."
    )

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Analyzing portfolio..."):
            answer = run_agent(
                question,
                persona,
                chat_history=st.session_state.chat_history
            )

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.chat_history.append({
            "user": question,
            "assistant": answer
        })
