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

    if any(k in lower_name for k in ["initiative", "project"]):
        initiative_df = df

    elif any(k in lower_name for k in ["feature", "story"]):
        feature_df = df

    elif "epic" in lower_name:
        epic_df = df

    elif any(k in lower_name for k in ["cost", "budget"]):
        cost_df = df

    elif "risk" in lower_name:
        risk_df = df


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
# VECTOR STORE
# ---------------------------------------------------
@st.cache_resource
def load_vectorstore():
    if not docs:
        return None

    try:
        embeddings = OpenAIEmbeddings(
            api_key=st.secrets["OPENAI_API_KEY"]
        )

        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

        return vectorstore.as_retriever()

    except Exception:
        return None


retriever = load_vectorstore()


# ---------------------------------------------------
# LLM
# ---------------------------------------------------
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=st.secrets["OPENAI_API_KEY"]
    )


agent_llm = load_llm()


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
# INTENT CLASSIFIER
# ---------------------------------------------------
def classify_user_intent(question):
    prompt = f"""
    Classify this PMO question into ONE category:

    - cost
    - risk
    - schedule
    - ownership
    - portfolio
    - delivery

    Question:
    {question}

    Return only category name.
    """

    try:
        response = agent_llm.invoke(prompt)
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

    if initiative_df is None:
        st.warning("Initiative dataset missing")
        return

    status_col = find_column(
        
        initiative_df,
        ["status", "state"]
    )

    completion_col = find_column(
        initiative_df,
        ["completion", "progress"]
    )

    total_projects = len(initiative_df)

    delayed_projects = 0
    if status_col:
        delayed_projects = len(
            initiative_df[
                initiative_df[status_col]
                .astype(str)
                .str.lower()
                .isin(["delayed"])
            ]
        )

    avg_completion = 0
    if completion_col:
        avg_completion = initiative_df[
            completion_col
        ].mean()

    open_risks = len(risk_df) if risk_df is not None else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Initiatives", total_projects)
    c2.metric("Delayed Projects", delayed_projects)
    c3.metric("Portfolio Completion", f"{avg_completion:.1f}%")
    c4.metric("Open Risks", open_risks)

    if status_col:
        fig = px.pie(
            initiative_df,
            names=status_col,
            title="Portfolio Status Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    if risk_df is not None:
        impact_col = find_column(risk_df, ["impact"])
        probability_col = find_column(
            risk_df,
            ["probability", "likelihood"]
        )

        if impact_col and probability_col:
            heatmap = risk_df.groupby(
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
    if epic_df is None:
        st.warning("Epic dataset missing")
        return

    # Column detection
    epic_name_col = find_column(
        epic_df,
        ["epic_name", "summary", "name"]
    )

    epic_status_col = find_column(
        epic_df,
        ["status", "state"]
    )

    epic_id_col = find_column(
        epic_df,
        ["epic_id", "id", "issue_key"]
    )

    cost_epic_col = None
    actual_cost_col = None

    if cost_df is not None:
        cost_epic_col = find_column(
            cost_df,
            ["epic_id", "initiative_id", "id"]
        )

        actual_cost_col = find_column(
            cost_df,
            ["actual_cost"]
        )
    # -----------------------------
    # KPI Calculations
    # -----------------------------
    total_epics = len(epic_df)

    blocked_items = 0
    completed_items = 0
    in_progress_items = 0

    if epic_status_col:
        blocked_items = len(
            epic_df[
                epic_df[epic_status_col]
                .astype(str)
                .str.lower()
                .isin(["blocked"])
            ]
        )

        completed_items = len(
            epic_df[
                epic_df[epic_status_col]
                .astype(str)
                .str.lower()
                .isin(["done", "completed"])
            ]
        )

        in_progress_items = len(
            epic_df[
                epic_df[epic_status_col]
                .astype(str)
                .str.lower()
                .isin(["in progress"])
            ]
        )

    total_epic_cost = 0

    if cost_df is not None and actual_cost_col:
        total_epic_cost = numeric_safe(
            cost_df[actual_cost_col]
        ).sum()

    open_risks = len(risk_df) if risk_df is not None else 0

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
            epic_df,
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
    if risk_df is not None:
        impact_col = find_column(
            risk_df,
            ["impact"]
        )

        probability_col = find_column(
            risk_df,
            ["probability", "likelihood"]
        )

        if impact_col and probability_col:
            heatmap_df = risk_df.groupby(
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
        cost_df is not None
        and epic_id_col
        and cost_epic_col
        and actual_cost_col
    ):
        try:
            epic_cost_df = epic_df.merge(
                cost_df,
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
        epic_df.head(20),
        use_container_width=True
    )
# ---------------------------------------------------
# CIO DASHBOARD
# ---------------------------------------------------
def show_cio_dashboard():
    st.subheader("CIO Strategic Dashboard")

    if initiative_df is None or cost_df is None:
        st.warning("Required datasets missing")
        return

    # Column detection
    completion_col = find_column(initiative_df, ["completion"])
    planned_budget_col = find_column(cost_df, ["planned"])
    actual_cost_col = find_column(cost_df, ["actual"])
    planned_completion_col = find_column(cost_df, ["planned_completion"])

    # KPI calculations
    portfolio_completion = (
        initiative_df[completion_col].mean()
        if completion_col else 0
    )

    planned_budget = (
        numeric_safe(cost_df[planned_budget_col]).sum()
        if planned_budget_col else 0
    )

    actual_cost = (
        numeric_safe(cost_df[actual_cost_col]).sum()
        if actual_cost_col else 0
    )

    forecast_cost = (
        actual_cost / (portfolio_completion / 100)
        if portfolio_completion > 0 else actual_cost
    )

    spi = (
        portfolio_completion / 100 /
        (cost_df[planned_completion_col].mean() / 100)
        if planned_completion_col else 0
    )

    cpi = (
        planned_budget / actual_cost
        if actual_cost > 0 else 0
    )

    open_risks = len(risk_df) if risk_df is not None else 0

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
    if risk_df is not None:
        impact_col = find_column(risk_df, ["impact"])
        probability_col = find_column(risk_df, ["probability", "likelihood"])

        if impact_col and probability_col:
            heatmap = risk_df.groupby(
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
    initiative_name_col = find_column(cost_df, ["initiative", "name"])

    if initiative_name_col and planned_budget_col and actual_cost_col:
        budget_df = cost_df.groupby(
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
            initiative_df,
            x=initiative_name_col if initiative_name_col else initiative_df.index,
            y=completion_col,
            title="Completion % by Initiative"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Table
    st.subheader("Initiative Details")
    st.dataframe(initiative_df.head(10))
# ---------------------------------------------------
# CHATBOT ENGINE
# ---------------------------------------------------
# ---------------------------------------------------
# CHATBOT ENGINE
# ---------------------------------------------------
def run_agent(question, persona):
    try:
        if initiative_df is None:
            return "No initiative dataset found."

        intent = classify_user_intent(question)

        initiative_id_col = find_column(
            initiative_df,
            ["initiative_id", "issue_key", "id"]
        )

        initiative_name_col = find_column(
            initiative_df,
            ["initiative_name", "summary", "name"]
        )

        planned_budget_col = find_column(
            cost_df,
            ["planned_budget", "planned"]
        )

        actual_cost_col = find_column(
            cost_df,
            ["actual_cost", "actual"]
        )

        risk_link_col = find_column(
            risk_df,
            ["linked_id", "initiative_id"]
        )

        master_df = initiative_df.copy()

        # Merge cost data
        if (
            cost_df is not None
            and initiative_id_col
            and initiative_id_col in cost_df.columns
        ):
            master_df = master_df.merge(
                cost_df,
                on=initiative_id_col,
                how="left"
            )

        # Merge risk data
        if (
            risk_df is not None
            and risk_link_col
            and initiative_id_col
        ):
            master_df = master_df.merge(
                risk_df,
                left_on=initiative_id_col,
                right_on=risk_link_col,
                how="left"
            )

        status_col = find_column(
            master_df,
            ["status", "state"]
        )

        risk_impact_col = find_column(
            master_df,
            ["impact"]
        )

        # Risk projects
        high_risk_projects = pd.DataFrame()

        if risk_impact_col:
            high_risk_projects = master_df[
                master_df[risk_impact_col]
                .astype(str)
                .str.lower()
                .str.contains(
                    "high|critical|severe|red|p1",
                    na=False
                )
            ]

        # Delayed projects
        delayed_projects = pd.DataFrame()

        if status_col:
            delayed_projects = master_df[
                master_df[status_col]
                .astype(str)
                .str.lower()
                .isin(["delayed"])
            ]

        # Over budget projects
        over_budget_projects = pd.DataFrame()

        if planned_budget_col and actual_cost_col:
            over_budget_projects = master_df[
                numeric_safe(master_df[actual_cost_col]) >
                numeric_safe(master_df[planned_budget_col])
            ]

        # Intent routing
        if intent == "risk":
            analysis_df = high_risk_projects

        elif intent == "cost":
            analysis_df = over_budget_projects

        elif intent == "schedule":
            analysis_df = delayed_projects

        else:
            analysis_df = master_df

        if analysis_df.empty:
            analysis_df = master_df.head(5)

        summary_cols = []

        for col in [
            initiative_name_col,
            status_col,
            planned_budget_col,
            actual_cost_col,
            risk_impact_col
        ]:
            if col and col in analysis_df.columns:
                summary_cols.append(col)

        if summary_cols:
            final_summary = analysis_df[
                summary_cols
            ].head(5)
        else:
            final_summary = analysis_df.head(5)

        # RAG retrieval
        raci_context = ""

        if retriever:
            try:
                docs = retriever.invoke(question)

                raci_context = "\n".join([
                    doc.page_content
                    for doc in docs[:3]
                ])

            except Exception:
                pass

        persona_guidance = {
            "Project Manager":
                "Focus on blockers, execution recovery and sprint delivery.",

            "Director":
                "Focus on portfolio risks, escalations and dependencies.",

            "CIO":
                "Focus on strategy, financial exposure and executive decisions."
        }

        prompt = f"""
You are an intelligent PMO AI Assistant.

Question:
{question}

Persona:
{persona}

Intent:
{intent}

Communication Style:
{persona_guidance.get(persona)}

Analyzed Data:
{final_summary.to_string()}

RAG Context:
{raci_context}

Instructions:
- Analyze cross dataset relationships
- Provide business summary
- Recommend actions
- Avoid repetitive responses
"""

        response = agent_llm.invoke(prompt)

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
        if cost_df is not None:
            planned_col = find_column(
                cost_df,
                ["planned_budget", "planned"]
            )

            actual_col = find_column(
                cost_df,
                ["actual_cost", "actual"]
            )

            if planned_col and actual_col:
                planned_total = numeric_safe(
                    cost_df[planned_col]
                ).sum()

                actual_total = numeric_safe(
                    cost_df[actual_col]
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
        if initiative_df is not None:
            status_col = find_column(
                initiative_df,
                ["status"]
            )

            if status_col:
                delayed_count = len(
                    initiative_df[
                        initiative_df[status_col]
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
        if risk_df is not None:
            impact_col = find_column(
                risk_df,
                ["impact"]
            )

            if impact_col:
                high_risk_count = len(
                    risk_df[
                        risk_df[impact_col]
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

persona = st.selectbox(
    "Select Persona",
    ["Director", "Project Manager", "CIO"]
)

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
    st.subheader("AI Notification Center")

    alerts = generate_persona_alerts(persona)

    if alerts:
        for alert in alerts:
            if "🔴" in alert:
                st.error(alert)

            elif "🟡" in alert:
                st.warning(alert)

            else:
                st.success(alert)

    st.subheader("PMO Chat Assistant")

    # Initialize session history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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
                persona
            )

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.chat_history.append({
            "user": question,
            "assistant": answer
        })