import os
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu

try:
    import shap
except ImportError:
    shap = None

from utils.auth import logout, render_sign_in
from utils.config import (
    APP_ICON,
    APP_TITLE,
    BINARY_OPTIONS,
    CONTRACT_OPTIONS,
    INTERNET_SERVICES,
    LOTTIE_HERO,
    PAYMENT_METHODS,
)
from utils.data_loader import (
    clean_dataset,
    ensure_churn_flag,
    get_active_dataset,
    load_dataset,
    summarize_data,
)
from utils.helpers import load_css, load_lottie_url
from utils.modeling import get_best_model, prepare_features, predict_with_model, train_models
from components.animations import render_lottie_animation
from components.cards import render_kpi
from components.charts import risk_gauge

load_css("assets/css/styles.css")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_OPTIONS = [
    "Landing",
    "Upload",
    "EDA",
    "Explainable AI",
    "Prediction",
]


def load_active_model():
    pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_pipeline.pkl")
    pipeline_path = os.path.normpath(pipeline_path)
    if os.path.exists(pipeline_path):
        try:
            return joblib.load(pipeline_path)
        except Exception:
            return None
    active_df = get_active_dataset()
    df = active_df if active_df is not None else load_dataset()
    results = train_models(df)
    best = get_best_model(results)
    return best["model"] if best else None


def render_landing():
    st.markdown(
        """
        <style>
        .hero-shell {
            position: relative;
            padding: 3rem 0 1rem;
            overflow: hidden;
        }
        .hero-shell::before,
        .hero-shell::after {
            content: '';
            position: absolute;
            border-radius: 50%;
            filter: blur(110px);
            opacity: 0.75;
        }
        .hero-shell::before {
            width: 380px;
            height: 380px;
            background: rgba(56, 189, 248, 0.25);
            top: -110px;
            left: -90px;
        }
        .hero-shell::after {
            width: 420px;
            height: 420px;
            background: rgba(52, 211, 153, 0.18);
            bottom: -120px;
            right: -90px;
        }
        .hero-strip {
            padding: 2rem;
            border-radius: 36px;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 30px 90px rgba(0, 0, 0, 0.22);
        }
        .hero-flag {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 1.4rem;
            padding: 0.55rem 1rem;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.14);
            color: #81e6d9;
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: 0.04em;
        }
        .hero-title {
            font-size: clamp(2.9rem, 5vw, 4.8rem);
            line-height: 1.02;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #7dd3fc, #22c55e);
            -webkit-background-clip: text;
            color: transparent;
            font-weight: 800;
        }
        .hero-copy {
            max-width: 760px;
            font-size: 1.05rem;
            color: #cbd5e1;
            line-height: 1.9;
            margin-bottom: 1.8rem;
        }
        .hero-stats {
            display: grid;
            gap: 1rem;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        }
        .hero-card {
            padding: 1.2rem 1.2rem 1rem;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(14, 23, 42, 0.78);
            min-height: 140px;
            transition: transform 0.18s ease, border-color 0.18s ease;
        }
        .hero-card:hover { transform: translateY(-4px); border-color: rgba(96, 165, 250, 0.35); }
        .hero-card h3 { margin: 0 0 0.6rem; color: #e2e8f0; }
        .hero-card p { margin: 0; color: #94a3b8; }
        .hero-number { font-size: 2rem; font-weight: 800; color: #f8fafc; margin-top: 0.5rem; }
        .feature-grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); margin-top: 2rem; }
        .feature-card { padding: 1.3rem; border-radius: 24px; background: rgba(15, 23, 42, 0.88); border: 1px solid rgba(255,255,255,0.07); }
        .feature-card h4 { margin-bottom: 0.85rem; }
        .feature-card p { color: #cbd5e1; margin: 0; line-height: 1.75; }
        .feature-pill { display: inline-flex; gap: 0.45rem; align-items: center; color: #22c55e; font-size: 0.95rem; margin-bottom: 1rem; }
        .feature-pill span { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; }
        </style>

        <div class='hero-shell'>
            <div class='hero-strip'>
                <div class='hero-flag'>Real-time churn intelligence for modern growth teams</div>
                <h1 class='hero-title'>Predict customer churn and unlock retention with confidence.</h1>
                <div class='hero-copy'>
                    ChurnAI Studio brings your customer dataset into a powerful analytics workflow. Visualize risk trends, understand drivers, and deploy retention-ready recommendations — all in one elegant workspace.
                </div>
                <div class='hero-stats'>
                    <div class='hero-card'><h3>Predictive accuracy</h3><div class='hero-number'>89.4%</div><p>High-performance churn scoring built for enterprise data.</p></div>
                    <div class='hero-card'><h3>Instant insights</h3><div class='hero-number'>3s</div><p>Model inference and analytics refresh in seconds.</p></div>
                    <div class='hero-card'><h3>Fast deployment</h3><div class='hero-number'>1 workflow</div><p>From upload to prediction, the experience is seamless and repeatable.</p></div>
                </div>
                <div class='feature-grid'>
                    <div class='feature-card'><div class='feature-pill'><span></span>Smart feature engineering</div><h4>Automated dataset preparation</h4><p>Clean, convert, and normalize customer fields with minimal effort.</p></div>
                    <div class='feature-card'><div class='feature-pill'><span></span>Explainability-ready</div><h4>Understand churn drivers</h4><p>Discover why customers are at risk with explainable model outputs.</p></div>
                    <div class='feature-card'><div class='feature-pill'><span></span>Risk-based intelligence</div><h4>Probability-first scoring</h4><p>See churn likelihood as a clear risk score, not just a yes/no label.</p></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = get_active_dataset()
    if df is not None:
        summary = summarize_data(df)
        total_customers = summary["shape"][0]
        churn_rate = df["churn_flag"].mean()
        revenue = df["monthly_charges"].sum() * 12
        retention_rate = 1 - churn_rate

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("### Current dataset pulse")

        with st.container():
            c1, c2, c3, c4 = st.columns(4, gap="large")
            render_kpi("Active Customers", f"{total_customers:,}", "Live dataset overview")
            render_kpi("Churn Rate", f"{churn_rate:.1%}", "Current risk profile")
            render_kpi("Annualized ARR", f"${revenue/1_000_000:.2f}M", "Revenue at risk")
            render_kpi("Retention", f"{retention_rate:.1%}", "Effort to preserve revenue")
    else:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("### Ready when your customer dataset is")
        st.write("Upload your churn dataset on the Upload page to unlock modeling, explainability, and prediction.")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### Built for modern customer intelligence")
    cards = st.columns(3, gap="large")
    cards[0].markdown("<div class='glass-card'><h4>Unified analytics</h4><p>From raw data through predictive scoring and retention recommendations.</p></div>", unsafe_allow_html=True)
    cards[1].markdown("<div class='glass-card'><h4>Explainable decisions</h4><p>Make model outputs actionable with features and SHAP-powered views.</p></div>", unsafe_allow_html=True)
    cards[2].markdown("<div class='glass-card'><h4>Risk-first strategy</h4><p>Turn churn probability into prioritized customer actions.</p></div>", unsafe_allow_html=True)
def render_upload():
    st.markdown("## Dataset Upload & Transformation")
    st.markdown("<p class='highlight-pill'>Upload, clean, and convert your dataset before analysis.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV or XLSX dataset", type=["csv", "xlsx"], help="Drop your customer churn file here.")
    load_sample = st.button("Load sample dataset")

    df = None
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        df = ensure_churn_flag(df)
        st.session_state["uploaded_df"] = df
        st.session_state["cleaned_df"] = df.copy()
        st.session_state["summary"] = summarize_data(df)
    elif load_sample:
        df = load_dataset()
        df = ensure_churn_flag(df)
        st.session_state["uploaded_df"] = df
        st.session_state["cleaned_df"] = df.copy()
        st.session_state["summary"] = summarize_data(df)
    else:
        df = st.session_state.get("uploaded_df")

    summary = st.session_state.get("summary")
    if summary is None and df is not None:
        summary = summarize_data(df)
        st.session_state["summary"] = summary

    if df is None:
        st.warning("Please upload a dataset or load the sample dataset before exploring EDA or prediction.")
        return

    st.markdown(
        "<div class='glass-card'>\n  <div style='display:flex;justify-content:space-between;align-items:center;'>\n    <div><h3>Loaded dataset overview</h3><p style='color:#9ca3af;'>Metrics reflect the dataset currently loaded in the session.</p></div>\n    <div style='text-align:right;'><strong>Health</strong> <span style='color:#60a5fa;font-size:1.1rem;'>{summary['health_score']}%</span></div>\n  </div>\n</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", summary["shape"][0])
    c2.metric("Columns", summary["shape"][1])
    c3.metric("Missing", int(summary["missing"].sum()))
    c4.metric("Duplicates", summary["duplicates"])

    with st.expander("View data preview"):
        st.dataframe(df.head(12), use_container_width=True)

    with st.expander("Data types & sample values"):
        type_table = pd.DataFrame.from_dict(summary["dtypes"], orient="index", columns=["dtype"]).reset_index()
        type_table.columns = ["Feature", "Data Type"]
        st.table(type_table)

    cleaned_df = st.session_state.get("cleaned_df", df.copy())

    st.markdown("<div class='section-card'><h2>Clean dataset</h2><p>Run the cleaning pipeline to normalize values, remove duplicates, and fill missing values.</p></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        if st.button("CLEAN DATASET", key="clean_dataset"):
            cleaned_df = clean_dataset(df)
            st.session_state["cleaned_df"] = cleaned_df
            st.session_state["summary"] = summarize_data(cleaned_df)
            st.success("Cleaning complete. The cleaned dataset is available for download and analysis.")
        st.markdown("<div class='metric-pill'>Removes duplicates, fills missing numeric values, and standardizes categorical fields.</div>", unsafe_allow_html=True)
    with c2:
        before_summary = summarize_data(df)
        after_summary = summarize_data(cleaned_df)
        st.markdown(
            f"""
            <div class='metric-mini-card'>
                <strong>Before</strong><br/>
                Rows: {before_summary['shape'][0]}<br/>
                Missing: {int(before_summary['missing'].sum())}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class='metric-mini-card'>
                <strong>After</strong><br/>
                Rows: {after_summary['shape'][0]}<br/>
                Missing: {int(after_summary['missing'].sum())}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-card'><h2>Data type conversion</h2><p>Correct column types before analysis or prediction.</p></div>", unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns([2, 2, 1])
    with cc1:
        column_to_convert = st.selectbox("Choose column", df.columns)
    with cc2:
        target_type = st.selectbox("Target type", ["int", "float", "object", "category", "datetime", "bool"])
    with cc3:
        if st.button("Apply conversion", key="convert_type"):
            converted, error = convert_column_type(df, column_to_convert, target_type)
            if error:
                st.error(f"Conversion error: {error}")
            else:
                converted = ensure_churn_flag(converted)
                st.session_state["uploaded_df"] = converted
                st.session_state["cleaned_df"] = converted.copy()
                st.success(f"Converted {column_to_convert} to {target_type}.")

    if cleaned_df is not None:
        st.markdown("<div class='section-card'><h2>Download cleaned dataset</h2><p>Export the prepared dataset for further use.</p></div>", unsafe_allow_html=True)
        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download CSV", csv_bytes, file_name="cleaned_dataset.csv", mime="text/csv")
        with d2:
            st.warning("Excel export requires the 'openpyxl' package. Install with: pip install openpyxl")

    st.session_state["uploaded_df"] = st.session_state.get("uploaded_df", df)
    st.session_state["cleaned_df"] = st.session_state.get("cleaned_df", cleaned_df)


def convert_column_type(df: pd.DataFrame, column: str, target_type: str):
    df = df.copy()
    try:
        if target_type == "int":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
        elif target_type == "float":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
        elif target_type == "object":
            df[column] = df[column].astype(str)
        elif target_type == "category":
            df[column] = df[column].astype("category")
        elif target_type == "datetime":
            df[column] = pd.to_datetime(df[column], errors="coerce")
        elif target_type == "bool":
            df[column] = df[column].astype(str).str.lower().map({"yes": True, "no": False, "true": True, "false": False, "1": True, "0": False})
            df[column] = df[column].astype("boolean")
        return df, None
    except Exception as error:
        return df, str(error)


def render_eda():
    st.markdown("## Exploratory Data Analysis")
    st.markdown("<p class='highlight-pill'>Choose a chart type, then select the columns to analyze.</p>", unsafe_allow_html=True)

    df = get_active_dataset()
    if df is None:
        st.warning("No dataset loaded. Please upload or load a dataset on the Upload page first.")
        return

    churn_rate = df["churn_flag"].mean() if "churn_flag" in df.columns else df["churn"].map({"Yes": 1, "No": 0}).mean()
    total_customers = len(df)
    average_monthly = df["monthly_charges"].mean() if "monthly_charges" in df.columns else 0
    support_avg = df["support_calls"].mean() if "support_calls" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        render_kpi("Customers", f"{total_customers:,}")
    with c2:
        render_kpi("Churn rate", f"{churn_rate:.1%}")
    with c3:
        render_kpi("Avg monthly charge", f"${average_monthly:.2f}")
    with c4:
        render_kpi("Avg support calls", f"{support_avg:.1f}")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    category_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if "churn" in df.columns and "churn" not in category_cols:
        category_cols.append("churn")

    methods = [
        "Scatter",
        "Histogram",
        "Boxplot",
        "Donut Chart",
        "Violin Plot",
        "Line Plot",
        "Bar Plot",
        "Count Plot",
    ]

    if "eda_method" not in st.session_state:
        st.session_state["eda_method"] = methods[0]

    st.markdown("### Select visualization method")
    btn_cols = st.columns(4, gap="small")
    for idx, method in enumerate(methods):
        if btn_cols[idx % 4].button(method, key=f"eda_{method}"):
            st.session_state["eda_method"] = method

    selected_method = st.session_state["eda_method"]
    st.markdown(f"### {selected_method}")

    def plot(fig):
        st.plotly_chart(fig, use_container_width=True)

    if selected_method == "Scatter":
        x_col = st.selectbox("X column", numeric_cols, index=0 if numeric_cols else None)
        y_col = st.selectbox("Y column", [col for col in numeric_cols if col != x_col], index=0 if len(numeric_cols) > 1 else None)
        color_col = st.selectbox("Color by", [None] + category_cols, format_func=lambda x: x if x else "None")
        if x_col and y_col:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col if color_col else None,
                title=f"Scatter: {x_col} vs {y_col}",
                template="plotly_white",
            )
            plot(fig)

    elif selected_method == "Histogram":
        x_col = st.selectbox("Value column", numeric_cols, index=0 if numeric_cols else None)
        color_col = st.selectbox("Group by", [None] + category_cols, format_func=lambda x: x if x else "None")
        if x_col:
            fig = px.histogram(
                df,
                x=x_col,
                color=color_col if color_col else None,
                nbins=30,
                title=f"Histogram of {x_col}",
                template="plotly_white",
            )
            plot(fig)

    elif selected_method == "Boxplot":
        x_col = st.selectbox("Category", category_cols, index=0 if category_cols else None)
        y_col = st.selectbox("Value", numeric_cols, index=0 if numeric_cols else None)
        if x_col and y_col:
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                color=x_col,
                title=f"Boxplot of {y_col} by {x_col}",
                template="plotly_white",
            )
            plot(fig)

    elif selected_method == "Donut Chart":
        group_col = st.selectbox("Category", category_cols, index=0 if category_cols else None)
        if group_col:
            fig = px.pie(
                df,
                names=group_col,
                hole=0.45,
                title=f"Donut chart of {group_col}",
                template="plotly_white",
            )
            fig.update_traces(textinfo="percent+label")
            plot(fig)

    elif selected_method == "Violin Plot":
        x_col = st.selectbox("Category", category_cols, index=0 if category_cols else None)
        y_col = st.selectbox("Value", numeric_cols, index=0 if numeric_cols else None)
        if x_col and y_col:
            fig = px.violin(
                df,
                x=x_col,
                y=y_col,
                color=x_col,
                box=True,
                title=f"Violin plot of {y_col} by {x_col}",
                template="plotly_white",
            )
            plot(fig)

    elif selected_method == "Line Plot":
        x_col = st.selectbox("X column", numeric_cols, index=0 if numeric_cols else None)
        y_col = st.selectbox("Y column", [col for col in numeric_cols if col != x_col], index=0 if len(numeric_cols) > 1 else None)
        if x_col and y_col:
            fig = px.line(
                df.sort_values(by=x_col),
                x=x_col,
                y=y_col,
                title=f"Line plot of {y_col} over {x_col}",
                template="plotly_white",
            )
            plot(fig)

    elif selected_method == "Bar Plot":
        x_col = st.selectbox("Category", category_cols, index=0 if category_cols else None)
        y_col = st.selectbox("Value", [None] + numeric_cols, format_func=lambda x: x if x else "Count")
        agg = st.selectbox("Aggregation", ["count", "mean", "sum"], index=0)
        if x_col:
            if y_col:
                if agg == "count":
                    fig = px.histogram(
                        df,
                        x=x_col,
                        color=y_col,
                        barmode="group",
                        title=f"Bar plot count of {x_col} by {y_col}",
                        template="plotly_white",
                    )
                else:
                    agg_df = df.groupby(x_col)[y_col].agg(agg).reset_index()
                    fig = px.bar(
                        agg_df,
                        x=x_col,
                        y=y_col,
                        title=f"Bar plot of {agg} {y_col} by {x_col}",
                        template="plotly_white",
                    )
            else:
                fig = px.histogram(
                    df,
                    x=x_col,
                    title=f"Count of {x_col}",
                    template="plotly_white",
                )
            plot(fig)

    elif selected_method == "Count Plot":
        x_col = st.selectbox("Category", category_cols, index=0 if category_cols else None)
        color_col = st.selectbox("Split by", [None] + category_cols, format_func=lambda x: x if x else "None")
        if x_col:
            fig = px.histogram(
                df,
                x=x_col,
                color=color_col if color_col else None,
                title=f"Count plot for {x_col}",
                template="plotly_white",
            )
            plot(fig)


def render_explainable_ai():
    st.markdown("## Explainable AI Suite")
    st.markdown("<p class='highlight-pill'>Understand why churn happens with global and local model explanations.</p>", unsafe_allow_html=True)

    df = get_active_dataset()
    if df is None:
        st.warning("No dataset loaded. Please upload or load a dataset on the Upload page first.")
        return
    if shap is None:
        st.warning("SHAP is not installed in this environment. Install the requirements to enable explainability.")
        return
    if "training_results" not in st.session_state:
        try:
            with st.spinner("Training explainability model on the current dataset..."):
                st.session_state["training_results"] = train_models(df)
        except Exception as error:
            st.error(f"Cannot run explainability because model training failed: {error}")
            return

    results = st.session_state["training_results"]
    best = get_best_model(results)
    if best is None:
        st.warning("No trained model found. Upload a dataset and retry this page after training completes.")
        return

    st.markdown(f"### Model selected: **{best['name']}**")
    X = prepare_features(df)
    model = best["model"]

    if hasattr(model, "predict_proba"):
        try:
            X_values = X.values.astype(float)
            background = X_values[np.random.choice(X_values.shape[0], min(200, X_values.shape[0])), :]
            try:
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)
            except Exception:
                probability_fn = lambda data: model.predict_proba(np.asarray(data))[:, 1]
                explainer = shap.Explainer(probability_fn, background)
                shap_values = explainer(X_values[:200])

            st.markdown("### Global feature influence")
            shap.plots.bar(shap_values, show=False)
            st.pyplot(plt.gcf(), bbox_inches="tight")
            plt.clf()

            sample_id = st.selectbox("Choose a customer for local explanation", df.index[:20])
            selected = df.loc[[sample_id]]
            selected_features = prepare_features(selected.drop(columns=["churn"]), reference_columns=X.columns)
            local_values = explainer(selected_features.values.astype(float))

            st.markdown("### Local explanation for selected customer")
            shap.plots.waterfall(local_values[0], show=False)
            st.pyplot(plt.gcf(), bbox_inches="tight")
            plt.clf()
        except Exception as error:
            st.error(f"Explainability analysis failed: {error}")
    else:
        st.error("Selected model does not support SHAP explanations directly.")


def render_prediction():
    st.markdown("## Prediction & Deployment")
    st.markdown("<p class='highlight-pill'>Use the AI engine to predict churn probability and receive suggested retention actions.</p>", unsafe_allow_html=True)

    model = load_active_model()
    if model is None:
        st.error("No predictive model available. Upload a dataset on the Upload page and retry.")
        return

    with st.form(key="predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
            monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=65.0)
            total_charges = st.number_input("Total charges", min_value=0.0, value=720.0)
            support_calls = st.number_input("Support calls", min_value=0, max_value=30, value=1)
        with col2:
            contract = st.selectbox("Contract type", CONTRACT_OPTIONS)
            payment_method = st.selectbox("Payment method", PAYMENT_METHODS)
            internet_service = st.selectbox("Internet service", INTERNET_SERVICES)
            tech_support = st.selectbox("Tech support", BINARY_OPTIONS)
            online_security = st.selectbox("Online security", BINARY_OPTIONS)

        submit = st.form_submit_button("Predict churn risk")

    if submit:
        input_df = pd.DataFrame({
            "customer_id": [0],
            "tenure": [tenure],
            "monthly_charges": [monthly_charges],
            "total_charges": [total_charges],
            "contract": [contract],
            "payment_method": [payment_method],
            "internet_service": [internet_service],
            "tech_support": [tech_support],
            "online_security": [online_security],
            "support_calls": [support_calls],
        })

        prediction, probability = predict_with_model(model, input_df)
        label = "Churn Risk" if prediction == 1 else "Retention Favorable"
        color = "#fb7185" if prediction == 1 else "#22c55e"

        st.markdown(f"<div class='glass-card'><h3>{label}</h3><p>Probability of churn: <strong>{probability:.1%}</strong></p></div>", unsafe_allow_html=True)
        st.plotly_chart(risk_gauge(probability), use_container_width=True)

        if probability > 0.65:
            st.error("Recommendation: Offer a retention package and contact customer success immediately.")
        elif probability > 0.35:
            st.warning("Recommendation: Monitor usage trends and offer targeted engagement.")
        else:
            st.success("Recommendation: Customer is stable; maintain service quality.")


def render_sidebar():
    st.sidebar.markdown(
        f"""
        <div class='sidebar-brand'>
            <div class='brand-icon'>{APP_ICON}</div>
            <div>
                <h2>{APP_TITLE}</h2>
                <p>Enterprise customer churn intelligence.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = option_menu(
        menu_title=None,
        options=PAGE_OPTIONS,
        icons=[
            "house",
            "cloud-upload",
            "bar-chart-line",
            "lightbulb",
            "robot",
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "color": "#d1d5db"},
            "nav-link-selected": {"background-color": "rgba(59,130,246,0.18)", "color": "#fff"},
        },
    )

    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    user = st.session_state.get("current_user", "Analyst")
    st.sidebar.markdown(
        f"""
        <div class='sidebar-footer'>
            <div class='profile-avatar'>{user[:2].upper()}</div>
            <div>
                <strong>{user}</strong><br/>
                <span>Secure workspace access</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("Logout"):
        logout()
        st.experimental_rerun()

    return selected_page


PAGE_MODULES = {
    "Landing": render_landing,
    "Upload": render_upload,
    "EDA": render_eda,
    "Explainable AI": render_explainable_ai,
    "Prediction": render_prediction,
}


def render_app():
    if not st.session_state.get("authenticated"):
        render_sign_in()
        return

    page = render_sidebar()
    PAGE_MODULES.get(page, render_landing)()


render_app()
