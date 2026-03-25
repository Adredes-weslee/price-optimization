import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add the parent directory to path to import from src
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Add the streamlit directory to path
streamlit_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(streamlit_dir)

from src import config, utils  # noqa: F401
from utils import st_utils


def load_sample_data():
    """Load committed sample artifacts into session state."""
    raw_data_path = Path(parent_dir) / "data" / "raw" / "sales_data.csv"
    if raw_data_path.exists() and st.session_state.raw_data is None:
        try:
            st.session_state.raw_data = pd.read_csv(raw_data_path)
        except Exception as exc:
            st.error(f"Error loading raw sample data: {exc}")

    processed_data_path = Path(parent_dir) / "data" / "processed" / "aggregated_df.csv"
    if processed_data_path.exists() and st.session_state.processed_data is None:
        try:
            st.session_state.processed_data = pd.read_csv(processed_data_path)
        except Exception as exc:
            st.error(f"Error loading processed sample data: {exc}")

    segmentation_path = Path(parent_dir) / "data" / "segmentation" / "customer_segmentation_df.csv"
    if segmentation_path.exists() and st.session_state.segmentation_results is None:
        try:
            st.session_state.segmentation_results = pd.read_csv(segmentation_path)
        except Exception as exc:
            st.error(f"Error loading segmentation results: {exc}")

    elasticity_candidates = [
        Path(parent_dir) / "data" / "optimization" / "price_elasticities_df.csv",
        Path(parent_dir) / "data" / "optimization" / "price_elasticities_calculated.csv",
    ]
    elasticity_path = next((path for path in elasticity_candidates if path.exists()), None)
    if elasticity_path is not None and st.session_state.elasticity_results is None:
        try:
            st.session_state.elasticity_results = pd.read_csv(elasticity_path)
        except Exception as exc:
            st.error(f"Error loading elasticity results: {exc}")

    optimization_path = Path(parent_dir) / "data" / "optimization" / "revenue_optimization_results.csv"
    if optimization_path.exists() and st.session_state.optimization_results is None:
        try:
            st.session_state.optimization_results = pd.read_csv(optimization_path)
        except Exception as exc:
            st.error(f"Error loading optimization results: {exc}")


def fmt_count(value):
    return f"{value:,}" if value is not None else "Pending"


st.set_page_config(
    page_title="Retail Price Optimizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .app-kicker {
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-size: 0.82rem;
        font-weight: 700;
        color: #55727c;
        margin-bottom: 0.35rem;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #4f6570;
        max-width: 48rem;
        margin-bottom: 1.2rem;
    }
    .workflow-callout {
        border: 1px solid rgba(53, 116, 149, 0.18);
        background: #f6fbfd;
        border-radius: 0.9rem;
        padding: 1rem 1.1rem;
        margin: 0.9rem 0 1.1rem 0;
    }
    .workflow-callout strong {
        display: block;
        margin-bottom: 0.25rem;
        color: #163640;
    }
    section[data-testid="stSidebar"] .css-pkbazv,
    section[data-testid="stSidebar"] .css-17lntkn,
    section[data-testid="stSidebar"] span.css-10trblm {
        text-transform: uppercase !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class="footer">Committed sample artifacts loaded for demo mode</div>
    """,
    unsafe_allow_html=True,
)

st_utils.initialize_session_state()
load_sample_data()

raw_rows = (
    len(st.session_state.raw_data)
    if st.session_state.raw_data is not None
    else None
)
processed_rows = (
    len(st.session_state.processed_data)
    if st.session_state.processed_data is not None
    else None
)
customer_count = (
    len(st.session_state.segmentation_results)
    if st.session_state.segmentation_results is not None
    else None
)
optimization_rows = (
    len(st.session_state.optimization_results)
    if st.session_state.optimization_results is not None
    else None
)

st.markdown(
    '<div class="app-kicker">Commercial strategy / pricing workflow</div>',
    unsafe_allow_html=True,
)
st.title("Retail Price Optimization Dashboard")
st.markdown(
    '<div class="app-subtitle">A staged pricing workflow that turns committed sales transactions into customer segments, elasticity estimates, and optimization-ready price recommendations.</div>',
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Transactions", fmt_count(raw_rows))
with metric_cols[1]:
    st.metric("Prepared rows", fmt_count(processed_rows))
with metric_cols[2]:
    st.metric("Customers segmented", fmt_count(customer_count))
with metric_cols[3]:
    st.metric("Optimized recommendations", fmt_count(optimization_rows))

st.markdown(
    """
    <div class="workflow-callout">
      <strong>Start here</strong>
      Move through the sidebar in order: <em>Customer Segmentation</em>, <em>Price Elasticity</em>, then <em>Revenue Optimization</em>. The committed CSV outputs make the first render readable immediately, even before rerunning the full pipeline.
    </div>
    """,
    unsafe_allow_html=True,
)

stage_cols = st.columns(3)
with stage_cols[0]:
    st.subheader("1. Segment the customer base")
    st.write(
        "Begin by reviewing the RFM-driven segments so the rest of the dashboard reads like a commercial decision flow instead of a loose collection of charts."
    )
with stage_cols[1]:
    st.subheader("2. Inspect price sensitivity")
    st.write(
        "Use the elasticity stage to understand which products are price-sensitive and which categories have more room for controlled price movement."
    )
with stage_cols[2]:
    st.subheader("3. Compare optimized price moves")
    st.write(
        "Once the elasticity outputs look credible, move into revenue optimization and the simulator to test the recommended price changes."
    )

st.markdown("---")
st.subheader("What is already available in this demo?")
st.write(
    "The landing page is wired to committed sample artifacts, so the app opens with the current workflow state visible instead of forcing a preprocessing step just to understand what the project does."
)

st.sidebar.header("Workflow Status")
st.sidebar.caption(
    "Green signals mean the committed sample artifact for that stage is already available locally."
)
if st.session_state.raw_data is not None:
    st.sidebar.success(f"Transactions loaded ({len(st.session_state.raw_data)} rows)")
else:
    st.sidebar.warning("Transactions not loaded")

if st.session_state.processed_data is not None:
    st.sidebar.success(
        f"Prepared pricing table ready ({len(st.session_state.processed_data)} rows)"
    )

if st.session_state.segmentation_results is not None:
    st.sidebar.success(
        f"Customer segments ready ({len(st.session_state.segmentation_results)} customers)"
    )

if st.session_state.elasticity_results is not None:
    st.sidebar.success("Elasticity estimates loaded")

if st.session_state.optimization_results is not None:
    st.sidebar.success("Optimization recommendations loaded")
