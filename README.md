# Advanced Retail Price Optimization System

Retail price-optimization pipeline that turns transaction CSVs into customer segments, elasticity estimates, and optimized price recommendations, with a Streamlit dashboard on top.

The code is designed for anonymized customer codes and SKU-level transactions. `Customer Name` is dropped if present, and segmentation uses `Customer Code` as the customer key.

<!-- README_SURFACE_START -->
![Python](https://img.shields.io/badge/Python-Optimization_Engine-3776AB?style=flat-square&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![Gurobi](https://img.shields.io/badge/Gurobi-Required-EE3524?style=flat-square)

[![Portfolio Article](https://img.shields.io/badge/Portfolio%20Article-102A43?style=flat-square)](https://adredes-weslee.github.io/data-science/pricing-strategy/business-analytics/commercial-strategy/2024/08/15/customer-segmentation-price-optimization.html) [![Live Demo](https://img.shields.io/badge/Live%20Demo-FF8B2B?style=flat-square)](https://adredes-weslee-price-optimization-streamlitapp-yxjoe3.streamlit.app/)
## Quickstart

```bash
pip install -r requirements.txt
python -m src.main
streamlit run streamlit/app.py
```

See [Setup and Run](#setup-and-run) for the full environment and verification path.

<!-- README_SURFACE_END -->

## Why This Repository Exists

- Given sales transactions with `Transaction Date`, `Sales Order No.`, `Customer Code`, `Inventory Code`, `Qty`, and `Total Base Amt`, the repo tries to answer who the customers are, how price-sensitive the top products are, and what price changes maximize revenue.
- The implemented decision surface is price multipliers plus simple same-category consistency constraints, not a broader pricing system with external data feeds or business-rule engines.

## Architecture at a Glance

- The code suggests an artifact-driven pipeline: each stage reads and writes CSVs between `data/raw`, `data/processed`, `data/segmentation`, and `data/optimization`.
- The Streamlit app uses `st.session_state` as the data handoff between pages, and the page files live under `streamlit/pages/`.
- The repo already contains sample raw and generated CSVs, including `sales_data.csv`, `aggregated_df.csv`, `customer_segmentation_df.csv`, `price_elasticities_calculated.csv`, and `revenue_optimization_results.csv`.

## Repository Layout

- `data/`
- `src/`
- `streamlit/`
- `.gitignore`
- `environment.yaml`
- `README.md`
- `requirements.txt`

## Setup and Run

1. Use `environment.yaml` or `requirements.txt`; the pinned stack includes Python 3.11, Streamlit 1.45.0, scikit-learn 1.6.1, statsmodels 0.14.4, and Gurobi support via `gurobipy`.
2. Put the raw file at `data/raw/sales_data.csv`; the committed sample shows the expected transaction, customer, SKU, quantity, and revenue fields.
3. Use module form for the CLI steps, for example `python -m src.main` and `python -m src.data_preprocessing`. The direct `python src/main.py` form fails here because `src/main.py` uses relative imports.
4. Start the dashboard with `streamlit run streamlit/app.py`; optimization still requires a valid Gurobi license via `GRB_LICENSE_FILE`.

## Core Workflows

- Preprocessing loads raw data, reconciles customer and inventory fields, fills missing values, aggregates transactions, engineers `Year`/`Month`/`Day`, `Customer Category Broad`, and `Item Category`, then writes `aggregated_df.csv` and `no_customer_name_agg_df.csv`.
- Segmentation computes recency, frequency, monetary value, and quantity per customer, scores each dimension into deciles, assigns RFM labels such as `Champions` and `Lost Customers`, and clusters with `RobustScaler` + KMeans. A second pass runs if one cluster dominates.
- Elasticity defaults to `SUPERMARKET` and `RETAIL`, picks the top 15 SKUs per item category, fits OLS log-log own-price models and pairwise cross-price models with month dummies, and keeps significant cross effects.
- Optimization reads the elasticity CSV, builds Gurobi price-multiplier variables, applies a linearized demand approximation plus same-category price-difference constraints, and saves `revenue_optimization_results.csv`.
- The price simulator currently only implements the `Price Bounds` scenario; `Product Selection` is explicitly not implemented and the other branches are placeholders.

## Known Limitations

- The repository does not implement CLV integration, A/B testing, Kafka, cloud deployment, API integration, or automated validation tests.
- The implementation uses plain OLS with `pvalues` and `rsquared`, but it does not include Durbin-Watson, Breusch-Pagan, RESET, or cross-validation checks.
- The dashboard has no preprocessing page, so preprocessing is CLI-only in `src/data_preprocessing.py` even though the home page copy suggests a full end-to-end UI.
- The elasticity pages `Minimum Data Points` slider is not wired into the model; the estimator still uses hardcoded `>10` checks.
- The optimizer does not implement the documented exact power-law demand formula or a minimum-margin constraint, and there is no cost input in the model.
- Filename drift exists between loaders and outputs: `streamlit/app.py` looks for `price_elasticities_df.csv`, while `src/config.py` writes `price_elasticities_calculated.csv`; `st_utils.py` also references stale names like `customer_segments.csv` and `optimized_prices.csv`.
- The modules use relative imports, so module form like `python -m src.main` is required.
- There is no `tests/` directory, `LICENSE` file, notebook set, or referenced `*_original.md` source docs in this snapshot.
- Some older documentation still uses page names that do not match the current lowercase `streamlit/pages/` filenames.
