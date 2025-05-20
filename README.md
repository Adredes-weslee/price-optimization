# 📦 BCG RISE 2.0 Capstone Project: Customer Segmentation & Price Optimization for CS Tay

> Data-driven commercial strategy for a leading frozen food distributor in Singapore.

---

## 🎯 Objective

This capstone project for the BCG RISE program involved building an end-to-end data science pipeline to support pricing and customer targeting strategies for **CS Tay**, a frozen food manufacturer and distributor.

The project was divided into three key phases:
- **Customer Segmentation** using RFM scoring and unsupervised clustering
- **Price Elasticity Modeling** to quantify demand response
- **Revenue Optimization** using linear programming with Gurobi

---

## 📂 Project Structure

| File/Folder                     | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| `Customer_segmentation.md`     | Methodology for RFM scoring, feature engineering, and clustering           |
| `Price_elasticity.md`          | Cross-price elasticity modeling and demand estimation                      |
| `Data_preprocessing.md`        | Data cleaning, joining, and exploratory data analysis (EDA)                |
| `Group7 - CS TAY.pptx`         | Final presentation deck submitted to BCG RISE program                      |
| `*.html` notebooks             | Rendered reports for reproducibility and evaluation                        |

---

## 🧪 Methodology

### 1. 🧮 Customer Segmentation

- Performed RFM scoring on 2023 transaction data
- Clustered customers using K-Means after PCA dimensionality reduction
- Identified key personas: **Champions**, **Loyalists**, **Hibernating**, **Lost**, **New**

### 2. 📉 Price Elasticity Analysis

- Modeled demand sensitivity using log-log regression across SKUs
- Estimated **own-price** and **cross-price** elasticities to uncover substitution effects
- Applied regression filtering to isolate robust elasticity estimates

### 3. 🧠 Optimization

- Formulated a constrained revenue maximization problem
- Used **Gurobi** to optimize pricing across top-selling SKUs
- Simulated revenue impact of proposed price changes under elasticity estimates

---

## 📊 Key Findings

| Segment        | Insights                                                                 |
|----------------|--------------------------------------------------------------------------|
| Supermarket    | Larger basket sizes, high repeat purchase rate                          |
| Retail         | More price-sensitive, higher SKU substitution behavior                  |
| New Customers  | High potential segment with low conversion, needs onboarding incentives |

- Pricing optimizations could unlock **~SGD 4M in additional annual revenue**
- Champions willing to absorb moderate price increases, while Retail customers showed substitution behavior across brands

---

## 🧠 Tools & Stack

- **Python** (Pandas, Scikit-learn, Matplotlib, Seaborn)
- **Gurobi** for optimization modeling
- **PowerPoint** for stakeholder presentation
- Markdown & HTML notebooks for reproducible reports

---

## 📌 Outcome

Delivered actionable insights to CS Tay’s management, including:
- Strategic pricing recommendations based on elasticity
- Segmentation-driven marketing playbooks
- Revenue simulation dashboards

Presented findings to a panel of BCG consultants and company stakeholders.

---

## 🏅 Program

**📈 BCG RISE 2.0 — Business & Data Analytics Graduate Programme**  
Wave 15 · Group 7 · 2024

---
