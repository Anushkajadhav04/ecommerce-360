"""
E-COMMERCE 360° — Streamlit App

3-page interactive dashboard:
  Page 1: Executive Overview
  Page 2: Customer Behaviour (RFM)
  Page 3: Churn Prediction Engine

Run with:
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib, os, warnings

warnings.filterwarnings("ignore")

#  Page config
st.set_page_config(
    page_title="E-Commerce 360° Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Paths 
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")

# Load data (cached) 
@st.cache_data
def load_orders():
    return pd.read_csv(os.path.join(DATA, "clean_ecommerce.csv"), parse_dates=["date"])

@st.cache_data
def load_rfm():
    return pd.read_csv(os.path.join(DATA, "rfm_features.csv"))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(MODELS, "churn_model.pkl"))

orders = load_orders()
rfm    = load_rfm()
artifact = load_model()

# Sold orders only
sold = orders[(orders["is_cancelled"]==0) & (orders["revenue"]>0)].copy()

PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2",
           "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]

# SIDEBAR NAV

st.sidebar.title("🛒 E-Commerce 360°")
st.sidebar.caption("Amazon India Sales Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Executive Overview", "👥 Customer Behaviour", "🎯 Churn Prediction Engine"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset**: Amazon India Fashion\n\n"
    f"**Orders**: {len(orders):,}\n\n"
    f"**Date range**: {orders['date'].min().date()} → {orders['date'].max().date()}\n\n"
    f"**Categories**: {orders['category'].nunique()}"
)

# PAGE 1 — EXECUTIVE OVERVIEW

if page == "📊 Executive Overview":
    st.title("📊 Executive Overview")
    st.caption("Revenue, orders, and performance KPIs at a glance")

    #  KPI row 
    total_rev   = sold["revenue"].sum()
    total_orders= sold["order_id"].nunique()
    aov         = sold["revenue"].mean()
    cancel_rate = orders["is_cancelled"].mean() * 100
    churn_rate  = rfm["churned"].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total Revenue",     f"₹{total_rev/1e6:.2f}M")
    c2.metric("📦 Total Orders",      f"{total_orders:,}")
    c3.metric("🛍️ Avg Order Value",   f"₹{aov:,.0f}")
    c4.metric("❌ Cancellation Rate", f"{cancel_rate:.1f}%")
    c5.metric("📉 Churn Rate",        f"{churn_rate:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # Monthly revenue
    with col1:
        st.subheader("Monthly Revenue Trend")
        monthly = sold.groupby("month")["revenue"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(monthly["month"], monthly["revenue"], marker="o",
                color=PALETTE[0], linewidth=2.5)
        ax.fill_between(monthly["month"], monthly["revenue"],
                        alpha=0.15, color=PALETTE[0])
        ax.set_xticklabels(monthly["month"], rotation=30, fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"₹{x/1e6:.1f}M"))
        ax.set_ylabel("Revenue")
        sns.despine(ax=ax)
        st.pyplot(fig)

    # Revenue by category
    with col2:
        st.subheader("Revenue by Category (Top 8)")
        cat_rev = (sold.groupby("category")["revenue"]
                       .sum().sort_values(ascending=False).head(8))
        fig, ax = plt.subplots(figsize=(6,3))
        ax.barh(cat_rev.index[::-1], cat_rev.values[::-1], color=PALETTE)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"₹{x/1e6:.1f}M"))
        sns.despine(ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    # Orders by state
    with col3:
        st.subheader("Orders by State (Top 10)")
        state_o = (sold.groupby("ship_state")["order_id"]
                       .nunique().sort_values(ascending=False).head(10))
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(state_o.index, state_o.values, color=PALETTE)
        ax.set_xticklabels(state_o.index, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("Orders")
        sns.despine(ax=ax)
        st.pyplot(fig)

    # Fulfilment split
    with col4:
        st.subheader("Fulfilment Method Split")
        ful = sold.groupby("fulfilment")["revenue"].sum()
        fig, ax = plt.subplots(figsize=(5,3))
        ax.pie(ful.values, labels=ful.index, autopct="%1.1f%%",
               colors=PALETTE, startangle=140)
        st.pyplot(fig)

# PAGE 2 — CUSTOMER BEHAVIOUR

elif page == "👥 Customer Behaviour":
    st.title("👥 Customer Behaviour & RFM Segmentation")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👤 Total Customers",    f"{len(rfm):,}")
    c2.metric("🔁 Repeat Buyers",
              f"{(rfm['frequency']>1).sum():,}",
              delta=f"{(rfm['frequency']>1).mean()*100:.1f}% of total")
    c3.metric("💎 Champions",
              f"{(rfm['segment']=='Champions').sum():,}")
    c4.metric("⚠️ At-Risk Customers",
              f"{(rfm['segment']=='At Risk').sum():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # RFM Segment distribution
    with col1:
        st.subheader("Customer Segments (RFM)")
        seg = rfm["segment"].value_counts()
        fig, ax = plt.subplots(figsize=(6,3))
        bars = ax.bar(seg.index, seg.values, color=PALETTE)
        ax.set_ylabel("Customers")
        ax.set_xticklabels(seg.index, rotation=25, ha="right", fontsize=9)
        for bar, v in zip(bars, seg.values):
            ax.text(bar.get_x()+bar.get_width()/2, v+10, str(v),
                    ha="center", fontsize=8)
        sns.despine(ax=ax)
        st.pyplot(fig)

    # Monetary by segment
    with col2:
        st.subheader("Avg Spend by Segment")
        seg_mon = rfm.groupby("segment")["monetary"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.barh(seg_mon.index[::-1], seg_mon.values[::-1], color=PALETTE)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"₹{x:,.0f}"))
        sns.despine(ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    # RFM scatter
    with col3:
        st.subheader("Recency vs Monetary (by Segment)")
        fig, ax = plt.subplots(figsize=(6,4))
        for i, seg in enumerate(rfm["segment"].unique()):
            sub = rfm[rfm["segment"]==seg]
            ax.scatter(sub["recency"], sub["monetary"],
                       label=seg, alpha=0.5, s=15,
                       color=PALETTE[i % len(PALETTE)])
        ax.set_xlabel("Recency (days)")
        ax.set_ylabel("Monetary (₹)")
        ax.legend(fontsize=7)
        sns.despine(ax=ax)
        st.pyplot(fig)

    # Frequency histogram
    with col4:
        st.subheader("Order Frequency Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(rfm["frequency"].clip(upper=10), bins=10,
                color=PALETTE[1], edgecolor="white")
        ax.set_xlabel("Number of Orders")
        ax.set_ylabel("Customers")
        sns.despine(ax=ax)
        st.pyplot(fig)

    # Data table
    st.subheader("Customer Segment Table (top 200 by spend)")
    top_customers = (rfm.sort_values("monetary", ascending=False)
                        .head(200)
                        [["customer_id","segment","recency","frequency",
                          "monetary","rfm_score","churned"]])
    st.dataframe(top_customers, use_container_width=True)

# PAGE 3 — CHURN PREDICTION ENGINE

elif page == "🎯 Churn Prediction Engine":
    st.title("🎯 Churn Prediction Engine")
    st.caption(f"Model: **{artifact['model_name']}** | "
               f"ROC-AUC: **{artifact['metrics']['roc_auc']:.3f}** | "
               f"F1: **{artifact['metrics']['f1']:.3f}**")

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.subheader("Enter Customer Data")

        recency        = st.slider("Recency (days since last order)", 1, 180, 30)
        frequency      = st.slider("Order Frequency (# orders)",      1, 20,   2)
        monetary       = st.number_input("Total Spend (₹)",           100.0, 50000.0, 2000.0, step=100.0)
        avg_order_val  = st.number_input("Avg Order Value (₹)",       100.0, 10000.0, 700.0,  step=50.0)
        avg_qty        = st.slider("Avg Items per Order",              1.0,   10.0,   1.5,     step=0.5)
        total_qty      = st.slider("Total Items Bought",               1,     100,    4)
        num_categories = st.slider("No. of Product Categories",        1,     10,     2)
        r_score        = st.selectbox("Recency Score (1=old, 5=recent)", [1,2,3,4,5], index=2)
        f_score        = st.selectbox("Frequency Score",                 [1,2,3,4,5], index=1)
        m_score        = st.selectbox("Monetary Score",                  [1,2,3,4,5], index=2)
        state_enc      = st.number_input("State Encoded (0–35)",         0, 35, 10, step=1)

    with col_result:
        st.subheader("Prediction")

        if st.button("🔮  Predict Churn", type="primary", use_container_width=True):
            input_data = pd.DataFrame([{
                "recency"        : recency,
                "frequency"      : frequency,
                "monetary"       : monetary,
                "avg_order_value": avg_order_val,
                "avg_qty"        : avg_qty,
                "total_qty"      : total_qty,
                "num_categories" : num_categories,
                "r_score"        : r_score,
                "f_score"        : f_score,
                "m_score"        : m_score,
                "state_enc"      : state_enc,
            }])

            model = artifact["model"]

            if artifact["model_name"] == "Logistic Regression":
                scaler = artifact["scaler"]
                X_in   = scaler.transform(input_data)
            else:
                X_in = input_data.values

            prob   = model.predict_proba(X_in)[0][1]
            label  = "🔴 LIKELY TO CHURN" if prob >= 0.5 else "🟢 LIKELY TO STAY"
            colour = "#FF4444" if prob >= 0.5 else "#00CC44"

            st.markdown(f"""
            <div style='background:{colour}22; border-left:5px solid {colour};
                        padding:16px; border-radius:8px; margin-bottom:12px;'>
                <h2 style='color:{colour}; margin:0'>{label}</h2>
                <h3 style='margin:4px 0'>Churn Probability: {prob*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig, ax = plt.subplots(figsize=(4,2))
            ax.barh(["Risk"], [prob],        color=colour,      height=0.4)
            ax.barh(["Risk"], [1-prob], left=[prob],
                    color="#EEEEEE", height=0.4)
            ax.set_xlim(0,1)
            ax.set_xlabel("Churn Probability")
            ax.axvline(0.5, color="black", linewidth=1.5, linestyle="--")
            ax.set_yticks([])
            sns.despine(ax=ax, left=True)
            st.pyplot(fig)

            # SHAP waterfall for this customer
            st.markdown("#### Why this prediction?")
            try:
                explainer = artifact["explainer"]
                if artifact["model_name"] == "Logistic Regression":
                    shap_vals = explainer.shap_values(X_in)
                else:
                    shap_vals = explainer.shap_values(input_data)
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]
                    elif shap_vals.ndim == 3:
                        shap_vals = shap_vals[:, :, 1]

                feat_names = artifact["features"]
                shap_series= pd.Series(shap_vals[0], index=feat_names).sort_values()

                fig2, ax2 = plt.subplots(figsize=(5,4))
                colors = ["#FF4444" if v>0 else "#4C72B0" for v in shap_series.values]
                ax2.barh(shap_series.index, shap_series.values, color=colors)
                ax2.axvline(0, color="black", linewidth=0.8)
                ax2.set_xlabel("SHAP Value (impact on churn probability)")
                ax2.set_title("Feature Impact (Red=↑ Churn, Blue=↓ Churn)", fontsize=9)
                sns.despine(ax=ax2)
                st.pyplot(fig2)
            except Exception as e:
                st.info("SHAP plot not available for this prediction.")
                st.caption(str(e))

    #  Churn overview section
    st.markdown("---")
    st.subheader("Portfolio-Level Churn Analysis")

    col_a, col_b = st.columns(2)
    with col_a:
        churn_by_seg = rfm.groupby("segment")["churned"].mean() * 100
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(churn_by_seg.index, churn_by_seg.values,
               color=PALETTE, edgecolor="white")
        ax.set_ylabel("Churn Rate %")
        ax.set_title("Churn Rate by Segment", fontsize=12, fontweight="bold")
        ax.set_xticklabels(churn_by_seg.index, rotation=25, ha="right")
        sns.despine(ax=ax)
        st.pyplot(fig)

    with col_b:
        at_risk = (rfm[(rfm["churned"]==1)]
                   .sort_values("monetary", ascending=False)
                   .head(10)
                   [["customer_id","segment","recency","frequency","monetary"]])
        st.markdown("**Top 10 At-Risk Customers (by Spend)**")
        st.dataframe(at_risk, use_container_width=True, height=260)
