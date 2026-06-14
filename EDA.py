"""
E-COMMERCE 360° — Data Cleaning + EDA
Dataset: Amazon India Sales (7 CSV files)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os, warnings

warnings.filterwarnings("ignore")

# colour palette (professional, works in dark & light) ──
PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2",
           "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]
sns.set_theme(style="whitegrid", palette=PALETTE)

#  paths ─
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data")
PLOTS  = os.path.join(BASE, "plots")
os.makedirs(DATA,  exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

# STEP 1 — LOAD ALL CSVs
print(" ")
print("  STEP 1 — Loading CSVs")
print(" ")

# Primary orders dataset
orders = pd.read_csv("Amazon_Sale_Report.csv", low_memory=False)
print(f"Amazon Sale Report  : {orders.shape}")

# International sales
intl   = pd.read_csv("International_sale_Report.csv", low_memory=False)
print(f"International Sales : {intl.shape}")

# Product / SKU pricing tables
pl     = pd.read_csv("P__L_March_2021.csv",   low_memory=False)
may    = pd.read_csv("May-2022.csv",           low_memory=False)
sale   = pd.read_csv("Sale_Report.csv",        low_memory=False)
cloud  = pd.read_csv("Cloud_Warehouse_Compersion_Chart.csv", low_memory=False)
expense= pd.read_csv("Expense_IIGF.csv",       low_memory=False)
print("All files loaded ✓")

# STEP 2 — CLEAN MAIN ORDERS TABLE
print(" ")
print("  STEP 2 — Cleaning orders")
print(" ")

df = orders.copy()

# Rename columns to snake_case
df.columns = (df.columns
              .str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("-", "_"))

print("Columns:", df.columns.tolist())

# Drop useless columns
df.drop(columns=["index","unnamed:_22","promotion_ids","asin"], errors="ignore", inplace=True)

# Parse date
df["date"] = pd.to_datetime(df["date"], format="%m-%d-%y", errors="coerce")

# Drop rows with no date or no order id
df.dropna(subset=["date","order_id"], inplace=True)

# Fill amount nulls with 0 (cancelled orders)
df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

# Qty
df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)

# Clean state names
df["ship_state"] = df["ship_state"].str.strip().str.upper()

# Clean category
df["category"] = df["category"].str.strip().str.title()

# Status → keep only meaningful statuses; create delivered flag
delivered_statuses = ["Shipped - Delivered to Buyer"]
df["is_delivered"] = df["status"].isin(delivered_statuses).astype(int)

# Cancelled flag
df["is_cancelled"] = (df["status"] == "Cancelled").astype(int)

# Revenue = amount * qty  (amount is per-unit in this dataset)
df["revenue"] = df["amount"] * df["qty"]

# Month / week columns (useful for time-series)
df["month"]   = df["date"].dt.to_period("M").astype(str)
df["week"]    = df["date"].dt.to_period("W").astype(str)
df["dow"]     = df["date"].dt.day_name()

print(f"\nShape after cleaning : {df.shape}")
print(f"Date range          : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Null counts:\n{df.isnull().sum()[df.isnull().sum()>0]}")

# STEP 3 — RFM FEATURE ENGINEERING (Customer-level)
print(" ")
print("  STEP 3 — RFM Feature Engineering")
print(" ")

# In this dataset 'order_id' is the unique transaction; we'll use
# 'order_id' as proxy for customer (Amazon doesn't expose customer IDs).
# We extract customer-level signals from order patterns.

# We'll define a "customer" by their ship-city + ship-state combo
# (closest proxy available without a customer_id column)
df["customer_id"] = df["ship_city"].str.strip().str.upper() + "_" + df["ship_state"]

# Reference date = day after last order (simulates "today")
REF_DATE = df["date"].max() + pd.Timedelta(days=1)

# Only consider non-cancelled orders for RFM
orders_rfm = df[df["is_cancelled"] == 0].copy()

rfm = (orders_rfm
       .groupby("customer_id")
       .agg(
           last_order_date  = ("date",    "max"),
           frequency        = ("order_id","nunique"),
           monetary         = ("revenue", "sum"),
           avg_order_value  = ("revenue", "mean"),
           avg_qty          = ("qty",     "mean"),
           total_qty        = ("qty",     "sum"),
           num_categories   = ("category","nunique"),
           state            = ("ship_state","first"),
       )
       .reset_index()
)

rfm["recency"] = (REF_DATE - rfm["last_order_date"]).dt.days
rfm.drop(columns=["last_order_date"], inplace=True)

# CHURN LABEL 
# No purchase in last 60 days = churned  (dataset spans ~3 months,
# so 60 days is a meaningful threshold; adjust if your dataset grows)
CHURN_THRESHOLD = 60
rfm["churned"] = (rfm["recency"] > CHURN_THRESHOLD).astype(int)

# RFM Scores (1–5 quintile ranks)
rfm["r_score"] = pd.qcut(rfm["recency"],   q=5, labels=[5,4,3,2,1]).astype(int)
rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"),  q=5, labels=[1,2,3,4,5]).astype(int)
rfm["rfm_score"]= rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

# RFM Segment
def segment(row):
    if row["rfm_score"] >= 13:
        return "Champions"
    elif row["rfm_score"] >= 10:
        return "Loyal Customers"
    elif row["rfm_score"] >= 7:
        return "At Risk"
    elif row["rfm_score"] >= 4:
        return "Needs Attention"
    else:
        return "Lost"

rfm["segment"] = rfm.apply(segment, axis=1)

print(rfm["segment"].value_counts())
print(f"\nChurn rate : {rfm['churned'].mean()*100:.1f}%")
print(f"RFM table shape : {rfm.shape}")

# STEP 4 — SAVE CLEANED DATA
df.to_csv(os.path.join(DATA, "clean_ecommerce.csv"),  index=False)
rfm.to_csv(os.path.join(DATA, "rfm_features.csv"),    index=False)
print("\n✅  clean_ecommerce.csv  and  rfm_features.csv  saved to /data/")

# STEP 5 — 10 EDA CHARTS
print(" ")
print("  STEP 5 — Generating 10 EDA Charts")
print(" ")

def save(fig, name):
    path = os.path.join(PLOTS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {name}")

# Filter to non-cancelled, non-zero revenue for most charts
sold = df[(df["is_cancelled"]==0) & (df["revenue"]>0)].copy()

# Chart 1: Revenue by Category 
cat_rev = (sold.groupby("category")["revenue"]
               .sum()
               .sort_values(ascending=False)
               .head(10))

fig, ax = plt.subplots(figsize=(10,5))
bars = ax.barh(cat_rev.index[::-1], cat_rev.values[::-1], color=PALETTE)
ax.set_xlabel("Total Revenue (INR)")
ax.set_title("Top 10 Categories by Revenue", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{x/1e6:.1f}M"))
for bar, val in zip(bars, cat_rev.values[::-1]):
    ax.text(val*1.01, bar.get_y()+bar.get_height()/2,
            f"₹{val/1e6:.2f}M", va="center", fontsize=8)
save(fig, "eda_01_revenue_by_category.png")

# Chart 2: Orders by State (Top 12)
state_orders = (sold.groupby("ship_state")["order_id"]
                    .nunique()
                    .sort_values(ascending=False)
                    .head(12))

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(range(len(state_orders)), state_orders.values, color=PALETTE*2)
ax.set_xticks(range(len(state_orders)))
ax.set_xticklabels(state_orders.index, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Number of Orders")
ax.set_title("Top 12 States by Order Volume", fontsize=14, fontweight="bold")
save(fig, "eda_02_orders_by_state.png")

#Chart 3: Revenue Over Time (Monthly)
monthly_rev = sold.groupby("month")["revenue"].sum()

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(monthly_rev.index, monthly_rev.values, marker="o", color=PALETTE[0], linewidth=2)
ax.fill_between(monthly_rev.index, monthly_rev.values, alpha=0.15, color=PALETTE[0])
ax.set_ylabel("Revenue (INR)")
ax.set_title("Monthly Revenue Trend", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{x/1e6:.1f}M"))
ax.tick_params(axis="x", rotation=30)
save(fig, "eda_03_monthly_revenue.png")

# Chart 4: Order Status Breakdown
status_counts = df["status"].value_counts()

fig, ax = plt.subplots(figsize=(7,5))
wedges, texts, autotexts = ax.pie(
    status_counts.values,
    labels=status_counts.index,
    autopct="%1.1f%%",
    colors=PALETTE,
    startangle=140,
    pctdistance=0.82
)
for at in autotexts:
    at.set_fontsize(8)
ax.set_title("Order Status Distribution", fontsize=14, fontweight="bold")
save(fig, "eda_04_order_status_pie.png")

# Chart 5: Average Order Value by Category
cat_aov = (sold.groupby("category")["revenue"]
               .mean()
               .sort_values(ascending=False)
               .head(10))

fig, ax = plt.subplots(figsize=(10,5))
ax.barh(cat_aov.index[::-1], cat_aov.values[::-1], color=PALETTE[1])
ax.set_xlabel("Avg Order Value (INR)")
ax.set_title("Average Order Value by Category (Top 10)", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{x:,.0f}"))
save(fig, "eda_05_aov_by_category.png")

#Chart 6: Revenue by Fulfilment Method 
ful_rev = sold.groupby("fulfilment")["revenue"].sum()

fig, ax = plt.subplots(figsize=(6,4))
ax.bar(ful_rev.index, ful_rev.values, color=[PALETTE[0], PALETTE[2]])
ax.set_ylabel("Total Revenue (INR)")
ax.set_title("Revenue: Amazon vs Merchant Fulfilment", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{x/1e6:.1f}M"))
save(fig, "eda_06_revenue_by_fulfilment.png")

#Chart 7: B2B vs B2C Revenue 
b2b_rev = sold.groupby("b2b")["revenue"].sum()
b2b_rev.index = ["B2C", "B2B"]

fig, ax = plt.subplots(figsize=(6,4))
ax.bar(b2b_rev.index, b2b_rev.values, color=[PALETTE[0], PALETTE[1]])
ax.set_ylabel("Total Revenue (INR)")
ax.set_title("B2B vs B2C Revenue Split", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₹{x/1e6:.1f}M"))
save(fig, "eda_07_b2b_vs_b2c.png")

# Chart 8: Orders by Day of Week 
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_counts = sold.groupby("dow")["order_id"].nunique().reindex(dow_order)

fig, ax = plt.subplots(figsize=(8,4))
ax.bar(dow_counts.index, dow_counts.values, color=PALETTE)
ax.set_ylabel("Number of Orders")
ax.set_title("Orders by Day of Week", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=30)
save(fig, "eda_08_orders_by_dow.png")

# Chart 9: RFM Segment Distribution
seg_counts = rfm["segment"].value_counts()

fig, ax = plt.subplots(figsize=(7,4))
ax.bar(seg_counts.index, seg_counts.values, color=PALETTE)
ax.set_ylabel("Number of Customers")
ax.set_title("Customer Segments (RFM)", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=20)
save(fig, "eda_09_rfm_segments.png")

# Chart 10: Size Distribution 
size_order = ["XS","S","M","L","XL","XXL","3XL","4XL","5XL","6XL","Free","XX"]
size_counts = (sold[sold["size"].notna()]
               .groupby("size")["qty"]
               .sum()
               .sort_values(ascending=False)
               .head(12))

fig, ax = plt.subplots(figsize=(9,4))
ax.bar(size_counts.index, size_counts.values, color=PALETTE*2)
ax.set_ylabel("Units Sold")
ax.set_title("Units Sold by Size", fontsize=13, fontweight="bold")
save(fig, "eda_10_units_by_size.png")

print("\n✅  All 10 charts saved to /plots/")

