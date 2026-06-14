--Q1. T0P 10 REVENUE CATEGORIES
SELECT
    category,
    ROUND(CAST(SUM(revenue) AS numeric), 2) AS total_revenue,
    COUNT(DISTINCT order_id) AS num_orders,
    ROUND(CAST(AVG(revenue) AS numeric), 2) AS avg_order_value
FROM orders
WHERE is_cancelled = 0 AND revenue > 0
GROUP BY category
ORDER BY total_revenue DESC
LIMIT 10;

--Q2 — States with Highest Average Order Value
SELECT
    ship_state,
    COUNT(DISTINCT order_id) AS num_orders,
    ROUND(CAST(AVG(revenue) AS numeric), 2) AS avg_order_value,
    ROUND(CAST(SUM(revenue) AS numeric), 2) AS total_revenue
FROM orders
WHERE is_cancelled = 0 AND revenue > 0
GROUP BY ship_state
HAVING COUNT(DISTINCT order_id) > 50
ORDER BY avg_order_value DESC
LIMIT 15;

--Q3 — Top Cities by Orders
SELECT
    ship_city,
    ship_state,
    COUNT(DISTINCT order_id) AS num_orders,
    ROUND(CAST(SUM(revenue) AS numeric), 2) AS total_revenue
FROM orders
WHERE is_cancelled = 0
GROUP BY ship_city, ship_state
ORDER BY num_orders DESC
LIMIT 15;

--Q4 — Month over Month Revenue Growth ⭐ Window Function
WITH monthly AS (
    SELECT
        month,
        ROUND(CAST(SUM(revenue) AS numeric), 2) AS monthly_revenue
    FROM orders
    WHERE is_cancelled = 0 AND revenue > 0
    GROUP BY month
)
SELECT
    month,
    monthly_revenue,
    LAG(monthly_revenue) OVER (ORDER BY month) AS prev_month_revenue,
    ROUND(
        100.0 * (monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY month))
        / NULLIF(LAG(monthly_revenue) OVER (ORDER BY month), 0),
    2) AS mom_growth_pct
FROM monthly
ORDER BY month;

--Q5 — Repeat vs One-Time Buyers ⭐ Window Function
WITH buyer_freq AS (
    SELECT
        customer_id,
        ship_state,
        COUNT(DISTINCT order_id) AS order_count
    FROM orders
    WHERE is_cancelled = 0
    GROUP BY customer_id, ship_state
),
state_summary AS (
    SELECT
        ship_state,
        COUNT(*) AS total_customers,
        SUM(CASE WHEN order_count = 1 THEN 1 ELSE 0 END) AS one_time_buyers,
        SUM(CASE WHEN order_count > 1 THEN 1 ELSE 0 END) AS repeat_buyers
    FROM buyer_freq
    GROUP BY ship_state
    HAVING COUNT(*) > 10
)
SELECT *,
    ROUND(100.0 * repeat_buyers / total_customers, 1) AS repeat_rate_pct,
    RANK() OVER (ORDER BY repeat_buyers DESC) AS repeat_rank
FROM state_summary
ORDER BY repeat_rank
LIMIT 15;

--Q6 — Top SKUs by Revenue ⭐ Window Function
SELECT
    sku,
    category,
    COUNT(DISTINCT order_id) AS num_orders,
    SUM(qty) AS total_units,
    ROUND(CAST(SUM(revenue) AS numeric), 2) AS total_revenue,
    RANK() OVER (ORDER BY SUM(revenue) DESC) AS revenue_rank
FROM orders
WHERE is_cancelled = 0 AND revenue > 0
GROUP BY sku, category
ORDER BY total_revenue DESC
LIMIT 10;

--Q7 — Fulfilment Method Comparison
SELECT
    fulfilment,
    COUNT(DISTINCT order_id) AS num_orders,
    ROUND(CAST(SUM(revenue) AS numeric), 2) AS total_revenue,
    ROUND(CAST(AVG(revenue) AS numeric), 2) AS avg_order_value,
    ROUND(100.0 * SUM(is_cancelled) / COUNT(*), 2) AS cancellation_rate_pct
FROM orders
GROUP BY fulfilment
ORDER BY total_revenue DESC;

--Q8 — B2B vs B2C by Category ⭐ Window Function
SELECT
    category,
    b2b,
    COUNT(DISTINCT order_id) AS num_orders,
    ROUND(CAST(SUM(revenue) AS numeric), 2) AS total_revenue,
    ROUND(
        CAST(100.0 * SUM(revenue)
        / SUM(SUM(revenue)) OVER (PARTITION BY category) AS numeric),
    1) AS pct_of_category_revenue
FROM orders
WHERE is_cancelled = 0 AND revenue > 0
GROUP BY category, b2b
ORDER BY total_revenue DESC
LIMIT 20;

--Q9 — Churn Rate by RFM Segment ⭐ Window Function
SELECT
    segment,
    COUNT(*) AS num_customers,
    SUM(churned) AS churned_customers,
    ROUND(100.0 * SUM(churned) / COUNT(*), 1) AS churn_rate_pct,
    ROUND(CAST(AVG(monetary) AS numeric), 2) AS avg_monetary,
    RANK() OVER (ORDER BY SUM(churned) DESC) AS churn_rank
FROM rfm
GROUP BY segment
ORDER BY churn_rate_pct DESC;

--Q10 — Top States by Total Revenue
SELECT
    ship_state,
    ROUND(CAST(SUM(revenue) AS numeric), 2) AS total_revenue,
    COUNT(DISTINCT order_id) AS num_orders,
    ROUND(CAST(AVG(revenue) AS numeric), 2) AS avg_order_value,
    RANK() OVER (ORDER BY SUM(revenue) DESC) AS revenue_rank
FROM orders
WHERE is_cancelled = 0 AND revenue > 0
GROUP BY ship_state
ORDER BY total_revenue DESC
LIMIT 15;