-- ============================================================
-- Customer Churn & Retention Intelligence
-- SQL Analysis
-- Database: churn_intelligence
-- Table: public.customer_churn
-- ============================================================


-- 1. Overall Customer & Churn Overview
SELECT
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    COUNT(*) - SUM(churn) AS retained_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate
FROM public.customer_churn;


-- 2. Revenue Overview
SELECT
    ROUND(SUM(monthly_charges), 2) AS total_monthly_revenue,
    ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END), 2)
        AS monthly_revenue_lost,
    ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END) * 12, 2)
        AS annualized_revenue_lost
FROM public.customer_churn;


-- 3. Churn by Contract
SELECT
    contract,
    COUNT(*) AS customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate
FROM public.customer_churn
GROUP BY contract
ORDER BY churn_rate DESC;


-- 4. Churn by Payment Method
SELECT
    payment_method,
    COUNT(*) AS customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate
FROM public.customer_churn
GROUP BY payment_method
ORDER BY churn_rate DESC;


-- 5. Churn by Internet Service
SELECT
    internet_service,
    COUNT(*) AS customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate
FROM public.customer_churn
GROUP BY internet_service
ORDER BY churn_rate DESC;


-- 6. Churn by Tenure Group
SELECT
    tenure_group,
    COUNT(*) AS customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate
FROM public.customer_churn
GROUP BY tenure_group
ORDER BY churn_rate DESC;


-- 7. Churn by Technical Support
SELECT
    tech_support,
    COUNT(*) AS customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate
FROM public.customer_churn
GROUP BY tech_support
ORDER BY churn_rate DESC;


-- 8. Churn by Online Security
SELECT
    online_security,
    COUNT(*) AS customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate
FROM public.customer_churn
GROUP BY online_security
ORDER BY churn_rate DESC;


-- 9. High-Risk Business Segment
SELECT
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate,
    ROUND(SUM(monthly_charges), 2) AS total_monthly_charges,
    ROUND(
        SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END),
        2
    ) AS monthly_revenue_lost
FROM public.customer_churn
WHERE contract = 'Month-to-month'
  AND tenure_months <= 12
  AND internet_service = 'Fiber optic'
  AND payment_method = 'Electronic check';


-- 10. Top Churn Reasons
SELECT
    churn_reason,
    COUNT(*) AS churned_customers
FROM public.customer_churn
WHERE churn = 1
  AND churn_reason IS NOT NULL
  AND TRIM(churn_reason) <> ''
GROUP BY churn_reason
ORDER BY churned_customers DESC
LIMIT 10;


-- 11. Customer Value Segment Analysis
SELECT
    customer_value_segment,
    COUNT(*) AS customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate,
    ROUND(AVG(cltv), 2) AS average_cltv,
    ROUND(
        SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END),
        2
    ) AS monthly_revenue_lost
FROM public.customer_churn
GROUP BY customer_value_segment
ORDER BY churn_rate DESC;