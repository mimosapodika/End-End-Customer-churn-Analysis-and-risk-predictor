
SELECT
    COUNT(*) AS total_customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_lost
FROM customers;


SELECT
    Contract,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_lost
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

SELECT
    PaymentMethod,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

SELECT
    CASE
        WHEN tenure <= 12 THEN '0-12 Months'
        WHEN tenure <= 24 THEN '13-24 Months'
        WHEN tenure <= 48 THEN '25-48 Months'
        ELSE '49+ Months'
    END AS tenure_cohort,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY tenure_cohort
ORDER BY churn_rate_pct DESC;


SELECT
    customerID,
    tenure,
    MonthlyCharges,
    Contract,
    PaymentMethod
FROM customers
WHERE churn_flag = 0
  AND Contract = 'Month-to-month'
  AND MonthlyCharges > 70
  AND tenure < 12
ORDER BY MonthlyCharges DESC
LIMIT 15;
