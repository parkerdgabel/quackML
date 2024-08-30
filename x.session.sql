-- CREATE TABLE x AS FROM '~/Downloads/x_analytics.csv'

-- SELECT Likes / sum(Impressions) OVER (ORDER BY Date)
-- FROM x

--   SELECT
--         Impressions,
--         COUNT(*) OVER (ORDER BY Impressions) * 1.0 / COUNT(*) OVER () AS ECDF
--     FROM
--         x

SELECT
    Date,
    Impressions,
    Impressions - LAG(Impressions, 1) OVER () AS RateOfChange,
    -- RateOfChange - LAG(RateOfChange, 1) OVER () AS SecondDerivative,
    RateOfChange / Impressions * 100 AS RateOfChangePct
FROM
    x
Qualify
    Impressions > 0
-- ORDER BY
--     Date ASC;