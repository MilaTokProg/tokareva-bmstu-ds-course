
SELECT 'mat_nap' AS Field, AVG(mat_nap) AS mean, percentile_cont(0.5) WITHIN GROUP (ORDER BY mat_nap) AS Mediana,
MAX(mat_nap), MIN(mat_nap), stddev(mat_nap)
FROM public."full_X_bp";

SELECT 'plotn' AS Field, AVG(plotn) AS mean, percentile_cont(0.5) WITHIN GROUP (ORDER BY plotn) AS Mediana,
MAX(plotn), MIN(plotn), stddev(plotn)
FROM public."full_X_bp";

SELECT 'modupr' AS Field, AVG(modupr) AS mean, percentile_cont(0.5) WITHIN GROUP (ORDER BY modupr) AS Mediana,
MAX(modupr), MIN(modupr), stddev(modupr)
FROM public."full_X_bp";

SELECT 'epoks' AS Field, AVG(epoks) AS mean, percentile_cont(0.5) WITHIN GROUP (ORDER BY epoks) AS Mediana,
MAX(epoks), MIN(epoks), stddev(epoks)
FROM public."full_X_bp";

SELECT 'temperature' AS Field, AVG(temperature) AS mean, percentile_cont(0.5) WITHIN GROUP (ORDER BY temperature) AS Mediana,
MAX(temperature), MIN(temperature), stddev(temperature)
FROM public."full_X_bp";

SELECT 'povpl' AS Field, AVG(povpl) AS mean, percentile_cont(0.5) WITHIN GROUP (ORDER BY povpl) AS Mediana,
MAX(povpl), MIN(povpl), stddev(povpl)
FROM public."full_X_bp";

SELECT 'moduprrast' AS Field, AVG(moduprrast) AS mean, percentile_cont(0.5) WITHIN GROUP (ORDER BY moduprrast) AS Mediana,
MAX(moduprrast), MIN(moduprrast), stddev(moduprrast)
FROM public."full_X_bp";

SELECT 'procn' AS Field, AVG(procn) AS mean, percentile_cont(0.5) WITHIN GROUP (ORDER BY procn) AS Mediana,
MAX(procn), MIN(procn), stddev(procn)
FROM public."full_X_bp";

SELECT 'potrsmoli' AS Field, AVG(potrsmoli) AS mean, percentile_cont(0.5) WITHIN GROUP (ORDER BY potrsmoli) AS Mediana,
MAX(potrsmoli), MIN(potrsmoli), stddev(potrsmoli)
FROM public."full_X_bp";

SELECT 'ugnash' AS Field, AVG(ugnash) AS mean, percentile_cont(0.5) WITHIN GROUP (ORDER BY ugnash) AS Mediana,
MAX(ugnash), MIN(ugnash), stddev(ugnash)
FROM public."full_X_bp";

SELECT 'stepnash' AS Field, AVG(stepnash) AS mean, percentile_cont(0.5) WITHIN GROUP (ORDER BY stepnash) AS Mediana,
MAX(stepnash), MIN(stepnash), stddev(stepnash)
FROM public."full_X_bp";

SELECT 'plotnash' AS Field, AVG(plotnash) AS mean, percentile_cont(0.5) WITHIN GROUP (ORDER BY plotnash) AS Mediana,
MAX(plotnash), MIN(plotnash), stddev(plotnash)
FROM public."full_X_bp";

