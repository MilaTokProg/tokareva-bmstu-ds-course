SELECT *
FROM public."full_X_bp"
WHERE mat_nap > ((SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY mat_nap) FROM public."full_X_bp") - 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY mat_nap) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY mat_nap) FROM public."full_X_bp")))
AND mat_nap < ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY mat_nap) FROM public."full_X_bp") + 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY mat_nap) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY mat_nap) FROM public."full_X_bp")))
AND plotn > ((SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY plotn) FROM public."full_X_bp") - 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY plotn) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY plotn) FROM public."full_X_bp")))
AND plotn < ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY plotn) FROM public."full_X_bp") + 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY plotn) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY plotn) FROM public."full_X_bp")))
AND modupr > ((SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY modupr) FROM public."full_X_bp") - 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY modupr) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY modupr) FROM public."full_X_bp")))
AND modupr < ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY modupr) FROM public."full_X_bp") + 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY modupr) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY modupr) FROM public."full_X_bp")))
AND kolotb > ((SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY kolotb) FROM public."full_X_bp") - 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY kolotb) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY kolotb) FROM public."full_X_bp")))
AND kolotb < ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY kolotb) FROM public."full_X_bp") + 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY kolotb) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY kolotb) FROM public."full_X_bp")))
AND epoks > ((SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY epoks) FROM public."full_X_bp") - 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY epoks) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY epoks) FROM public."full_X_bp")))
AND epoks < ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY epoks) FROM public."full_X_bp") + 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY epoks) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY epoks) FROM public."full_X_bp")))
AND temperature > ((SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY temperature) FROM public."full_X_bp") - 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY temperature) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY temperature) FROM public."full_X_bp")))
AND temperature < ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY temperature) FROM public."full_X_bp") + 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY temperature) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY temperature) FROM public."full_X_bp")))
AND povpl > ((SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY povpl) FROM public."full_X_bp") - 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY povpl) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY povpl) FROM public."full_X_bp")))
AND povpl < ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY povpl) FROM public."full_X_bp") + 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY povpl) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY povpl) FROM public."full_X_bp")))
AND moduprrast > ((SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY moduprrast) FROM public."full_X_bp") - 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY moduprrast) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY moduprrast) FROM public."full_X_bp")))
AND moduprrast < ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY moduprrast) FROM public."full_X_bp") + 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY moduprrast) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY moduprrast) FROM public."full_X_bp")))
AND procn > ((SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY procn) FROM public."full_X_bp") - 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY procn) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY procn) FROM public."full_X_bp")))
AND procn < ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY procn) FROM public."full_X_bp") + 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY procn) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY procn) FROM public."full_X_bp")))
AND potrsmoli > ((SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY potrsmoli) FROM public."full_X_bp") - 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY potrsmoli) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY potrsmoli) FROM public."full_X_bp")))
AND potrsmoli < ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY potrsmoli) FROM public."full_X_bp") + 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY potrsmoli) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY potrsmoli) FROM public."full_X_bp")))
AND ugnash > ((SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY ugnash) FROM public."full_X_bp") - 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY ugnash) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY ugnash) FROM public."full_X_bp")))
AND ugnash < ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY ugnash) FROM public."full_X_bp") + 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY ugnash) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY ugnash) FROM public."full_X_bp")))
AND stepnash > ((SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY stepnash) FROM public."full_X_bp") - 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY stepnash) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY stepnash) FROM public."full_X_bp")))
AND stepnash < ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY stepnash) FROM public."full_X_bp") + 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY stepnash) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY stepnash) FROM public."full_X_bp")))
AND plotnash > ((SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY plotnash) FROM public."full_X_bp") - 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY plotnash) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY plotnash) FROM public."full_X_bp")))
AND plotnash < ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY plotnash) FROM public."full_X_bp") + 1.5 * ((SELECT percentile_cont(0.75) 
WITHIN GROUP (ORDER BY plotnash) FROM public."full_X_bp") - (SELECT percentile_cont(0.25) 
WITHIN GROUP (ORDER BY plotnash) FROM public."full_X_bp")));