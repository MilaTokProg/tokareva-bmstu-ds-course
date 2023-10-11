SELECT COUNT(*) FROM public."full_X_bp";

SELECT mat_nap, plotn, modupr, kolotb, epoks, temperature, povpl, 
moduprrast, procn, potrsmoli, ugnash, stepnash, plotnash
FROM public."full_X_bp" 
WHERE 
mat_nap > 
((SELECT AVG(mat_nap) FROM public."full_X_bp") - 3.0000000 * (SELECT stddev(mat_nap) FROM public."full_X_bp" )) 
AND 
mat_nap <
((SELECT AVG(mat_nap) FROM public."full_X_bp") + 3.0000000 * (SELECT stddev(mat_nap) FROM public."full_X_bp" )) 
AND
plotn >
((SELECT AVG(plotn) FROM public."full_X_bp") - 3.0000000 * (SELECT stddev(plotn) FROM public."full_X_bp" )) 
AND plotn < 
((SELECT AVG(plotn) FROM public."full_X_bp") + 3.0000000 * (SELECT stddev(plotn) FROM public."full_X_bp" )) 
AND
modupr > 
((SELECT AVG(modupr) FROM public."full_X_bp") - 3.000000 * (SELECT stddev(modupr) FROM public."full_X_bp" )) 
AND modupr <
((SELECT AVG(modupr) FROM public."full_X_bp") + 3.000000 * (SELECT stddev(modupr) FROM public."full_X_bp" )) 
AND
kolotb >
((SELECT AVG(kolotb) FROM public."full_X_bp") - 3.0000000 * (SELECT stddev(kolotb) FROM public."full_X_bp" )) 
AND kolotb < 
((SELECT AVG(kolotb) FROM public."full_X_bp") + 3.0000000 * (SELECT stddev(kolotb) FROM public."full_X_bp" )) 
AND
temperature >
((SELECT AVG(temperature) FROM public."full_X_bp") - 3.000000 * (SELECT stddev(temperature) FROM public."full_X_bp" )) 
AND temperature <
((SELECT AVG(temperature) FROM public."full_X_bp") + 3.000000 * (SELECT stddev(temperature) FROM public."full_X_bp" ))
AND
povpl > 
((SELECT AVG(povpl) FROM public."full_X_bp") - 3.0000000 * (SELECT stddev(povpl) FROM public."full_X_bp" )) 
AND povpl < 
((SELECT AVG(povpl) FROM public."full_X_bp") + 3.000000 * (SELECT stddev(povpl) FROM public."full_X_bp" ))
AND
moduprrast > 
((SELECT AVG(moduprrast) FROM public."full_X_bp") - 3.000000 * (SELECT stddev(moduprrast) FROM public."full_X_bp" )) 
AND moduprrast <
((SELECT AVG(moduprrast) FROM public."full_X_bp") + 3.000000 * (SELECT stddev(moduprrast) FROM public."full_X_bp" ))
AND
procn > 
((SELECT AVG(procn) FROM public."full_X_bp") - 3.000000 * (SELECT stddev(procn) FROM public."full_X_bp" )) 
AND procn <
((SELECT AVG(procn) FROM public."full_X_bp") + 3.000000 * (SELECT stddev(procn) FROM public."full_X_bp" ))
AND
potrsmoli > 
((SELECT AVG(potrsmoli) FROM public."full_X_bp") - 3.000000 * (SELECT stddev(potrsmoli) FROM public."full_X_bp" )) 
AND potrsmoli <
((SELECT AVG(potrsmoli) FROM public."full_X_bp") + 3.000000 * (SELECT stddev(potrsmoli) FROM public."full_X_bp" ))
AND
stepnash >
((SELECT AVG(stepnash) FROM public."full_X_bp") - 3.0000000 * (SELECT stddev(stepnash) FROM public."full_X_bp" )) 
AND stepnash < 
((SELECT AVG(stepnash) FROM public."full_X_bp") + 3.0000000 * (SELECT stddev(stepnash) FROM public."full_X_bp" ))
AND
plotnash >
((SELECT AVG(plotnash) FROM public."full_X_bp") - 3.0000000 * (SELECT stddev(plotnash) FROM public."full_X_bp" )) 
AND
plotnash <
((SELECT AVG(plotnash) FROM public."full_X_bp") + 3.0000000 * (SELECT stddev(plotnash) FROM public."full_X_bp" ) );
