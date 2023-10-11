-- Проверка на пустые значения
SELECT *
FROM public."X_bp" WHERE 
mat_nap IS NULL OR plotn IS NULL OR modupr IS NULL OR kolotb IS NULL OR epoks IS NULL OR temperature
IS NULL OR povpl IS NULL OR moduprrast IS NULL OR procn IS NULL OR potrsmoli IS NULL;

SELECT *
FROM public."X_nup" WHERE 
ugnash IS NULL OR stepnash IS NULL OR plotnash IS NULL;


SELECT indx, mat_nap, plotn, modupr, kolotb, epoks, temperature, povpl, moduprrast, procn, potrsmoli, ugnash, stepnash, plotnash
FROM public."full_X_bp" WHERE 
mat_nap IS NULL OR plotn IS NULL OR modupr IS NULL OR kolotb IS NULL OR epoks IS NULL OR temperature
IS NULL OR povpl IS NULL OR moduprrast IS NULL OR procn IS NULL OR potrsmoli IS NULL OR ugnash IS NULL OR stepnash IS NULL 
OR plotnash IS NULL;

	
	