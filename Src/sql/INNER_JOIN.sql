SELECT A.indx, A.mat_nap, A.plotn, A.modupr, A.kolotb, A.epoks, 
A.temperature, A.povpl, A.moduprrast, A.procn, A.potrsmoli,
B.ugnash, B.stepnash, B.plotnash
INTO public."full_X_bp"
FROM public."X_bp" AS A INNER JOIN public."X_nup" AS B
ON A."indx" = B."Indx";