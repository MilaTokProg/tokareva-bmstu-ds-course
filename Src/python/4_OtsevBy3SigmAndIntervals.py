import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so

import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

X_bp_df = pd.read_excel(open('X_bp.xlsx', 'rb'),sheet_name='X_bp.csv', index_col=0)  
X_nup_df = pd.read_excel(open('X_nup.xlsx', 'rb'),sheet_name='X_nup.csv', index_col=0)  
X_bp_nup_df = pd.concat([X_bp_df, X_nup_df], axis=1, join="inner")

filtered_df = X_bp_nup_df.copy ()
print('Таблица без очистки от выбросов')
print(filtered_df.shape)
for i in filtered_df.columns:
 if i != 'Угол нашивки, град': 
   #meanvalue = filtered_df[i].mean()
   filtered_df = filtered_df[(filtered_df[i] > filtered_df[i].mean() - 3 *filtered_df[i].std()) & (filtered_df[i] < filtered_df[i].mean() + 3 *filtered_df[i].std())]
print('Таблица после очистки от выбросов методом 3-х сигм')
print(filtered_df.shape)

def q1(x):
 return x.quantile(0.25)
def q3(x):
 return x.quantile(0.75)

##Limmax​=Q3​+Ni​∗IQR
##Limmin=Q1−Ni∗IQRLimmin​=Q1​−Ni​∗IQR
##где Q3Q3 — третий квартиль, Q1Q1 — первый квартиль,
## IQRIQR — интерквартильное расстояние (или интерквартильный размах), определяемое по формуле IQR=Q3−Q1IQR=Q3​−Q1​,
##Ni=1.5Ni​=1.5 — заданное число интерквартильного размаха. В некоторых случаях, когда данные сильно распределены, этот параметр можно варьировать в сторону увеличения.

filtered_df = X_bp_nup_df.copy ()
print('Таблица без очистки от выбросов')
print(filtered_df.shape)
for i in filtered_df.columns:
 if i != 'Угол нашивки, град': 
   q3v = q3(filtered_df[i])
   q1v = q1(filtered_df[i])
   NIQR = (q3v - q1v)*1.5
   Limmax = q3v + NIQR
   Limmin = q1v - NIQR
   filtered_df = filtered_df[(filtered_df[i] > Limmin) & (filtered_df[i] < Limmax)]

print('Таблица после очистки от выбросов методом интерквартильного интервала')
print(filtered_df.shape)



print('Выполнено')
