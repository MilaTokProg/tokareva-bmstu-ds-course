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
print('копирование')
print(filtered_df)
for i in filtered_df.columns:
 if i != 'Угол нашивки, град': 
   #meanvalue = filtered_df[i].mean()
   filtered_df = filtered_df[(filtered_df[i] > filtered_df[i].mean() - 3 *filtered_df[i].std()) & (filtered_df[i] < filtered_df[i].mean() + 3 *filtered_df[i].std())]

print(filtered_df.shape)

plt.figure(figsize=(40,40))
plt.show()
print('Оценка плотности очищенной коллекции')
for i in filtered_df.columns:
    sns.displot(x=i,kde=True,data=filtered_df)
    plt.show()


print('Оценка распределения вероятности величин ощищеной коллекции')
plt.figure(figsize=(40,40))
plt.show()
for i in filtered_df.columns:
    sns.boxplot(x=filtered_df[i], data=filtered_df,
            whis=[0, 100], width=.6, palette="vlag")
    sns.stripplot(x=filtered_df[i], data=filtered_df,
              size=4, color=".3", linewidth=0)
    plt.show()

print('Выводим Попарные графики рассеяния для очищенной коллекции')
print('Ждите, выполняется долго!')
g = sns.PairGrid(filtered_df, diag_sharey=False)
g.map_upper(sns.scatterplot) #, s=15)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)
plt.show()

filtered_df.to_excel("clearing_X_bp_nup.xlsx", sheet_name='ClearData') 


print('Выполнено')
