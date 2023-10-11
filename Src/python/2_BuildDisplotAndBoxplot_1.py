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

print('Оценка плотности')
plt.figure(figsize=(20,20))
plt.show()
for i in X_bp_nup_df.columns:
    sns.displot(x=i,kde=True,data=X_bp_nup_df)
    sns.rugplot(x=i, data=X_bp_nup_df) 
    plt.tight_layout()
    plt.show()
    #break

print('Оценка распределения вероятности величин')
plt.figure(figsize=(20,20))
plt.show()
for i in X_bp_nup_df.columns:
    sns.boxplot(x=X_bp_nup_df[i])
    sns.stripplot(x=X_bp_nup_df[i], data=X_bp_nup_df, \
          size=4, color=".3", linewidth=0) #,palette='bright' )
    plt.tight_layout()
    plt.show()
    #break




print('Выполнено')
