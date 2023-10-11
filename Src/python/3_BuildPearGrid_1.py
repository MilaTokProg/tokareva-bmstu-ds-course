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

print('Выводим Попарные графики рассеяния')
print('Ждите, выполняется долго!')

g = sns.PairGrid(X_bp_nup_df, diag_sharey=False) # парный график распределения
g.map_upper(sns.scatterplot) #
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)
plt.tight_layout()
plt.show()


print('Выполнено')
