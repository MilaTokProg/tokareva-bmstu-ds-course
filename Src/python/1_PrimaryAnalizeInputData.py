import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import json

import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

#Загрузка исходных данных
print('Загрузка файла X_bp.xlsx')
print("X_bp_df = pd.read_excel(open('X_bp.xlsx', 'rb'),sheet_name='X_bp.csv', index_col=0)")
X_bp_df = pd.read_excel(open('X_bp.xlsx', 'rb'),sheet_name='X_bp.csv', index_col=0)  
      
#print(X_bp_df.head())
#Вывожу сведения о таблице X_bp  и свойствах полей  
print('X_bp_df.shape')
print(X_bp_df.shape) # показывает сколько строк
print('X_bp_df.info()')
print(X_bp_df.info())

X_nup_df = pd.read_excel(open('X_nup.xlsx', 'rb'),sheet_name='X_nup.csv', index_col=0)  
      
print(X_nup_df.head())
#Вывожу сведения о таблице X_nup  и свойствах полей 
print('X_nup_df.shape')
print(X_nup_df.shape)
print('X_nup_df.info()')
print(X_nup_df.info())

#Объединяю данные соединением INNER JOIN
print('Соединение INNER JOIN')
print('X_bp_nup_df = pd.concat([X_bp_df, X_nup_df], axis=1, join="inner")')      
X_bp_nup_df = pd.concat([X_bp_df, X_nup_df], axis=1, join="inner")

#print(X_bp_nup_df.head())
#Вывожу сведения о таблице X_bp_nup  и свойствах полей 
print('X_bp_nup_df.shape')
print(X_bp_nup_df.shape)
#print('X_bp_nup_df.describe()')
#print(X_bp_nup_df.describe())
print('X_bp_nup_df.info()')
print(X_bp_nup_df.info())

#print( X_bp_nup_df.columns)

#Вывожу показатели описательной статистики
#каждого из полей таблицы X_bp_nup 
for i in X_bp_nup_df.columns:
    print(i)
    print('Кол-во уникальных значений: ' + str(X_bp_nup_df[i].nunique())) # 
    print('Кол-во пропусков: ' + str(X_bp_nup_df[i].isna().sum()))
    print(X_bp_nup_df[i].describe()) #describe  - Выводит показательную статистику (max, min и т.д.)



for i in X_bp_nup_df.columns:
    print(i)
    print('Кол-во уникальных значений: ' + str(X_bp_nup_df[i].nunique())) # 
    print('Кол-во пропусков: ' + str(X_bp_nup_df[i].isna().sum()))


result_tests_df = pd.DataFrame(columns = ['Свойство', \
          'среднее',\
          'медиана', 'макс.', 'минимум',
          'станд. откл.'])

# Словарь для хранения полей со значениями MIN, MAX, AVG
# для работы консольного приложения
# 
dic_const = {}


for i in X_bp_nup_df.columns:
    tmp_df = pd.DataFrame({'Свойство' : [i], \
                'среднее' : [X_bp_nup_df[i].mean()], \
                'медиана' : [X_bp_nup_df[i].median()], \
                'макс.' : [X_bp_nup_df[i].max()], \
                'минимум' : [X_bp_nup_df[i].min()], \
                'станд. откл.' : [X_bp_nup_df[i].std()]})
    # Это будет необходимо при работе консольного приложения
    dic_const[i] = dict(minimum = np.float64(X_bp_nup_df[i].min()), \
                        maximum = np.float64(X_bp_nup_df[i].max()), \
                        avg = np.float64(X_bp_nup_df[i].mean())) 

    if  (len(result_tests_df) == 0):
         result_tests_df = tmp_df.copy()
    else:
         result_tests_df = pd.concat([result_tests_df, tmp_df], ignore_index=True)
        
# Сохраняем словарь в виде файла JSON
with open("minmaxmeans.json", "w", encoding="utf-8") as f:
     json.dump(dic_const, f, indent = 4)


result_tests_df.head()
  
# Нарисуем таблицу описательной статистики
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText = result_tests_df.values, \
                 colLabels = result_tests_df.columns, \
                 loc = 'center')

table.auto_set_font_size(False)
table.set_fontsize(10)
#table.scale(2, 2)
table.auto_set_column_width(col = list(range(len(result_tests_df.columns))))
#fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()
print('Выполнено')
