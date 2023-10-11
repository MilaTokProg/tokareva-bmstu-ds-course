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

clearing_df = pd.read_excel(open('clearing_X_bp_nup.xlsx', 'rb'),sheet_name='ClearData', index_col=0)  
      
print(clearing_df.head())
print('clearing_df.shape')
print(clearing_df.shape)
print('X_bp_df.info()')
print(clearing_df.info())

# Словарь для хранения множителя и дельты для восстановления
# данных из нормализованных в истинные
dic_const = {}

##(df-df.min ())/(df.max ()-df.min ())
normalized_df = clearing_df.copy ()
##print('копирование')
print(normalized_df.head())
for i in normalized_df.columns:
    minvalue = normalized_df[i].min()
    maxvalue = normalized_df[i].max()
    mnozitelvalue = maxvalue - minvalue
    deltavalue = minvalue
    dic_const[i] = dict(mnozitel = np.float64(mnozitelvalue),delta = np.float64(deltavalue)) 
    print(i + ' Delta: ' + str(deltavalue) + ' Mnoj: ' + str(mnozitelvalue))
    normalized_df[i] = (normalized_df[i] - deltavalue) / mnozitelvalue 

print(normalized_df.head())

# Сохраняем словарь в виде файла JSON
with open("mnojanddelta.json", "w", encoding="utf-8") as f:
     json.dump(dic_const, f, indent = 4)


result_tests_df = pd.DataFrame(columns = ['Свойство', \
          'среднее',\
          'медиана', 'макс.', 'минимум',
          'станд. откл.'])

for i in normalized_df.columns:
    tmp_df = pd.DataFrame({'Свойство' : [i], \
                'среднее' : [normalized_df[i].mean()], \
                'медиана' : [normalized_df[i].median()], \
                'макс.' : [normalized_df[i].max()], \
                'минимум' : [normalized_df[i].min()], \
                'станд. откл.' : [normalized_df[i].std()]})
    if  (len(result_tests_df) == 0):
         result_tests_df = tmp_df.copy()
    else:
         result_tests_df = pd.concat([result_tests_df, tmp_df], ignore_index=True)
        

result_tests_df.head()
  
# Нарисуем таблицу описательной статистики для нормализованных данных
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

# строим тепловую карту кореляций
#plt.figure(figsize=(40,40))
corr_df =normalized_df.corr()
print(corr_df.head())
plt.figure(figsize = (12,8))
sns.heatmap(corr_df, annot=True, cmap= 'coolwarm',fmt='.1f',linewidth=.5 )
#sns.heatmap(normalized_df.corr(method='spearman'), annot=True, cmap= 'coolwarm',fmt='.1f',linewidth=.5 )
# sns.heatmap(normalized_df.corr(), annot=True, cmap= 'coolwarm',fmt='g',linewidth=.5 )
#fmt='g' 
#sns.heatmap(normalized_df.corr(), annot=True,square=True,   fmt='.1f', linecolor='black', cmap= 'coolwarm', linewidth=.5)
# оптимизируем поля и расположение объектов
plt.tight_layout()
plt.show()

normalized_df.to_excel("normalized_data.xlsx", sheet_name='normalize')

normalized_df = pd.read_excel(open('normalized_data.xlsx', 'rb'),sheet_name='normalize', index_col=0)  

print(normalized_df.columns)

#Проверяю восстановление данных
dic_const_vost  = {}
with open("mnojanddelta.json", "r") as f:
     dic_const_vost = json.load(f)

print(dic_const_vost) 

proverka_df = normalized_df.copy ()
for i in proverka_df.columns:
      proverka_df[i] = proverka_df[i] * dic_const_vost[i]['mnozitel'] + dic_const_vost[i]['delta'] 

print('востановленные данные')
print(proverka_df.head())



print('Выполнено')
