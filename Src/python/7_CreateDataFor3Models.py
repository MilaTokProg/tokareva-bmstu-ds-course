import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so

import os

import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")


normalized_df = pd.read_excel(open('normalized_data.xlsx', 'rb'),sheet_name='normalize', index_col=0)  
      
print(normalized_df.head())
print(normalized_df.shape)
print(normalized_df.info())

print(normalized_df.columns)

moduprrast_df = normalized_df[['Соотношение матрица-наполнитель', \
                              'Плотность, кг/м3', \
       'модуль упругости, ГПа', 'Количество отвердителя, м.%', \
       'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2', \
       'Поверхностная плотность, г/м2','Потребление смолы, г/м2', \
       'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки', \
                              'Модуль упругости при растяжении, ГПа']]

moduprrast_df.info()
## 0   Соотношение матрица-наполнитель       1000 non-null   float64
## 1   Плотность, кг/м3                      1000 non-null   float64
## 2   модуль упругости, ГПа                 1000 non-null   float64
## 3   Количество отвердителя, м.%           1000 non-null   float64
## 4   Содержание эпоксидных групп,%_2       1000 non-null   float64
## 5   Температура вспышки, С_2              1000 non-null   float64
## 6   Поверхностная плотность, г/м2         1000 non-null   float64
## 7   Потребление смолы, г/м2               1000 non-null   float64
## 8   Угол нашивки, град                    1000 non-null   int64  
## 9   Шаг нашивки                           1000 non-null   float64
## 10  Плотность нашивки                     1000 non-null   float64
## 11  Модуль упругости при растяжении, ГПа  

nodprochrast_df = normalized_df[['Соотношение матрица-наполнитель', \
         'Плотность, кг/м3', 'модуль упругости, ГПа', \
         'Количество отвердителя, м.%', 'Содержание эпоксидных групп,%_2', \
         'Температура вспышки, С_2', 'Поверхностная плотность, г/м2', \
         'Потребление смолы, г/м2',  'Угол нашивки, град', 'Шаг нашивки', \
         'Плотность нашивки', 'Прочность при растяжении, МПа']]

nodprochrast_df.info()
##0   Соотношение матрица-наполнитель  1000 non-null   float64
## 1   Плотность, кг/м3                 1000 non-null   float64
## 2   модуль упругости, ГПа            1000 non-null   float64
## 3   Количество отвердителя, м.%      1000 non-null   float64
## 4   Содержание эпоксидных групп,%_2  1000 non-null   float64
## 5   Температура вспышки, С_2         1000 non-null   float64
## 6   Поверхностная плотность, г/м2    1000 non-null   float64
## 7   Потребление смолы, г/м2          1000 non-null   float64
## 8   Угол нашивки, град               1000 non-null   int64  
## 9   Шаг нашивки                      1000 non-null   float64
## 10  Плотность нашивки                1000 non-null   float64
## 11  Прочность при растяжении, МПа    1000 non-null   float64

mat_nap_df = normalized_df[['Плотность, кг/м3', \
      'модуль упругости, ГПа', 'Количество отвердителя, м.%', \
      'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2', \
      'Поверхностная плотность, г/м2', 'Модуль упругости при растяжении, ГПа', \
      'Прочность при растяжении, МПа', 'Потребление смолы, г/м2', \
      'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки', \
      'Соотношение матрица-наполнитель']]

mat_nap_df.info()
## 0   Плотность, кг/м3                      1000 non-null   float64
## 1   модуль упругости, ГПа                 1000 non-null   float64
## 2   Количество отвердителя, м.%           1000 non-null   float64
## 3   Содержание эпоксидных групп,%_2       1000 non-null   float64
## 4   Температура вспышки, С_2              1000 non-null   float64
## 5   Поверхностная плотность, г/м2         1000 non-null   float64
## 6   Модуль упругости при растяжении, ГПа  1000 non-null   float64
## 7   Прочность при растяжении, МПа         1000 non-null   float64
## 8   Потребление смолы, г/м2               1000 non-null   float64
## 9   Угол нашивки, град                    1000 non-null   int64  
## 10  Шаг нашивки                           1000 non-null   float64
## 11  Плотность нашивки                     1000 non-null   float64
## 12  Соотношение матрица-наполнитель       1000 non-null   float64


#compression_opts = dict(method='zip', archive_name='out.csv')  
#df.to_csv('out.zip', index=False, compression=compression_opts)  
##from pathlib import Path  
##filepath = Path('folder/subfolder/out.csv')  
##filepath.parent.mkdir(parents=True, exist_ok=True)  
##df.to_csv(filepath)  

os.makedirs('output_data', exist_ok=True) 
moduprrast_df.to_csv('output_data/moduprrast.csv', index=False, header = False)
nodprochrast_df.to_csv('output_data/modprochrast.csv', index=False, header = False)
mat_nap_df.to_csv('output_data/mat_nap.csv', index=False, header = False)

print('Выполнено')
