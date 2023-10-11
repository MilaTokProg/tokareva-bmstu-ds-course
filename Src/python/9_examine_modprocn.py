from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
from sklearn.dummy import DummyRegressor
import json
import seaborn as sns


dic_const_vost  = {}
with open("mnojanddelta.json", "r") as f:
     dic_const_vost = json.load(f)

#Прочность при растяжении, МПа Delta: 1036.85660535 Mnoj: 2811.58012652618
MAT_NAP_DELTA = dic_const_vost['Прочность при растяжении, МПа']['delta']
MAT_NAP_MNOJ = dic_const_vost['Прочность при растяжении, МПа']['mnozitel']


# random seed for reproducibility
np.random.seed(2)


dataset = pd.read_csv('output_data/modprochrast.csv', header = None).add_prefix("data")
#dataset.head()
dataset.info()
#print(dataset.columns)

inputscolumn = ['Соотношение матрица-наполнитель', \
         'Плотность, кг/м3', 'модуль упругости, ГПа', \
         'Количество отвердителя, м.%', 'Содержание эпоксидных групп,%_2', \
         'Температура вспышки, С_2', 'Поверхностная плотность, г/м2', \
         'Потребление смолы, г/м2',  'Угол нашивки, град', 'Шаг нашивки', \
         'Плотность нашивки']
outputscolumns = ['Прочность при растяжении, МПа']




dataset.columns = ['Соотношение матрица-наполнитель', \
         'Плотность, кг/м3', 'модуль упругости, ГПа', \
         'Количество отвердителя, м.%', 'Содержание эпоксидных групп,%_2', \
         'Температура вспышки, С_2', 'Поверхностная плотность, г/м2', \
         'Потребление смолы, г/м2',  'Угол нашивки, град', 'Шаг нашивки', \
         'Плотность нашивки', 'Прочность при растяжении, МПа']

dataset.info()

result_statistics_df = pd.DataFrame(columns = ['Набор данных','Кол-во строк','Свойство', \
          'среднее',\
          'медиана', 'макс.', 'минимум',
          'станд. откл.'])



values = dataset.to_numpy()


# 12 столбцов параметры, 13 столбец результат
# split into input (X) and output (Y) variables, splitting csv data
X = values[:,:-1] # Все кроме последненго
Y = values[:,-1] # Только последний

# split X, Y into a train and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# Описательная статистика по обучающей и тестирующей выборке.
df_0 = pd.DataFrame(x_train, columns = inputscolumn)
for i in df_0.columns:
    tmp_df = pd.DataFrame({'Набор данных' : ['train'], \
                'Кол-во строк' : [df_0[i].count()], \
                'Свойство' : [i], \
                'среднее' : [df_0[i].mean()], \
                'медиана' : [df_0[i].median()], \
                'макс.' : [df_0[i].max()], \
                'минимум' : [df_0[i].min()], \
                'станд. откл.' : [df_0[i].std()]})
    
    if  (len(result_statistics_df) == 0):
        result_statistics_df = tmp_df.copy()
    else:
         result_statistics_df = pd.concat([result_statistics_df, tmp_df], ignore_index=True)
df_0 = pd.DataFrame(y_train, columns = outputscolumns)
for i in df_0.columns:
    tmp_df = pd.DataFrame({'Набор данных' : ['train'], \
                'Кол-во строк' : [df_0[i].count()], \
                'Свойство' : [i], \
                'среднее' : [df_0[i].mean()], \
                'медиана' : [df_0[i].median()], \
                'макс.' : [df_0[i].max()], \
                'минимум' : [df_0[i].min()], \
                'станд. откл.' : [df_0[i].std()]})
    if  (len(result_statistics_df) == 0):
         result_statistics_df = tmp_df.copy()
    else:
         result_statistics_df = pd.concat([result_statistics_df, tmp_df], ignore_index=True)

df_0 = pd.DataFrame(x_test, columns = inputscolumn)
for i in df_0.columns:
    tmp_df = pd.DataFrame({'Набор данных' : ['test'], \
                'Кол-во строк' : [df_0[i].count()], \
                'Свойство' : [i], \
                'среднее' : [df_0[i].mean()], \
                'медиана' : [df_0[i].median()], \
                'макс.' : [df_0[i].max()], \
                'минимум' : [df_0[i].min()], \
                'станд. откл.' : [df_0[i].std()]})
    
    if  (len(result_statistics_df) == 0):
        result_statistics_df = tmp_df.copy()
    else:
         result_statistics_df = pd.concat([result_statistics_df, tmp_df], ignore_index=True)
df_0 = pd.DataFrame(y_test, columns = outputscolumns)
for i in df_0.columns:
    tmp_df = pd.DataFrame({'Набор данных' : ['test'], \
                'Кол-во строк' : [df_0[i].count()], \
                'Свойство' : [i], \
                'среднее' : [df_0[i].mean()], \
                'медиана' : [df_0[i].median()], \
                'макс.' : [df_0[i].max()], \
                'минимум' : [df_0[i].min()], \
                'станд. откл.' : [df_0[i].std()]})
    if  (len(result_statistics_df) == 0):
         result_statistics_df = tmp_df.copy()
    else:
         result_statistics_df = pd.concat([result_statistics_df, tmp_df], ignore_index=True)





         
# Нарисуем таблицу описательной статистики
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText = result_statistics_df.values, \
                 colLabels = result_statistics_df.columns, \
                 loc = 'center')

table.auto_set_font_size(False)
table.set_fontsize(10)
#table.scale(2, 2)
table.auto_set_column_width(col = list(range(len(result_statistics_df.columns))))
#fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()






result_tests_df = pd.DataFrame(columns = ['Method Name', \
          'Par', \
          'MAE',\
          'MSE', 'RMSE', 'PercentError', \
          'MAPE','ME', \
          'R2']) 

names = ['DummyRegressor', 'LinearRegression', 'Lasso', \
         'Ridge', 'svm', 'DecisionTreeRegressor', \
         'KNeighborsRegressor', \
         'RandomForestRegressor', \
         'GradientBoosting']

regsDefault = [DummyRegressor(strategy="mean"), LinearRegression(),Lasso(), \
               Ridge(), SVR(), DecisionTreeRegressor(), \
               KNeighborsRegressor(),
               RandomForestRegressor(max_depth=2, random_state=0),
               GradientBoostingRegressor(random_state=0)]

regsOptimal = [DummyRegressor(strategy="mean"), LinearRegression(), Lasso(alpha = 0.0), \
               Ridge(alpha = 200, solver = 'sag'), SVR(C = 100, gamma = 0.0001, kernel = 'rbf'), \
               DecisionTreeRegressor(max_depth = 2), \
               KNeighborsRegressor(n_neighbors = 87), \
               RandomForestRegressor(max_depth=2, random_state=0),
               GradientBoostingRegressor(random_state=0)]
# по тесту r2 :ridge {'alpha': 10, 'solver': 'lsqr'}
# по тесту mse 'alpha': 4, 'solver': 'sag'

for method, clf in zip( names, regsDefault):
    reg = clf.fit(x_train, y_train)
    # Делаем прогноз по тесту   
    predict = reg.predict(x_test)
    #Возвращаем истинные значения
    y_test2 = y_test * MAT_NAP_MNOJ + MAT_NAP_DELTA
    predict2 = predict * MAT_NAP_MNOJ + MAT_NAP_DELTA
    ### Выводим первые 10 записей результата

    #for i in range(10): #(len(predict2)):
    #  print(str(predict2[i]), str(y_test2[i]))
    #Заполняем метрики
    mean_absolute_percentage_error =  metrics.mean_absolute_percentage_error(y_test2,predict2)
    max_error  =  metrics.max_error(y_test2,predict2)
    mean_abs_error = metrics.mean_absolute_error(y_test2,predict2) 
    mean_sq_error = metrics.mean_squared_error(y_test2,predict2)
    meansvar = np.mean(y_test2)
    sqrtmsevar = np.sqrt(metrics.mean_squared_error(y_test2,predict2))
    percenterr = (sqrtmsevar / meansvar) * 100
    #r2 = metrics.r2_score
    # append rows to an empty DataFrame
    tmp_df = pd.DataFrame({'Method Name' : [method],'Par': ['test_default'], \
                           'MAE' : [mean_abs_error],'MSE' : [mean_sq_error], \
                           'RMSE' : [sqrtmsevar],'PercentError' : [percenterr], \
                           'MAPE' : [mean_absolute_percentage_error],'ME' : [max_error],'R2' : [0.0]  })

    #print(tmp_df.head())
    #print(tmp_df.info())

    if (len(result_tests_df) == 0):
      result_tests_df = tmp_df.copy()
    else:
      result_tests_df = pd.concat([result_tests_df, tmp_df], ignore_index=True)

    #print('проверяю вставленные записи')
    #print(result_tests_df.head())
    #print(result_tests_df.info())

    print('Точность предсказаний методом ' + method) 
    print('mean_absolute_error ' +  str(mean_abs_error)) 
    print('mean_squared_error ' +  str(mean_sq_error))
    print('Root mean_squared_error ' +  \
      str(np.sqrt(sqrtmsevar)))      
    print('PercentError ' +  \
      str(percenterr) + '%') 
 
    # Делаем прогноз по train   
    predict = reg.predict(x_train)
    #Возвращаем истинные значения
    y_test2 = y_train * MAT_NAP_MNOJ + MAT_NAP_DELTA
    predict2 = predict * MAT_NAP_MNOJ + MAT_NAP_DELTA
    ### Выводим первые 10 записей результата

    #for i in range(10): #(len(predict2)):
    #  print(str(predict2[i]), str(y_test2[i]))
    #Заполняем метрики
    mean_absolute_percentage_error =  metrics.mean_absolute_percentage_error(y_test2,predict2)
    max_error  =  metrics.max_error(y_test2,predict2)
    mean_abs_error = metrics.mean_absolute_error(y_test2,predict2) 
    mean_sq_error = metrics.mean_squared_error(y_test2,predict2)
    meansvar = np.mean(y_test2)
    sqrtmsevar = np.sqrt(metrics.mean_squared_error(y_test2,predict2))
    percenterr = (sqrtmsevar / meansvar) * 100
    #r2 = metrics.r2_score
    # append rows to an empty DataFrame
    tmp_df = pd.DataFrame({'Method Name' : [method],'Par': ['train_default'], \
                           'MAE' : [mean_abs_error],'MSE' : [mean_sq_error], \
                           'RMSE' : [sqrtmsevar],'PercentError' : [percenterr], \
                           'MAPE' : [mean_absolute_percentage_error],'ME' : [max_error],'R2' : [0.0]  })

    #print(tmp_df.head())
    #print(tmp_df.info())

    if (len(result_tests_df) == 0):
      result_tests_df = tmp_df.copy()
    else:
      result_tests_df = pd.concat([result_tests_df, tmp_df], ignore_index=True)

    #print('проверяю вставленные записи')
    #print(result_tests_df.head())
    #print(result_tests_df.info())

    print('Точность предсказаний методом ' + method) 
    print('mean_absolute_error ' +  str(mean_abs_error)) 
    print('mean_squared_error ' +  str(mean_sq_error))
    print('Root mean_squared_error ' +  \
      str(np.sqrt(sqrtmsevar)))      
    print('PercentError ' +  \
      str(percenterr) + '%')





for method, clf in zip( names, regsOptimal):
  if (method != 'DummyRegressor'): 
    reg = clf.fit(x_train, y_train)
    # Делаем прогноз тесту  
    predict = reg.predict(x_test)
    #Возвращаем истинные значения
    y_test2 = y_test * MAT_NAP_MNOJ + MAT_NAP_DELTA
    predict2 = predict * MAT_NAP_MNOJ + MAT_NAP_DELTA
    ### Выводим первые 10 записей результата

    #for i in range(10): #(len(predict2)):
    #  print(str(predict2[i]), str(y_test2[i]))
    #Заполняем метрики
    mean_absolute_percentage_error =  metrics.mean_absolute_percentage_error(y_test2,predict2)
    max_error  =  metrics.max_error(y_test2,predict2)
    mean_abs_error = metrics.mean_absolute_error(y_test2,predict2) 
    mean_sq_error = metrics.mean_squared_error(y_test2,predict2)
    meansvar = np.mean(y_test2)
    sqrtmsevar = np.sqrt(metrics.mean_squared_error(y_test2,predict2))
    percenterr = (sqrtmsevar / meansvar) * 100
    #r2 = metrics.r2_score
    # append rows to an empty DataFrame
    tmp_df = pd.DataFrame({'Method Name' : [method],'Par': ['test_optimize'], \
                           'MAE' : [mean_abs_error],'MSE' : [mean_sq_error], \
                           'RMSE' : [sqrtmsevar],'PercentError' : [percenterr], \
                           'MAPE' : [mean_absolute_percentage_error],'ME' : [max_error],'R2' : [0.0]  })

    #print(tmp_df.head())
    #print(tmp_df.info())

    if (len(result_tests_df) == 0):
      result_tests_df = tmp_df.copy()
    else:
      result_tests_df = pd.concat([result_tests_df, tmp_df], ignore_index=True)

    #print('проверяю вставленные записи')
    #print(result_tests_df.head())
    #print(result_tests_df.info())

    print('Точность предсказаний методом ' + method) 
    print('mean_absolute_error ' +  str(mean_abs_error)) 
    print('mean_squared_error ' +  str(mean_sq_error))
    print('Root mean_squared_error ' +  \
      str(np.sqrt(sqrtmsevar)))      
    print('PercentError ' +  \
      str(percenterr) + '%') 

    # Делаем прогноз по train  
    predict = reg.predict(x_train)
    #Возвращаем истинные значения
    y_test2 = y_train * MAT_NAP_MNOJ + MAT_NAP_DELTA
    predict2 = predict * MAT_NAP_MNOJ + MAT_NAP_DELTA
    ### Выводим первые 10 записей результата

    #for i in range(10): #(len(predict2)):
    #  print(str(predict2[i]), str(y_test2[i]))
    #Заполняем метрики
    mean_absolute_percentage_error =  metrics.mean_absolute_percentage_error(y_test2,predict2)
    max_error  =  metrics.max_error(y_test2,predict2)
    mean_abs_error = metrics.mean_absolute_error(y_test2,predict2) 
    mean_sq_error = metrics.mean_squared_error(y_test2,predict2)
    meansvar = np.mean(y_test2)
    sqrtmsevar = np.sqrt(metrics.mean_squared_error(y_test2,predict2))
    percenterr = (sqrtmsevar / meansvar) * 100
    #r2 = metrics.r2_score
    # append rows to an empty DataFrame
    tmp_df = pd.DataFrame({'Method Name' : [method],'Par': ['train_optimize'], \
                           'MAE' : [mean_abs_error],'MSE' : [mean_sq_error], \
                           'RMSE' : [sqrtmsevar],'PercentError' : [percenterr], \
                           'MAPE' : [mean_absolute_percentage_error],'ME' : [max_error],'R2' : [0.0]  })

    #print(tmp_df.head())
    #print(tmp_df.info())

    if (len(result_tests_df) == 0):
      result_tests_df = tmp_df.copy()
    else:
      result_tests_df = pd.concat([result_tests_df, tmp_df], ignore_index=True)

    #print('проверяю вставленные записи')
    #print(result_tests_df.head())
    #print(result_tests_df.info())

    print('Точность предсказаний методом ' + method) 
    print('mean_absolute_error ' +  str(mean_abs_error)) 
    print('mean_squared_error ' +  str(mean_sq_error))
    print('Root mean_squared_error ' +  \
      str(np.sqrt(sqrtmsevar)))      
    print('PercentError ' +  \
      str(percenterr) + '%') 



#########################################################################################

###строим sklearn.tree import DecisionTreeRegressor
###regressor = DecisionTreeRegressor(max_depth=2)
###cross_val_score(regressor, x_train, y_train, cv=10)
###reg =  regressor.fit(x_train, y_train)
##
#TensorFlow

model = Sequential()
model.add(Dense(15, input_shape=(x_train[0].shape), activation='relu')) # input layer requires input_dim param
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(1, activation='linear'))

# Функция ранней остановки
stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=6)
model.compile(loss="mse", optimizer="adam", metrics=['mae'])

# call the function to fit to the data (training the network)
history = model.fit(x_train, y_train, epochs = 1000, batch_size=20, callbacks=[stop], validation_data=(x_test, y_test))



# Предсказываем на тесте
predict = model.predict(x_test)

# Возвращаем истинные значения
y_test2 = y_test * MAT_NAP_MNOJ + MAT_NAP_DELTA
predict2 = predict * MAT_NAP_MNOJ + MAT_NAP_DELTA
# Выводим первые 10 записей результата
for i in range(10): #(len(predict2)):
    print(str(predict2[i,0]), str(y_test2[i]))

method = 'TensorFlow'
mean_absolute_percentage_error =  metrics.mean_absolute_percentage_error(y_test2,predict2)
max_error  =  metrics.max_error(y_test2,predict2)
mean_abs_error = metrics.mean_absolute_error(y_test2,predict2) 
mean_sq_error = metrics.mean_squared_error(y_test2,predict2)
meansvar = np.mean(y_test2)
sqrtmsevar = np.sqrt(metrics.mean_squared_error(y_test2,predict2))
percenterr = (sqrtmsevar / meansvar) * 100
#r2 = metrics.r2_score

# append rows to an empty DataFrame
tmp_df = pd.DataFrame({'Method Name' : [method], \
                'Par': ['test_default'], \
                'MAE' : [mean_abs_error], \
                'MSE' : [mean_sq_error], \
                'RMSE' : [sqrtmsevar], \
                'PercentError' : [percenterr], \
                'MAPE' : [mean_absolute_percentage_error],\
                   'ME' : [max_error], \
                'R2' : [0.0]  \
                       })

result_tests_df = pd.concat([result_tests_df, tmp_df], ignore_index=True)
print('проверяю вставленные записи')
print(result_tests_df.head())
print(result_tests_df.info())

print('Точность предсказаний методом TensorFlow') 
print('mean_absolute_error ' +  str(metrics.mean_absolute_error(y_test2,predict2))) 
print('mean_squared_error ' +  str(metrics.mean_squared_error(y_test2,predict2)))
print('Root mean_squared_error ' +  \
      str(np.sqrt(sqrtmsevar)))      
print('PercentError ' +  \
      str(percenterr) + '%') 


# Предсказываем на train
predict = model.predict(x_train)

# Возвращаем истинные значения
y_test2 = y_train * MAT_NAP_MNOJ + MAT_NAP_DELTA
predict2 = predict * MAT_NAP_MNOJ + MAT_NAP_DELTA
# Выводим первые 10 записей результата
for i in range(10): #(len(predict2)):
    print(str(predict2[i,0]), str(y_test2[i]))

method = 'TensorFlow'
mean_absolute_percentage_error =  metrics.mean_absolute_percentage_error(y_test2,predict2)
max_error  =  metrics.max_error(y_test2,predict2)
mean_abs_error = metrics.mean_absolute_error(y_test2,predict2) 
mean_sq_error = metrics.mean_squared_error(y_test2,predict2)
meansvar = np.mean(y_test2)
sqrtmsevar = np.sqrt(metrics.mean_squared_error(y_test2,predict2))
percenterr = (sqrtmsevar / meansvar) * 100
#r2 = metrics.r2_score

# append rows to an empty DataFrame
tmp_df = pd.DataFrame({'Method Name' : [method], \
                'Par': ['train_default'], \
                'MAE' : [mean_abs_error], \
                'MSE' : [mean_sq_error], \
                'RMSE' : [sqrtmsevar], \
                'PercentError' : [percenterr], \
                'MAPE' : [mean_absolute_percentage_error],\
                   'ME' : [max_error], \
                'R2' : [0.0]  \
                       })

result_tests_df = pd.concat([result_tests_df, tmp_df], ignore_index=True)
print('проверяю вставленные записи')
print(result_tests_df.head())
print(result_tests_df.info())

print('Точность предсказаний методом TensorFlow') 
print('mean_absolute_error ' +  str(metrics.mean_absolute_error(y_test2,predict2))) 
print('mean_squared_error ' +  str(metrics.mean_squared_error(y_test2,predict2)))
print('Root mean_squared_error ' +  \
      str(np.sqrt(sqrtmsevar)))      
print('PercentError ' +  \
      str(percenterr) + '%') 




#plt.figure(figsize=(40,40))
fig, ax = plt.subplots()
plt.grid()
bp = sns.barplot(result_tests_df, x='Method Name', y='PercentError', hue = 'Par')  
ax.set_ylabel('PercentError')
ax.set_title('Процент ошибки к среднему значению')
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.subplots_adjust(left=0.2, bottom=0.45)
plt.show()

#plt.figure(figsize=(40,40))
fig, ax = plt.subplots()
plt.grid()
bp = sns.barplot(result_tests_df, x='Method Name', y='MAE', hue = 'Par')  
ax.set_ylabel('MAE')
ax.set_title('Средняя абсолютная ошибка')
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.subplots_adjust(left=0.2, bottom=0.45)
plt.show()

#plt.figure(figsize=(40,40))
fig, ax = plt.subplots()
plt.grid()
bp = sns.barplot(result_tests_df, x='Method Name', y='MSE', hue = 'Par')  
ax.set_ylabel('MSE')
ax.set_title('Среднеквадратичная ошибка')
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.subplots_adjust(left=0.2, bottom=0.45)
plt.show()

#plt.figure(figsize=(40,40))
fig, ax = plt.subplots()
plt.grid()
bp = sns.barplot(result_tests_df, x='Method Name', y='RMSE', hue = 'Par')  
ax.set_ylabel('RMSE')
ax.set_title('Корень из средней квадратичной ошибки')
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.subplots_adjust(left=0.2, bottom=0.45)
plt.show()

#plt.figure(figsize=(40,40))
fig, ax = plt.subplots()
plt.grid()
bp = sns.barplot(result_tests_df, x='Method Name', y='MAPE', hue = 'Par')  
ax.set_ylabel('MAPE')
ax.set_title('Средняя абсолютная процентная ошибка')
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.subplots_adjust(left=0.2, bottom=0.45)
plt.show()

#plt.figure(figsize=(40,40))
fig, ax = plt.subplots()
plt.grid()
bp = sns.barplot(result_tests_df, x='Method Name', y='ME', hue = 'Par')  
ax.set_ylabel('ME')
ax.set_title('Максимальная ошибка')
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.subplots_adjust(left=0.2, bottom=0.45)
plt.show()




# Нарисуем таблицу описательной статистики
result_tests_df = result_tests_df[['Method Name', \
          'Par', \
          'MAE',\
          'MSE', 'RMSE', 'PercentError', \
          'MAPE','ME']]

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText = result_tests_df.values, \
                 colLabels = result_tests_df.columns, \
                 loc = 'center')

table.auto_set_font_size(False)
table.set_fontsize(8)
#table.scale(2, 2)
fig.tight_layout()
plt.show()




print('выполнено')


                        
