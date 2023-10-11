#сохраняю модель для предсказания модуля упругости
#
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import json

dic_const_vost  = {}
with open("mnojanddelta.json", "r") as f:
     dic_const_vost = json.load(f)

#Соотношение матрица-наполнитель Delta: 0.389402605178414 Mnoj: 5.202338993519127
MAT_NAP_DELTA = dic_const_vost['Соотношение матрица-наполнитель']['delta']
MAT_NAP_MNOJ = dic_const_vost['Соотношение матрица-наполнитель']['mnozitel']


# random seed for reproducibility
np.random.seed(2)

dataset = pd.read_csv('output_data/mat_nap.csv', header = None).add_prefix("data")
#dataset.head()
dataset.info()
#print(dataset.columns)

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

dataset.columns = ['Плотность, кг/м3', \
      'модуль упругости, ГПа', 'Количество отвердителя, м.%', \
      'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2', \
      'Поверхностная плотность, г/м2', 'Модуль упругости при растяжении, ГПа', \
      'Прочность при растяжении, МПа', 'Потребление смолы, г/м2', \
      'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки', \
      'Соотношение матрица-наполнитель']



dataset.info()

values = dataset.to_numpy()


# 8 столбцов параметры, 9 столбец результат 0 или 1
# split into input (X) and output (Y) variables, splitting csv data
X = values[:,:-1] # Все кроме последненго
Y = values[:,-1] # Только последний

# split X, Y into a train and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#print(y_test)

# create model, add dense layers one by one specifying activation function
model = Sequential()
model.add(Dense(15, input_shape=(x_train[0].shape), activation='relu')) # input layer requires input_dim param
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(1, activation='linear'))

#Выводим модель в виде суммари
model.summary()

#Выводим модель в виде
tf.keras.utils.plot_model(model, 'mat_nap_modelbygraph.png', show_shapes=True)

# Функция ранней остановки
stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=6)
# compile the model, adam gradient descent (optimized)
#model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.compile(loss="mse", optimizer="adam", metrics=['mae'])



# call the function to fit to the data (training the network)
history = model.fit(x_train, y_train, epochs = 1000, batch_size=20, callbacks=[stop], validation_data=(x_test, y_test))



print(history.history.keys())


train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['mae']
val_acc = history.history['val_mae']

sns.set_theme()


plt.plot(history.history['mae'], label='train')
plt.plot(history.history['val_mae'], label='test')
plt.xlabel('Epoch')
plt.ylabel('mae')
#plt.ylim([0.7, 1])
plt.legend(['train', 'test'], loc='upper right')
#plt.legend(loc='best')
plt.show()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.legend(loc='best')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

model.save("mat_napmodel.keras")

loaded_model = tf.keras.models.load_model("mat_napmodel.keras")
#x = tf.random.uniform((10, 3))
 
#predict = loaded_model.predict(x_test)
predict = loaded_model.predict(x_test)


# Выводим первые 10 записей результата
for i in range(10): # len(predict)):
    print(str(predict[i,0]), str(y_test[i]))

print('mean_absolute_error ' +  str(metrics.mean_absolute_error(y_test,predict))) 
print('mean_squared_error ' +  str(metrics.mean_squared_error(y_test,predict)))
print('Root mean_squared_error ' +  str(np.sqrt(metrics.mean_squared_error(y_test,predict))))      
print('выполнено')

##mean_absolute_error 0.13544645458296545
##mean_squared_error 0.02880537403621034
##Root mean_squared_error 0.16972146015224573


# Возвращаем истинные значения
y_test2 = y_test * MAT_NAP_MNOJ + MAT_NAP_DELTA
predict2 = predict * MAT_NAP_MNOJ + MAT_NAP_DELTA
# Выводим первые 10 записей результата
for i in range(10): #(len(predict2)):
    print(str(predict2[i,0]), str(y_test2[i]))

meansvar = np.mean(y_test2)
sqrtmsevar = np.sqrt(metrics.mean_squared_error(y_test2,predict2))
percenterr = (sqrtmsevar / meansvar) * 100
print('mean_absolute_error ' +  str(metrics.mean_absolute_error(y_test2,predict2))) 
print('mean_squared_error ' +  str(metrics.mean_squared_error(y_test2,predict2)))
print('Root mean_squared_error ' +  \
      str(np.sqrt(sqrtmsevar)))      
print('PercentError ' +  \
      str(percenterr) + '%') 

print('выполнено')





                        
