import numpy as np
import pandas as pd
import json
import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Dense, Dropout


dic_const_vost  = {}
with open("mnojanddelta.json", "r") as f:
     dic_const_vost = json.load(f)

dic_const_fields  = {}
with open("minmaxmeans.json", "r") as f:
     dic_const_fields = json.load(f)


dic_tasks = {"1" : dict(inputfieds = ['Соотношение матрица-наполнитель', \
         'Плотность, кг/м3', 'модуль упругости, ГПа', \
         'Количество отвердителя, м.%', 'Содержание эпоксидных групп,%_2', \
         'Температура вспышки, С_2', 'Поверхностная плотность, г/м2', \
         'Потребление смолы, г/м2',  'Угол нашивки, град', 'Шаг нашивки', \
         'Плотность нашивки'], \
          outputfields = ['Модуль упругости при растяжении, ГПа', \
                          'Прочность при растяжении, МПа', ], \
          modelsdata = ['moduprmodel.keras', 'modprochrastmodel.keras']), \
             "2" : dict(inputfieds = ['Плотность, кг/м3', \
      'модуль упругости, ГПа', 'Количество отвердителя, м.%', \
      'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2', \
      'Поверхностная плотность, г/м2', 'Модуль упругости при растяжении, ГПа', \
      'Прочность при растяжении, МПа', 'Потребление смолы, г/м2', \
      'Угол нашивки, град', 'Шаг нашивки','Плотность нашивки'], \
       outputfields = ['Соотношение матрица-наполнитель'], \
       modelsdata = ['mat_napmodel.keras'] ) \
             }

print('Программа прогнозирования')
print('1 - Модуля упругости при растяжении, ГПа и Прочности при растяжении, МПа' )
print('2 - Соотношения матрицы-наполнителя' )

input_str = input("Введите число 1,2 или 0 для выхода:")
input_number = -1
if len(input_str) == 0:
    input_number = 0
else:
   try:
       input_number = int(input_str)
   except ValueError:
        print('Неверное число ' + input_str)

if input_number == 1 or input_number == 2:
  task = dic_tasks.get(str(input_number))
  inputfieds = task.get("inputfieds")
  outputfields = task.get("outputfields")
  modelsdata = task.get("modelsdata")

  task_df = pd.DataFrame(columns = inputfieds)
  result_df = pd.DataFrame(columns = outputfields)

  for i in range(len(inputfieds)):
    fieldname = inputfieds[i]
    minvalue = dic_const_fields[fieldname]["minimum"]
    maxvalue = dic_const_fields[fieldname]["maximum"]
    avgvalue = dic_const_fields[fieldname]["avg"]

    print("Введите значение " + fieldname )
    print("от " + str(minvalue) + " до " + str(maxvalue))
    print("или Enter для ввода значения по умалчанию " + str(avgvalue) )
    input_str = input("Введите число")

    if len(input_str) == 0:
      input_number = avgvalue
    else:
      try:
        input_number = float(input_str)
        print("ВВеденное число: " + str(input_number))
      except ValueError:
        print('Неверное число ' + input_str + ", будет устанвлено среднее " + str(avgvalue))
        input_number = avgvalue

    task_df[fieldname] = [input_number]


  print('вывожу введенные')
  print(task_df.head())   

  # Нормализую 
  for i in task_df.columns:
      task_df[i] = task_df[i] - dic_const_vost[i]['delta']
      task_df[i] = task_df[i] / dic_const_vost[i]['mnozitel'] 


  print(task_df.head())
  print(task_df.info())

  x_test = task_df.values

  for fn, dt in zip( outputfields, modelsdata): 
     loaded_model = tf.keras.models.load_model(dt)
     predict = loaded_model.predict(x_test)
     #print(predict)
     predict = predict[0]
     result_df[fn] = predict * dic_const_vost[fn]['mnozitel'] + dic_const_vost[fn]['delta'] 

  print(result_df.head())
  print()
  for i in task_df.columns:
      task_df[i] = task_df[i] * dic_const_vost[i]['mnozitel'] + dic_const_vost[i]['delta'] 
  print("Для входных параметров:")
  print(task_df.head())

  for i in task_df.columns:
      print(i + ": " + str(task_df[i][0]))

  print("Предсказанные значения:")   
  for i in result_df.columns:
      print(i + ": " + str(result_df[i][0]))

else:
    print('Неверный выбор задачи')
  

print('выполнено')



