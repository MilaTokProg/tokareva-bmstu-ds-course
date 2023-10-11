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
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import classification_report
#from sklearn.svm import SVC


#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import seaborn as sns
#import seaborn.objects as so
#import os
#import tensorflow as tf
#from sklearn.model_selection import train_test_split
#from sklearn.utils import resample
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score
#from sklearn import linear_model
#from sklearn.metrics import confusion_matrix


# random seed for reproducibility
np.random.seed(2)

result_tests_df = pd.DataFrame(columns = ['Данные', \
          'Метрика',\
          'Лучшие параметры'])

task = ['moduprrast.csv', \
        'modprochrast.csv', \
        'mat_nap.csv']

for j in range(len(task)):

 dataset = pd.read_csv('output_data/' + task[j], header = None).add_prefix("data")
 print(dataset.head())
 values = dataset.to_numpy()


 X = values[:,:-1] # Все кроме последненго
 Y = values[:,-1] # Только последний

 # split X, Y into a train and test set
 x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

 tuned_parameters = [ {'n_neighbors' : [x for x in range(1,100)] }]
 scores = ['max_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error','r2']
 ##
 for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        KNeighborsRegressor(), tuned_parameters, scoring='%s' % score, cv = 10)
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print(task[j])
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    tmp_df = pd.DataFrame({'Данные' : [task[j]], \
                'Метрика' : [score], \
                'Лучшие параметры' : [clf.best_params_]})
    if  (len(result_tests_df) == 0):
         result_tests_df = tmp_df.copy()
    else:
         result_tests_df = pd.concat([result_tests_df, tmp_df], ignore_index=True)

    print()


# Нарисуем таблицу лучших параметров
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
    




print('выполнено')


                        
