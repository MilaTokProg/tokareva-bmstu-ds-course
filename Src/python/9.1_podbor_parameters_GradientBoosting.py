from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import Ridge
#from sklearn.svm import SVR
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.model_selection import cross_val_score
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#import tensorflow as tf
#from sklearn.dummy import DummyRegressor
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


 tuned_parameters = [{'n_estimators' : [x for x in range(50,150)]}]
                     #, 'max_depth' : [x for x in range(1,10)]}]

 #'alpha': 0.9, 'ccp_alpha': 0.0, 'criterion': 'friedman_mse',
 #'init': None, 'learning_rate': 0.1, 'loss': 'squared_error',
 #'max_depth': 3, 'max_features': None, 'max_leaf_nodes': None,
 #'min_impurity_decrease': 0.0, 'min_samples_leaf': 1,
 #'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0,
 #'n_estimators': 100, 'n_iter_no_change': None, 'random_state': 0,
 #'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1,
 #'verbose': 0, 'warm_start': False


 print(tuned_parameters)

 #scores = ['neg_mean_absolute_percentage_error', 'neg_mean_absolute_error']
 scores = ['max_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error','r2']
 ###'neg_mean_squared_error', 'roc_auc_ovr', 'neg_mean_gamma_deviance',
 ###'precision', 'neg_negative_likelihood_ratio', 'max_error', 'recall_weighted',
 ###'f1_macro', 'top_k_accuracy', 'recall_micro', 'completeness_score',
 ###'v_measure_score', 'neg_log_loss', 'precision_weighted',
 ###'adjusted_rand_score', 'roc_auc_ovo_weighted', 'neg_mean_squared_log_error',
 ###'neg_median_absolute_error', 'jaccard_micro', 'f1_micro', 'roc_auc',
 ###'neg_mean_absolute_percentage_error', 'precision_micro',
 ###'recall_samples', 'mutual_info_score', 'r2', 'neg_mean_absolute_error',
 ###'jaccard_weighted', 'normalized_mutual_info_score', 'recall_macro',
 ###'balanced_accuracy', 'positive_likelihood_ratio', 'accuracy',
 ###'jaccard_macro', 'explained_variance', 'f1', 'adjusted_mutual_info_score',
 ###'jaccard', 'precision_macro', 'roc_auc_ovo', 'f1_weighted',
 ###'roc_auc_ovr_weighted', 'f1_samples', 'fowlkes_mallows_score',
 ###'neg_mean_poisson_deviance', 'matthews_corrcoef', 'rand_score',
 ###'jaccard_samples', 'homogeneity_score', 'precision_samples',
 ###'neg_root_mean_squared_error', 'recall', 'neg_brier_score',
 ###'average_precision'
 ##
 ##
 for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        GradientBoostingRegressor(), tuned_parameters, scoring='%s' % score, cv = 10)
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
    print()
    tmp_df = pd.DataFrame({'Данные' : [task[j]], \
                'Метрика' : [score], \
                'Лучшие параметры' : [clf.best_params_]})
    if  (len(result_tests_df) == 0):
         result_tests_df = tmp_df.copy()
    else:
         result_tests_df = pd.concat([result_tests_df, tmp_df], ignore_index=True)



  


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


                        
