##from sklearn.model_selection import train_test_split
##import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow as tf
from sklearn.dummy import DummyRegressor


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


for method, clf in zip( names, regsDefault):
    print(method)
    print(clf.get_params())
    print()

print('выполнено')


                        
