"""
create_projections.py
CSC 491 - 01 Senior Design
Author: Leo Stevens

This class creates the projections for every player and defense. 
The projections are created using sklearn machine learning libraries.
The dictionaries are used to convert values to ints which can be read by sklearn.
There are some commented out sections of this file which contain the methods used to tune the ML algos.

The flow of the projection creation is:
    create_projections (loops through all of the data frames):
        for all ML algos:
            run ML algo on data frame
            get score from projected stats
            save score into data frames
"""
import pandas as pd
import os
import math
import numpy as np
import sklearn.model_selection as ms
#import matplotlib.pyplot as plot
#import lightgbm as lgb
#import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn import neighbors
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn import metrics
from sklearn.pipeline import Pipeline
#from hyperopt import tpe, hp, fmin, Trials, STATUS_OK
#from hyperopt.pyll.base import scope
from datetime import datetime, timedelta

from sklearn.tree import export_graphviz
import pydotplus
from sklearn.tree import DecisionTreeRegressor
from subprocess import call
from player import score_offense_data
from defense import score_defense_data
#from IPython.display import Image

#These dictionaries hold integer values for all strings, boolean, ect.
#The ML algos need the data to be ints or floats
TEAMS = {'ARI':0, 'ATL':1, 'BAL':2, 'BUF':3, 'CAR':4, 'CHI':5,
         'CIN':6, 'CLE':7, 'DAL':8, 'DEN':9, 'DET':10, 'GB':11, 'HOU' :12,
         'IND' :13, 'JAC' :14, 'JAX' :14, 'KC' : 15, 'MIA' :16, 'MIN' :17,
         'NE' :18, 'NO' :19, 'NYG' :20, 'NYJ' :21, 'OAK' :22, 'PHI':23,
         'PIT' :24, 'SEA' :25, 'SF' :26, 'TB' :27, 'TEN' :28, 'WAS' : 29,
         'LAC' : 30, 'SD' : 30, 'LA' :31, 'STL' :31}

SURFACES = {'Grass':0, 'Hybrid':1, 'Turf':2}

STADIUMS = {'Open':0, 'Retractable':1, 'Fixed':2}

HOME = {False:0,'False':0, 'True':1, True:1}

CONF = {'AFC':0, 'NFC':1}

DIV = {'North':0, 'East':1, 'South':2, 'West':3}

DIV_CONF = {'AFC North':0, 'AFC East':1, 'AFC South':2, 'AFC West':3,
            'NFC North':4, 'NFC East':5, 'NFC South':6, 'NFC West':7}

GAME_TIME = {'12:30':0, '1:00':0, '4:05':1, '4:15':1, '4:25':1, '4:30':1, '8:15':2,'8:20':2, '8:25':2, '8:30':2, '9:30':3}

GAME_DAY = {'Sun':0, 'Thu':1, 'Mon':2, 'Sat':3}

CONDITIONS = {'Clear':0, 'Sunny' :0, 'Mostly Sunny':0, 'Mostly Clear':0, 'Partly Cloudy':0, 
         'Dry':0,
         'A Few Clouds':0, 'Dry and Partly Cloudy':0, 'Mostly Cloudy':1, 'Fair':1, 'Fog/Mist':1, 
         'Few Clouds':1, 'Partly Sunny':1, 'Overcast':1, 'Decreasing Clouds':1, 'Fog':1, 
         'Areas Fog':1, 'A few Clouds':1,
         'Foggy':1,'Humid and Partly Cloudy':1,'Humid and Mostly Cloudy':1, 'Possible Drizzle':2,   
         'Light Rain':2, 'Thunderstorm Light Rain':2, 'Areas Drizzle':2, 'Drizzle':2,  
         'Rain':3,
         'Rain Showers':3, 'Showers':3, 'Partly Cloudy and Breezy':4, 'Breezy':4, 
         'Overcast and Windy':4, 'Overcast and Breezy':4, 'Wintry Mix':5}


#These are the columns that have data for the next week that
#   we already know (independant variables)
KNOWN_COLUMNS_OFFENSE = ('season', 'week', 'years_played', 'opp', 'home', 'win_ratio', 
        'wins', 'losses', 'ties', 'streak',
        'def_pass_rank', 'def_rush_rank', 'def_pts_allowed', 'def_conf', 'def_div',
        'def_div_conf', 'stadium', 'surface', 'game_time', 'game_day', 'opp_win_ratio','opp_streak',
        'conditions', 'temp', 'humidity', 'visibility', 'barometric_pressure', 'dew_point')

KNOWN_COLUMNS_DEFENSE = ('season', 'week', 'opp', 'home', 'win_ratio', 'opp_win_ratio',
        'wins', 'losses', 'ties', 'streak',
        'opp_passing_rank', 'opp_rushing_rank', 'opp_scoring_rank', 'opp_conf', 'opp_div',
        'opp_div_conf', 'stadium', 'surface', 'game_time', 'game_day',
        'conditions', 'temp', 'humidity', 'visibility', 'barometric_pressure', 'dew_point')

#There are the columns that we want to predict (dependant variables)
TARGET_COLUMNS_OFFENSE = ('passing_tds', 'passing_yds', 'passing_int', 'rushing_tds', 'rushing_yds', 
        'receiving_tds', 'receiving_yds', 'receiving_rec', 'kickret_tds', 'fumbles_lost', 
        'puntret_tds','fumbles_rec_tds','receiving_twoptm', 'rushing_twoptm', 'passing_twoptm')

TARGET_COLUMNS_DEFENSE = ('defense_sk', 'defense_frec', 'defense_tds', 
        'defense_safe', 'defense_fgblk', 'defense_int', 'points_allowed')


def random_search(model, params, next_week, iters, kfold, X, Y):
    """
    This method performs and random search using inputed parameters. 
    It performs a random search and then prints out the mean squared error and cross val score.
    """
    print(model.get_params().keys())
    random_search_res = ms.RandomizedSearchCV(estimator=model, 
            param_distributions=params, 
            verbose=1,
            n_iter=iters,
            refit='AUC',
            cv=ms.KFold(kfold), 
            scoring='neg_mean_squared_error', 
            n_jobs=-1)
    random_search_res.fit(X, Y)
    best_random = random_search_res.best_estimator_
    print(random_search_res.best_score_)
    print(best_random)
    for param in random_search_res.best_params_:
        print(str(param) + ": " + str(random_search_res.best_params_[param]))
    best_random.fit(X, Y)
    next_week[Y.columns] = best_random.predict(next_week[X.columns])
    y_pred_train = best_random.predict(X)
    print(next_week)
    print("Trained Score: ", best_random.score(X, Y) * 100)
    print("Trained MSE: ", metrics.mean_squared_error(Y, y_pred_train))
    print("Trained CV: ", ms.cross_val_score(best_random, Y, y_pred_train, cv=10, scoring='neg_mean_squared_error'))

    return best_random

def grid_search(model, params, next_week, kfold, X, Y):
    """
    This method performs a grid search using inputed parameters.
    It performs the random search and then prints out the mean squared error and cross val score.
    """
    print(model.get_params().keys())
    random_search_res = ms.GridSearchCV(estimator=model, 
            param_grid=params, 
            verbose=1,
            refit='AUC',
            cv=ms.KFold(kfold), 
            scoring='neg_mean_squared_error', 
            n_jobs=-1)
    random_search_res.fit(X, Y)
    best_random = random_search_res.best_estimator_
    print(random_search_res.best_score_)
    for param in random_search_res.best_params_:
        print(str(param) + ": " + str(random_search_res.best_params_[param]))
    best_random.fit(X, Y)
    next_week[Y.columns] = best_random.predict(next_week[X.columns])
    y_pred_train = best_random.predict(X)
    print(next_week)
    print("Trained Score: ", best_random.score(X, Y) * 100)
    print("Trained MSE: ", metrics.mean_squared_error(Y, y_pred_train))
    print("Trained CV: ", ms.cross_val_score(best_random, Y, y_pred_train, cv=10, scoring='neg_mean_squared_error'))
    return best_random

def prepare_offense_data(df):
    """
    This method loads the dataframe for offenseive players, 
        filters out unused columns, replaces non-int data using the
        dictionaries, and gets the next week.
    """
    #df = pd.read_csv(csvfile)
    #print(df['conditions'])
    allColumns = KNOWN_COLUMNS_OFFENSE + TARGET_COLUMNS_OFFENSE
    mlDataframe = df.filter(allColumns)
    mlDataframe = mlDataframe.replace(TEAMS)
    mlDataframe = mlDataframe.replace(SURFACES)
    mlDataframe = mlDataframe.replace(STADIUMS)
    mlDataframe = mlDataframe.replace(HOME)
    mlDataframe = mlDataframe.replace(CONF)
    mlDataframe = mlDataframe.replace(DIV)
    mlDataframe = mlDataframe.replace(DIV_CONF)
    mlDataframe = mlDataframe.replace(GAME_TIME)
    mlDataframe = mlDataframe.replace(GAME_DAY)
    mlDataframe = mlDataframe.replace(CONDITIONS)
    #mlDataframe['conditions'] = pd.to_numeric(mlDataframe['conditions'], errors='coerce')
    next_week = mlDataframe.tail(1)
    mlDataframe = mlDataframe[:-1]
    mlDataframe = mlDataframe.apply(pd.to_numeric, errors = 'coerce')
    mlDataframe.fillna(0, inplace=True)
    return df, mlDataframe, next_week

def prepare_defense_data(df):
    """
    This method loads the data frame for team defenses, 
        filters out unused columns, replaces non-int data using the
        dictionaries, and gets the next week.
    """
    #df = pd.read_csv(csvfile)
    #print(df['conditions'])
    allColumns = KNOWN_COLUMNS_DEFENSE + TARGET_COLUMNS_DEFENSE
    mlDataframe = df.filter(allColumns)
    mlDataframe = mlDataframe.replace(TEAMS)
    mlDataframe = mlDataframe.replace(SURFACES)
    mlDataframe = mlDataframe.replace(STADIUMS)
    mlDataframe = mlDataframe.replace(HOME)
    mlDataframe = mlDataframe.replace(CONF)
    mlDataframe = mlDataframe.replace(DIV)
    mlDataframe = mlDataframe.replace(DIV_CONF)
    mlDataframe = mlDataframe.replace(GAME_TIME)
    mlDataframe = mlDataframe.replace(GAME_DAY)
    mlDataframe = mlDataframe.replace(CONDITIONS)
    #mlDataframe['conditions'] = pd.to_numeric(mlDataframe['conditions'], errors='coerce')
    next_week = mlDataframe.tail(1)
    mlDataframe = mlDataframe[:-1]
    mlDataframe = mlDataframe.apply(pd.to_numeric, errors = 'coerce')
    mlDataframe.fillna(0, inplace=True)
    return df, mlDataframe, next_week

def process_data(df, next_week, test_size = 0): 
    """
    This method creates the train, test, X, and Y splits using sklearn.
    """
    set_cells = next_week.loc[:,next_week.notnull().any(axis=0)]
    empty_cells = next_week.loc[:,next_week.isna().any(axis=0)].copy()
    X_train, X_test, Y_train, Y_test = ms.train_test_split(df[set_cells.columns], 
            df[empty_cells.columns], 
            test_size=test_size, 
            random_state=4)
    return X_train, X_test, Y_train, Y_test

"""
All of the machine learning methods have the same structure:
    make_prediction(dataframe, next_week, debug)
        Create dictionary of values used in parameters
        Process the data frames using process_data
        Declare model (using MultiOutputRegressor if the algo is not natively multivariate)
        Set the parameters
        Fit the model
        Use the model to make a prediction for each value in next_week we want
        Print debugging info if set

I am still fine-tuning the parameters for the algos so the tuning data is still in the code.
The space dictionary holds the values used in grid_search
The rand_space dictionary holds the values used in random_search
The params_old dictionary is the experimental set of the parameters I have been trying
"""

def make_lr_pred(df, next_week, debug = 0): 
    """
    This method creates predictions using linear regression.
    """
    #Tuned
    space = {'estimator__fit_intercept': [True, False],
            'estimator__normalize': [True,False]}
    params = {'estimator__fit_intercept': True,
            'estimator__normalize': False}
    X_train, X_test, Y_train, Y_test = process_data(df, next_week)
    multi_lr = MultiOutputRegressor(LinearRegression())
    #best_random = grid_search(multi_lr, space, next_week, 10, X_train, Y_train)
    multi_lr.set_params(**params)
    multi_lr.fit(X_train, Y_train)
    next_week[Y_train.columns] = multi_lr.predict(next_week[X_train.columns])  
    y_pred_untrain = multi_lr.predict(X_train)
    if debug:
        print(next_week)
        print("Score: ", multi_lr.score(X_train, Y_train) * 100)
        print("MSE: ", metrics.mean_squared_error(Y_train, y_pred_untrain)) 
        print("CV: ", ms.cross_val_score(multi_lr, Y_train, y_pred_untrain, cv=10, scoring='neg_mean_squared_error'))
    return next_week

def make_rf_pred(df, next_week, debug = 0):
    """
    This method creates predictions using random forest.
    """
    #Tuned##
    params_old={'estimator__bootstrap': True,
            'estimator__max_depth': 5,
            'estimator__max_features': 'sqrt',
            'estimator__random_state': 4,
            'estimator__min_samples_leaf': 9,
            'estimator__min_samples_split': 20,
            'estimator__n_estimators': 800} 

    params={'estimator__bootstrap': False,
            'estimator__max_depth': 3,
            'estimator__max_features': 'sqrt',
            'estimator__random_state': 4,
            'estimator__min_samples_leaf': 1,
            'estimator__min_samples_split': 2,
            'estimator__n_jobs':-1,
            'n_jobs':-1,
            'estimator__n_estimators': 200}
    rand_space={'estimator__bootstrap': [True, False],
            'estimator__max_depth': [int(x) for x in np.linspace(10,110, num=11)],
            'estimator__max_features': ['auto', 'sqrt'],
            'estimator__random_state': [4],
            'estimator__min_samples_leaf': [1, 2, 4, 8],#132
            'estimator__min_samples_split': [2, 5, 10],#396
            'estimator__n_estimators': [int(x) for x in np.linspace(200, 2000, num=10)]}#3960
    space={'estimator__bootstrap': [True],
            'estimator__max_depth': [5],
            'estimator__max_features': ['sqrt'],
            'estimator__random_state': [4],
            'estimator__min_samples_leaf': [9],#132
            'estimator__min_samples_split': [15, 20, 25],#396
            'estimator__n_estimators': [800]}#3960
    X_train, X_test, Y_train, Y_test = process_data(df, next_week)
    multi_rf = MultiOutputRegressor(RandomForestRegressor())
    multi_rf.set_params(**params)
    #best_random = random_search(multi_rf, rand_space, next_week, 100, 3, X_train, Y_train), 
    #best_random = grid_search(multi_rf, space, next_week, 3, X_train, Y_train)
    multi_rf.fit(X_train, Y_train)
    next_week[Y_train.columns] = multi_rf.predict(next_week[X_train.columns])
    if debug:
        y_pred_untrain = multi_rf.predict(X_train)
        print(next_week.to_string())
        print("Score: ", multi_rf.score(X_train, Y_train) * 100)
        print("MSE: ", metrics.mean_squared_error(Y_train, y_pred_untrain))
        print("CV: ", ms.cross_val_score(multi_rf, Y_train, y_pred_untrain, cv=10, scoring='neg_mean_squared_error'))
    return next_week

def make_ridge_pred(df, next_week, debug = 0):
    """
    This method creates predictions using ridge regression.
    """
    #Tuned##
    params_old =  {'alpha':10, 
            'max_iter':1,
            'solver':'cholesky',
            'normalize':True, 
            'fit_intercept':True}
    params =  {'alpha':10, 
            'max_iter':-1,
            'solver':'sparse_cg',
            'normalize':False, 
            'fit_intercept':True}
    rand_space = {'alpha':[1e-10, 1e-5, 1e-2, 1e-1, 1, 10, 100],
            'normalize':[True, False],
            'fit_intercept':[True,False],
            'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'max_iter':[int(x) for x in np.linspace(1, 1000, num=10)]}

    space = {'alpha':[8, 9, 10, 11],
            'normalize':[True, False],
            'fit_intercept':[True, False],
            'solver':['cholesky'],
            'max_iter':[None, 1, 2, 10]}
    X_train, X_test, Y_train, Y_test = process_data(df, next_week)
    multi_ridge = Ridge()
    multi_ridge.set_params(**params)
    #best_random = random_search(multi_ridge, rand_space, next_week, 100, 3, X_train, Y_train)
    #best_random = grid_search(multi_ridge, space, next_week, 3, X_train, Y_train)
    multi_ridge.fit(X_train, Y_train)
    next_week[Y_train.columns] = multi_ridge.predict(next_week[X_train.columns])
    if debug:
        y_pred_untrain = multi_ridge.predict(X_train)
        print(next_week)
        print("Score: ", multi_ridge.score(X_train, Y_train) * 100)
        print("MSE: ", metrics.mean_squared_error(Y_train, y_pred_untrain))
        print("CV: ", ms.cross_val_score(multi_ridge, Y_train, y_pred_untrain, cv=10, scoring='neg_mean_squared_error'))
    return next_week

def make_lasso_pred(df, next_week, debug = 0):
    """
    This method makes predictions using lasso regression.
    """
    #Tuned##
    rand_space = {'estimator__alpha':[900, 1000, 1100],
            'estimator__normalize':[True, False],
            'estimator__fit_intercept':[True, False],
            'estimator__positive':[True, False],
            'estimator__max_iter':[10000, 50000, 100000]}

    space = {'estimator__alpha':[3,4,5],
            'estimator__normalize':[True],
            'estimator__fit_intercept':[True],
            'estimator__positive':[False],
            'estimator__max_iter':[1]}
    params_old = {'estimator__alpha':3,
            'estimator__normalize':True,
            'estimator__fit_intercept':True,
            'estimator__positive':False,
            'estimator__max_iter':1}
    params = {'estimator__alpha':10,
            'estimator__normalize':False,
            'estimator__fit_intercept':True,
            'n_jobs':-1,
            'estimator__positive':False,
            'estimator__max_iter':750}
            #'estimator__max_iter':10}
    X_train, X_test, Y_train, Y_test = process_data(df, next_week)
    multi_lasso = MultiOutputRegressor(Lasso())
    multi_lasso.set_params(**params)
    #best_random = random_search(multi_lasso, rand_space, next_week, 50, 3, X_train, Y_train)
    #best_random = grid_search(multi_lasso, space, next_week, 3, X_train, Y_train)
    multi_lasso.fit(X_train, Y_train)
    next_week[Y_train.columns] = multi_lasso.predict(next_week[X_train.columns]) 
    if debug:
        y_pred_untrain = multi_lasso.predict(X_train)
        print(next_week)
        print("Score: ", multi_lasso.score(X_train, Y_train) * 100)
        print("MSE: ", metrics.mean_squared_error(Y_train, y_pred_untrain))
        print("CV: ", ms.cross_val_score(multi_lasso, Y_train, y_pred_untrain, cv=10, scoring='neg_mean_squared_error'))
    return next_week

def make_nn_pred(df, next_week, debug = 0):
    """
    This method creates predicitions using a neural network.
    """
    #Tuned##
    rand_space = {'hidden_layer_sizes': [(100,), (500,), (1000,)],
            'activation':['tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'shuffle':[True,False],
            'alpha': [1, 10, 100, 500],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'max_iter': [1, 10, 1000, 10000, 100000],
            'early_stopping':[True, False]}
    space = {'hidden_layer_sizes': [(500,), (550,), (600,), (650,), (700,)],
            'activation':['tanh'],
            'solver': ['lbfgs'],
            'shuffle':[False],
            'alpha': [5, 6,7,8],
            'learning_rate': ['constant'],
            'max_iter': [4,5,6,7],
            'early_stopping':[True]}
    params_old = {'hidden_layer_sizes': (600,),
            'activation':'tanh',
            'solver': 'lbfgs',
            'alpha': 4,
            'learning_rate': 'constant',
            'max_iter': 6,
            'shuffle':False,
            'early_stopping':True}
    params = {'hidden_layer_sizes': (1000,),
            'activation':'tanh',
            'solver': 'lbfgs',
            'alpha': 1,
            'learning_rate': 'invscaling',
            'max_iter': 10000,
            'early_stopping':True}
    X_train, X_test, Y_train, Y_test = process_data(df, next_week)
    multi_nnr = MLPRegressor()
    multi_nnr.set_params(**params)
    #best_random = random_search(multi_nnr, rand_space, next_week, 100, 3, X_train, Y_train)
    #best_random = grid_search(multi_nnr, space, next_week, 3, X_train, Y_train)
    multi_nnr.fit(X_train, Y_train)
    next_week[Y_train.columns] = multi_nnr.predict(next_week[X_train.columns])
    if debug:
        y_pred_untrain = multi_nnr.predict(X_train)
        print(next_week)
        print("Score: ", multi_nnr.score(X_train, Y_train) * 100)
        print("MSE: ", metrics.mean_squared_error(Y_train, y_pred_untrain))
        print("CV: ", ms.cross_val_score(multi_nnr, Y_train, y_pred_untrain, cv=10, scoring='neg_mean_squared_error'))
    return next_week

def make_bayesian_pred(df, next_week, debug = 0):
    """
    This method creates predictions using bayesian regression.
    """
    space = {'estimator__alpha_1': [1e-10, 1e-5, 1],
            'estimator__alpha_2': [1e-10, 1e-5, 1],
            'estimator__lambda_1': [1e-10, 1e-5, 1],
            'estimator__lambda_2': [1e-10, 1e-5, 1],
            'estimator__n_iter': [10, 300, 1000],
            'estimator__normalize':[True, False],
            'estimator__fit_intercept':[True, False]}
    params = {'estimator__alpha_1': [1e-10, 1e-5, 1, 5],
            'estimator__alpha_2': [1e-10, 1e-5, 1, 5],
            'estimator__lambda_1': [1e-10, 1e-5, 1, 5],
            'estimator__lambda_2': [1e-10, 1e-5, 1, 5],
            'estimator__n_iter': [10, 300, 1000],
            'estimator__normalize':[True, False],
            'estimator__n_jobs':-1,
            'n_jobs':-1,
            'estimator__fit_intercept':[True, False]}
    X_train, X_test, Y_train, Y_test = process_data(df, next_week)
    multi_bay = MultiOutputRegressor(BayesianRidge())
    #multi_bay.set_params(**params)
    #best_random = grid_search(multi_bay, space, next_week, 3, X_train, Y_train)
    multi_bay.fit(X_train, Y_train)
    next_week[Y_train.columns] = multi_bay.predict(next_week[X_train.columns]) 
    if debug:
        y_pred_untrain = multi_bay.predict(X_train)
        print(next_week)
        print("Score: ", multi_bay.score(X_train, Y_train) * 100)
        print("MSE: ", metrics.mean_squared_error(Y_train, y_pred_untrain))
        print("CV: ", ms.cross_val_score(multi_bay, Y_train, y_pred_untrain, cv=10, scoring='neg_mean_squared_error'))
    return next_week

def make_knn_pred(df, next_week, debug = 0):
    """
    This method creates predictions using k-nearest neighbors.
    """
    #Tuned##
    rand_space = {'estimator__n_neighbors': [5, 10, 15],
            'estimator__weights': ['uniform', 'distance'],
            'estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'estimator__leaf_size': [50, 100, 150, 200],
            'estimator__p': [1, 2, 3]}
    space = {'estimator__n_neighbors': [14,15,16],
            'estimator__weights': ['distance'],
            'estimator__algorithm': ['auto','brute'],
            'estimator__leaf_size': [50,90,100,110,150],
            'estimator__p': [1]}
    params_old = {'estimator__n_neighbors': 15,
            'estimator__weights': 'distance',
            'estimator__algorithm': 'brute',
            'estimator__leaf_size': 50,
            'estimator__p': 1}
    params = {'estimator__n_neighbors': 10,
            'estimator__weights': 'uniform',
            'estimator__algorithm': 'auto',
            'estimator__leaf_size': 1,
            'estimator__n_jobs':-1,
            'n_jobs':-1,
            'estimator__p': 1}
    X_train, X_test, Y_train, Y_test = process_data(df, next_week)
    multi_knn = MultiOutputRegressor(neighbors.KNeighborsRegressor())
    multi_knn.set_params(**params)
    #best_random = random_search(multi_knn, rand_space, next_week, 100, 3, X_train, Y_train)
    #best_random = grid_search(multi_knn, space, next_week, 3, X_train, Y_train)
    try:
        multi_knn.fit(X_train, Y_train)
        next_week[Y_train.columns] = multi_knn.predict(next_week[X_train.columns]) 
    except ValueError as error:
        params = {'estimator__n_neighbors': len(df.index) - 1,
                #'verbose':0,
                'estimator__weights': 'distance',
                'estimator__algorithm': 'brute',
                'estimator__leaf_size': 50,
                'estimator__p': 1}
        multi_knn.set_params(**params)
        multi_knn.fit(X_train, Y_train)
        next_week[Y_train.columns] = multi_knn.predict(next_week[X_train.columns]) 
    if debug:
        y_pred_untrain = multi_knn.predict(X_train)
        print(next_week)
        print("Score: ", multi_knn.score(X_train, Y_train) * 100)
        print("MSE: ", metrics.mean_squared_error(Y_train, y_pred_untrain))
        print("CV: ", ms.cross_val_score(multi_knn, Y_train, y_pred_untrain, cv=10, scoring='neg_mean_squared_error'))
    return next_week

def make_svr_pred(df, next_week, debug = 0):
    """
    This method creates predictions using support vector regression.
    """
    #Tuned##
    rand_space = {'estimator__kernel': ['linear', 'rbf', 'sigmoid'],
            'estimator__gamma': ['auto', 1e-10, 1e-6, 0.9],
            'estimator__epsilon': [1e-10, 1e-6, 0.1, 1],
            'estimator__C': [1e-2, 1, 10],
            'estimator__shrinking': [True, False],
            'estimator__max_iter': [-1, 1, 5, 10, 100, 1000]}  


    space = {'estimator__kernel': ['linear', 'rbf', 'sigmoid'],
            'estimator__gamma': ['auto'],
            'estimator__epsilon': [1e-10,1e-9,1e-8, 1e-7, 1e-6],
            'estimator__C': [4,5,6],
            'estimator__shrinking': [False],
            'estimator__max_iter': [19,20,21]}  

    params_old = {'estimator__kernel': 'rbf',
            'estimator__gamma': 'auto',
            'estimator__epsilon': 1e-8,
            'estimator__C': 5,
            'estimator__shrinking': False,
            'estimator__max_iter': 20}
    params = {'estimator__kernel': 'linear',
            'estimator__gamma': 'auto',
            'estimator__epsilon': 1e-10,
            'estimator__C': 1e-2,
            'n_jobs':-1,
            'estimator__shrinking': False,
            'estimator__max_iter': -1}
    X_train, X_test, Y_train, Y_test = process_data(df, next_week)
    multi_svr = MultiOutputRegressor(SVR())
    multi_svr.set_params(**params)
    #best_random = random_search(multi_svr, rand_space, next_week, 100, 3, X_train, Y_train)
    #best_random = grid_search(multi_svr, space, next_week, 3, X_train, Y_train)
    multi_svr.fit(X_train, Y_train)
    next_week[Y_train.columns] = multi_svr.predict(next_week[X_train.columns]) 
    if debug:
        y_pred_untrain = multi_svr.predict(X_train)
        print(next_week)
        print("Score: ", multi_svr.score(X_train, Y_train) * 100)
        print("MSE: ", metrics.mean_squared_error(Y_train, y_pred_untrain))
        print("CV: ", ms.cross_val_score(multi_svr, Y_train, y_pred_untrain, cv=10, scoring='neg_mean_squared_error'))
    return next_week

def make_gb_pred(df, next_week, debug = 0):
    """
    This method creates predictions using gradient boosting regression.
    """
    #Tuned##
    rand_space = {'estimator__alpha':[1e-6, 1e-5, 1e-4],
            'estimator__learning_rate': [0.4,0.5,0.6],
            'estimator__loss':['ls', 'lad', 'huber', 'quantile'],
            'estimator__n_estimators':[500, 1000, 1500],
            'estimator__max_leaf_nodes':[50, 100, 200],
            'estimator__min_samples_split':[4,5,6],
            'estimator__min_samples_leaf':[5,10,50],
            'estimator__min_weight_fraction_leaf':[0.4,0.5],
            'estimator__max_depth':[5,10,50],
            'estimator__max_features':['auto', 'sqrt', None, 1, 5]} 


    space = {'estimator__alpha':[0.6],
            'estimator__learning_rate': [0.5],
            'estimator__loss':['ls'],
            'estimator__n_estimators':[1000],
            'estimator__max_leaf_nodes':[36,37,38,39],
            'estimator__min_samples_split':[4],
            'estimator__min_samples_leaf':[10],
            'estimator__min_weight_fraction_leaf':[0.5],
            'estimator__max_depth':[14],
            'estimator__max_features':[1]} 

    params_old = {
            'estimator__alpha':0.6,
            'estimator__learning_rate': 0.5,
            'estimator__loss': 'ls',
            'estimator__n_estimators': 1000,
            'estimator__max_leaf_nodes':38,
            'estimator__min_samples_split': 4,
            'estimator__min_samples_leaf': 10,
            'estimator__min_weight_fraction_leaf': 0.5,
            'estimator__max_depth': 14,
            'estimator__max_features': 1}  
    params = {
            'estimator__learning_rate': 0.9,
            'estimator__loss': 'ls',
            'estimator__n_estimators': 1,
            'estimator__max_leaf_nodes':50,
            'estimator__min_samples_split': 10,
            'estimator__min_samples_leaf': 5,
            'estimator__min_weight_fraction_leaf': 0.2,
            'n_jobs':-1,
            'estimator__max_depth': 10,
            'estimator__max_features': 5}   
    X_train, X_test, Y_train, Y_test = process_data(df, next_week)
    multi_gbr = MultiOutputRegressor(GradientBoostingRegressor())
    #best_random = random_search(multi_gbr, rand_space, next_week, 200, 3, X_train, Y_train)
    #best_random = grid_search(multi_gbr, space, next_week, 3, X_train, Y_train)
    multi_gbr.set_params(**params)
    multi_gbr.fit(X_train, Y_train)
    next_week[Y_train.columns] = multi_gbr.predict(next_week[X_train.columns]) 
    if debug:
        y_pred_untrain = multi_gbr.predict(X_train)
        print(next_week.to_string())
        print("Score: ", multi_gbr.score(X_train, Y_train) * 100)
        print("MSE: ", metrics.mean_squared_error(Y_train, y_pred_untrain))
        print("CV: ", ms.cross_val_score(multi_gbr, Y_train, y_pred_untrain, cv=3, scoring='neg_mean_squared_error'))
    return next_week

def make_elastic_pred(df, next_week, debug = 0):
    """
    This method creates predictions using elastic net regression.
    """
    #Tuned##
    rand_space = {'estimator__alpha': [1e-1],
            'estimator__l1_ratio': [0.7,0.8],
            'estimator__fit_intercept': [True],
            'estimator__normalize': [True],
            'estimator__precompute': [False],
            'estimator__positive': [True],
            'estimator__max_iter': [11000,12000,13000],
            'estimator__selection': ['random']}
    space = {'estimator__alpha': [1e-5, 1e-1, 1, 10],
            'estimator__l1_ratio': [0, 0.25, 0.5, 0.75, 1],
            'estimator__fit_intercept': [True, False],
            'estimator__normalize': [True, False],
            'estimator__precompute': [True, False],
            'estimator__positive': [True, False],
            'estimator__max_iter': [10, 100, 1000, 10000],
            'estimator__selection': ['cyclic', 'random']}
    params_old = {'estimator__alpha': 0.1,
            'estimator__l1_ratio': 0.7,
            'estimator__fit_intercept': True,
            'estimator__normalize': True,
            'estimator__precompute': False,
            'estimator__positive': True,
            'estimator__max_iter': 11000,
            'estimator__selection': 'random'}
    params = {'estimator__alpha': 10,
            'estimator__l1_ratio': 1,
            'estimator__fit_intercept': True,
            'estimator__normalize': False,
            'estimator__precompute': True,
            'n_jobs':-1,
            'estimator__positive': True,
            #'estimator__max_iter': 10,
            'estimator__max_iter': 500,
            'estimator__selection': 'random'}
    X_train, X_test, Y_train, Y_test = process_data(df, next_week)
    multi_en = MultiOutputRegressor(ElasticNet())
    multi_en.set_params(**params)
    #best_random = random_search(multi_en, rand_space, next_week, 100, 3, X_train, Y_train)
    #best_random = grid_search(multi_en, rand_space, next_week, 3, X_train, Y_train)
    multi_en.fit(X_train, Y_train)
    next_week[Y_train.columns] = multi_en.predict(next_week[X_train.columns]) 
    if debug:
        y_pred_untrain = multi_en.predict(X_train)
        print(next_week)
        print("Score: ", multi_en.score(X_train, Y_train) * 100)
        print("MSE: ", metrics.mean_squared_error(Y_train, y_pred_untrain))
        print("CV: ", ms.cross_val_score(multi_en, Y_train, y_pred_untrain, cv=10, scoring='neg_mean_squared_error'))
    return next_week

#ML_FUNCTIONS = (make_lr_pred, make_rf_pred, make_ridge_pred, make_lasso_pred,
#        make_nn_pred, make_knn_pred, make_svr_pred, make_gb_pred, make_elastic_pred)

def build_projections(players, defense):
    """
    This method builds all the projections using the ml algos. 
    It has a very basic estimated time of completion. 
    """
    #Set variables for eta
    time_remaining = '0:00:00'
    start = datetime.now()
    stop = datetime.now()
    current_count = 1
    all_players = {}
    all_players.update(players)
    all_players.update(defense)
    total_files = len(all_players)

    #Load all data frames (Loop through all players)
    for player in all_players:
        if 'player' in  str(all_players[player].__class__):
            name = all_players[player].name + " " + all_players[player].team.name
        else:
            name = all_players[player].team.name + " Defense"
        if current_count is not 1:
            time_remaining = stop - start
            time_remaining = float(float(str(time_remaining.seconds) + "." + str(time_remaining.microseconds)) / (current_count))
            time_remaining = int(time_remaining * (total_files - current_count))
            time_remaining = timedelta(seconds=time_remaining)
            print("[" + str(current_count) + "/" + str(total_files) + "](" + str(time_remaining) + " remaining) Creating projections for: " + str(name))
        else:
            print("[" + str(current_count) + "/" + str(total_files) + "](" + str(time_remaining) + " remaining) Creating projections for: " + str(name))
        offense = False
        if 'player' in str(all_players[player].__class__):
            offense = True
            df, mlDataframe, next_week = prepare_offense_data(all_players[player].df)
        else:
            df, mlDataframe, next_week = prepare_defense_data(all_players[player].df)
        #Try-Catch to check if the player has data for next week
        try:
            #Make all predictions
            #print 'rf'
            next_week_rf = make_rf_pred(mlDataframe.copy(), next_week.copy(), 0)
            #print 'ridge'
            next_week_ridge = make_ridge_pred(mlDataframe.copy(), next_week.copy(), 0)
            #print 'lasso'
            next_week_lasso = make_lasso_pred(mlDataframe.copy(), next_week.copy(), 0)
            #print 'nn'
            next_week_nn = make_nn_pred(mlDataframe.copy(), next_week.copy(), 0)
            #print 'knn'
            next_week_knn = make_knn_pred(mlDataframe.copy(), next_week.copy(), 0)
            #print 'svr'
            next_week_svr = make_svr_pred(mlDataframe.copy(), next_week.copy(), 0)
            #print 'gb'
            next_week_gb = make_gb_pred(mlDataframe.copy(), next_week.copy(), 0)
            #print 'en'
            next_week_elastic = make_elastic_pred(mlDataframe.copy(), next_week.copy(), 0)

            #Get scores from predictions
            if offense:
                rf_pro_fd, rf_pro_dk = score_offense_data(next_week_rf)
                ridge_pro_fd, ridge_pro_dk = score_offense_data(next_week_ridge)
                lasso_pro_fd, lasso_pro_dk = score_offense_data(next_week_lasso)
                nn_pro_fd, nn_pro_dk = score_offense_data(next_week_nn)
                knn_pro_fd, knn_pro_dk = score_offense_data(next_week_knn)
                svr_pro_fd, svr_pro_dk = score_offense_data(next_week_svr)
                gb_pro_fd, gb_pro_dk = score_offense_data(next_week_gb)
                elastic_pro_fd, elastic_pro_dk = score_offense_data(next_week_elastic)
            else:
                rf_pro_fd, rf_pro_dk = score_defense_data(next_week_rf)
                ridge_pro_fd, ridge_pro_dk = score_defense_data(next_week_ridge)
                lasso_pro_fd, lasso_pro_dk = score_defense_data(next_week_lasso)
                nn_pro_fd, nn_pro_dk = score_defense_data(next_week_nn)
                knn_pro_fd, knn_pro_dk = score_defense_data(next_week_knn)
                svr_pro_fd, svr_pro_dk = score_defense_data(next_week_svr)
                gb_pro_fd, gb_pro_dk = score_defense_data(next_week_gb)
                elastic_pro_fd, elastic_pro_dk = score_defense_data(next_week_elastic)
            #Save scores in original data frame
            df.iloc[-1, df.columns.get_loc('randforest_pro_dk')] = rf_pro_dk[0]
            df.iloc[-1, df.columns.get_loc('randforest_pro_dk_floor')] = rf_pro_dk[1]
            df.iloc[-1, df.columns.get_loc('randforest_pro_dk_ceiling')] = rf_pro_dk[2]
            df.iloc[-1, df.columns.get_loc('randforest_pro_fd')] = rf_pro_fd[0]
            df.iloc[-1, df.columns.get_loc('randforest_pro_fd_floor')] = rf_pro_fd[1]
            df.iloc[-1, df.columns.get_loc('randforest_pro_fd_ceiling')] = rf_pro_fd[2]
            df.iloc[-1, df.columns.get_loc('ridge_pro_dk')] = ridge_pro_dk[0]
            df.iloc[-1, df.columns.get_loc('ridge_pro_dk_floor')] = ridge_pro_dk[1]
            df.iloc[-1, df.columns.get_loc('ridge_pro_dk_ceiling')] = ridge_pro_dk[2]
            df.iloc[-1, df.columns.get_loc('ridge_pro_fd')] = ridge_pro_fd[0]
            df.iloc[-1, df.columns.get_loc('ridge_pro_fd_floor')] = ridge_pro_fd[1]
            df.iloc[-1, df.columns.get_loc('ridge_pro_fd_ceiling')] = ridge_pro_fd[2]
            df.iloc[-1, df.columns.get_loc('lasso_pro_dk')] = lasso_pro_dk[0]
            df.iloc[-1, df.columns.get_loc('lasso_pro_dk_floor')] = lasso_pro_dk[1]
            df.iloc[-1, df.columns.get_loc('lasso_pro_dk_ceiling')] = lasso_pro_dk[2]
            df.iloc[-1, df.columns.get_loc('lasso_pro_fd')] = lasso_pro_fd[0]
            df.iloc[-1, df.columns.get_loc('lasso_pro_fd_floor')] = lasso_pro_fd[1]
            df.iloc[-1, df.columns.get_loc('lasso_pro_fd_ceiling')] = lasso_pro_fd[2]
            df.iloc[-1, df.columns.get_loc('nn_pro_dk')] = nn_pro_dk[0]
            df.iloc[-1, df.columns.get_loc('nn_pro_dk_floor')] = nn_pro_dk[1]
            df.iloc[-1, df.columns.get_loc('nn_pro_dk_ceiling')] = nn_pro_dk[2]
            df.iloc[-1, df.columns.get_loc('nn_pro_fd')] = nn_pro_fd[0]
            df.iloc[-1, df.columns.get_loc('nn_pro_fd_floor')] = nn_pro_fd[1]
            df.iloc[-1, df.columns.get_loc('nn_pro_fd_ceiling')] = nn_pro_fd[2]
            df.iloc[-1, df.columns.get_loc('knn_pro_dk')] = knn_pro_dk[0]
            df.iloc[-1, df.columns.get_loc('knn_pro_dk_floor')] = knn_pro_dk[1]
            df.iloc[-1, df.columns.get_loc('knn_pro_dk_ceiling')] = knn_pro_dk[2]
            df.iloc[-1, df.columns.get_loc('knn_pro_fd')] = knn_pro_fd[0]
            df.iloc[-1, df.columns.get_loc('knn_pro_fd_floor')] = knn_pro_fd[1]
            df.iloc[-1, df.columns.get_loc('knn_pro_fd_ceiling')] = knn_pro_fd[2]
            df.iloc[-1, df.columns.get_loc('svr_pro_dk')] = svr_pro_dk[0]
            df.iloc[-1, df.columns.get_loc('svr_pro_dk_floor')] = svr_pro_dk[1]
            df.iloc[-1, df.columns.get_loc('svr_pro_dk_ceiling')] = svr_pro_dk[2]
            df.iloc[-1, df.columns.get_loc('svr_pro_fd')] = svr_pro_fd[0]
            df.iloc[-1, df.columns.get_loc('svr_pro_fd_floor')] = svr_pro_fd[1]
            df.iloc[-1, df.columns.get_loc('svr_pro_fd_ceiling')] = svr_pro_fd[2]
            df.iloc[-1, df.columns.get_loc('gb_pro_dk')] = gb_pro_dk[0]
            df.iloc[-1, df.columns.get_loc('gb_pro_dk_floor')] = gb_pro_dk[1]
            df.iloc[-1, df.columns.get_loc('gb_pro_dk_ceiling')] = gb_pro_dk[2]
            df.iloc[-1, df.columns.get_loc('gb_pro_fd')] = gb_pro_fd[0]
            df.iloc[-1, df.columns.get_loc('gb_pro_fd_floor')] = gb_pro_fd[1]
            df.iloc[-1, df.columns.get_loc('gb_pro_fd_ceiling')] = gb_pro_fd[2]
            df.iloc[-1, df.columns.get_loc('enet_pro_dk')] = elastic_pro_dk[0]
            df.iloc[-1, df.columns.get_loc('enet_pro_dk_floor')] = elastic_pro_dk[1]
            df.iloc[-1, df.columns.get_loc('enet_pro_dk_ceiling')] = elastic_pro_dk[2]
            df.iloc[-1, df.columns.get_loc('enet_pro_fd')] = elastic_pro_fd[0]
            df.iloc[-1, df.columns.get_loc('enet_pro_fd_floor')] = elastic_pro_fd[1]
            df.iloc[-1, df.columns.get_loc('enet_pro_fd_ceiling')] = elastic_pro_fd[2]

            #Get average of projections
            total_proj_fd = rf_pro_fd[0] + ridge_pro_fd[0] + lasso_pro_fd[0] 
            total_proj_fd += nn_pro_fd[0] + knn_pro_fd[0] + svr_pro_fd[0] + gb_pro_fd[0] + elastic_pro_fd[0]
            total_proj_dk = rf_pro_dk[0] + ridge_pro_dk[0] + lasso_pro_dk[0] 
            total_proj_dk += nn_pro_dk[0] + knn_pro_dk[0] + svr_pro_dk[0] + gb_pro_dk[0] + elastic_pro_dk[0]

            total_floor_fd = rf_pro_fd[1] + ridge_pro_fd[1] + lasso_pro_fd[1] 
            total_floor_fd += nn_pro_fd[1] + knn_pro_fd[1] + svr_pro_fd[1] + gb_pro_fd[1] + elastic_pro_fd[1]
            total_floor_dk = rf_pro_dk[1] + ridge_pro_dk[1] + lasso_pro_dk[1] 
            total_floor_dk += nn_pro_dk[1] + knn_pro_dk[1] + svr_pro_dk[1] + gb_pro_dk[1] + elastic_pro_dk[1]

            total_ceil_fd = rf_pro_fd[2] + ridge_pro_fd[2] + lasso_pro_fd[2] 
            total_ceil_fd += nn_pro_fd[2] + knn_pro_fd[2] + svr_pro_fd[2] + gb_pro_fd[2] + elastic_pro_fd[2]
            total_ceil_dk = rf_pro_dk[2] + ridge_pro_dk[2] + lasso_pro_dk[2] 
            total_ceil_dk += nn_pro_dk[2] + knn_pro_dk[2] + svr_pro_dk[2] + gb_pro_dk[2] + elastic_pro_dk[2]
            
            avg_pro_fd = total_proj_fd / 8
            avg_pro_dk = total_proj_dk / 8
            
            avg_floor_fd = total_floor_fd / 8
            avg_floor_dk = total_floor_dk / 8

            avg_ceil_fd = total_ceil_fd / 8
            avg_ceil_dk = total_ceil_dk / 8
            #Print information about prejection
            print("Avg FD Proj: " + str(round(avg_pro_fd, 2)) + "\tAvg DK Proj: " + str(round(avg_pro_dk, 2)))
            print("Pro FD Proj: " + str(df.iloc[-1, df.columns.get_loc('projection_pro_fd')]) + "\tPro DK Proj: " +  str(df.iloc[-1, df.columns.get_loc('projection_pro_dk')]))
            print("Avg FD Floor: " + str(round(avg_floor_fd, 2)) + "\tAvg DK Floor: " + str(round(avg_floor_dk, 2)))
            print("Avg FD Ceiling: " + str(round(avg_ceil_fd, 2)) + "\tAvg DK Ceiling: " + str(round(avg_ceil_dk, 2)))
            #df.to_csv(filename, index=False)
        except ValueError as error:
            #print(error)
            print("Skipping " + str(name) + " (Player is on a bye week or does not have enough data)")
            pass
        current_count += 1
        stop = datetime.now()
