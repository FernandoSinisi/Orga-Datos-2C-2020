import pandas as pd
from sklearn import model_selection

from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, accuracy_score, make_scorer

from sklearn.model_selection import train_test_split, GridSearchCV

from keras.models import Sequential, Input
from keras.layers import  Dropout, Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras import layers

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPRegressor, MLPClassifier


def perceptron(x_train, y_train, x_validation, y_validation):
    perceptron = Perceptron(tol=1e-3, random_state=0)
    params_perc = {'alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]}
    per_gs = model_selection.GridSearchCV(perceptron, params_perc, cv=5)
    per_gs.fit(x_train, y_train)
    per_best = per_gs.best_estimator_
    print(per_gs.best_params_)
    print('perceptron: {}'.format(per_best.score(x_validation, y_validation)))
    return per_gs

def multi_perceptron(x_train, y_train, x_validation, y_validation):
    mult_perceptron = MLPClassifier(tol=1e-3, random_state=0)
    params_mult_perc =  {'hidden_layer_sizes': [(10,30,10),(20,)],'activation': ['tanh', 'relu'],'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],'learning_rate': ['constant','adaptive'],
    }
    mul_per_gs = model_selection.GridSearchCV(mult_perceptron, params_mult_perc, cv=5)
    mul_per_gs.fit(x_train, y_train)
    mul_per_best = mul_per_gs.best_estimator_
    print(mul_per_gs.best_params_)
    print('multi perceptron: {}'.format(mul_per_best.score(x_validation, y_validation)))
    return mul_per_best

def multi_perceptron_2(x_train, y_train, x_validation, y_validation):
    mult_perceptron_2 = MLPRegressor()
    params_mult_perc_2 = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (20,)],'activation': ['relu','tanh','logistic'],
          'alpha': [0.0001, 0.05],'learning_rate': ['constant','adaptive'],
          'solver': ['sgd', 'adam']}
    mul_per_gs_2 = model_selection.GridSearchCV(mult_perceptron_2, params_mult_perc_2, cv=5)
    mul_per_gs_2.fit(x_train, y_train)
    mul_per_best_2 = mul_per_gs_2.best_estimator_
    print(mul_per_gs_2.best_params_)
    print('multi perceptron 2: {}'.format(mul_per_best_2.score(x_validation, y_validation)))
    return mul_per_gs_2