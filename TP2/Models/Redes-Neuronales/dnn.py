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

def cross_val(model, x_train, y_train):
    score_cross_val = model_selection.cross_val_score(model, x_train, y_train, cv=5)
    print(score_cross_val.mean())

# Red neuronal profunda - https://keras.io/guides/sequential_model/
# nFeatures -> cantidad de features (columnas de set)
# nClasses -> cantidad de clases que pueden ser (2 -> lost vs won)
def DNN_model_bin(optimizer='rmsprop',init='glorot_uniform'):
    node = 512
    nClasses = 2
    dropout=0.5
    nFeatures = 185
    
    model = Sequential()
    model.add(Dense(node,input_dim=nFeatures,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,4):
        model.add(Dense(node,input_dim=node,activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
# Red neuronal profunda - https://keras.io/guides/sequential_model/
# nFeatures -> cantidad de features (columnas de set)
# nClasses -> cantidad de clases que pueden ser (2 -> lost vs won)
def DNN_model(optimizer='rmsprop',init='glorot_uniform'):
    node = 512
    nClasses = 2
    dropout=0.5
    nFeatures = 55
    
    model = Sequential()
    model.add(Dense(node,input_dim=nFeatures,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,4):
        model.add(Dense(node,input_dim=node,activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def dnn(x_train, y_train, x_validation, y_validation):
    keras_model = KerasClassifier(build_fn=DNN_model)
    optimizers = ['rmsprop', 'adam']
    init = ['glorot_uniform', 'normal', 'uniform']
    epochs = [50, 100, 150]
    batches = batches = [5, 10, 20]
    param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
    dnn_gs = GridSearchCV(keras_model, param_grid=param_grid)
    dnn_gs.fit(x_train, y_train)
    dnn_best = dnn_gs.best_estimator_
    print(dnn_gs.best_params_)
    print('dnn: {}'.format(dnn_best.score(x_validation, y_validation)))
    return dnn_best

def dnn_bin(x_train, y_train, x_validation, y_validation):
    keras_model = KerasClassifier(build_fn=DNN_model_bin)
    optimizers = ['rmsprop', 'adam']
    init = ['glorot_uniform', 'normal', 'uniform']
    epochs = [50, 100, 150]
    batches = batches = [5, 10, 20]
    param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
    dnn_gs = GridSearchCV(keras_model, param_grid=param_grid)
    dnn_gs.fit(x_train, y_train)
    dnn_best = dnn_gs.best_estimator_
    print(dnn_gs.best_params_)
    print('dnn: {}'.format(dnn_best.score(x_validation, y_validation)))
    return dnn_best

def test_model(model, x_test, y_test):
    predictions = model.predict_proba(x_test)[:,1]
    logloss = log_loss(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions.round())
    print("Accuracy: %.2f%%, Logloss: %.2f" % (accuracy*100.0, logloss))

