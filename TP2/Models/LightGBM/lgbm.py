import pandas as pd
import scipy.stats as stats
from sklearn import model_selection, metrics
from lightgbm import LGBMClassifier

def cross_val(model, x_train, y_train):
    score_cross_val = model_selection.cross_val_score(model, x_train, y_train, cv=5)
    print(score_cross_val.mean())

params = {
    'num_leaves':stats.randint(90,5000),
    'min_data_in_leaf':stats.randint(100,1000),
    'max_bin':stats.randint(70,300),
    'learning_rate':stats.uniform(0.01,1-0.01),
    'n_estimators':stats.randint(150,500)
}

def lgbm(x_train, y_train, x_validation, y_validation):
    lgbm_classifier = LGBMClassifier()
    lgbm_gs = model_selection.RandomizedSearchCV(lgbm_classifier, params, cv=2,verbose=1,n_iter=150)
    lgbm_gs.fit(x_train, y_train)
    lgbm_best = lgbm_gs.best_estimator_
    print(lgbm_gs.best_params_)
    print('lgbm: {}'.format(lgbm_best.score(x_validation, y_validation)))
    return lgbm_best

def test_model(model, x_test, y_test):
    predictions = model.predict_proba(x_test)[:,1]
    logloss = metrics.log_loss(y_test, predictions)
    accuracy = metrics.accuracy_score(y_test, predictions.round())
    print("Accuracy: %.2f%%, Logloss: %.2f" % (accuracy*100.0, logloss))

def best_features(model,train):
    importance = model.feature_importances_
    result = pd.DataFrame([train.columns,importance]).transpose()
    result.columns = ["Feature","Importance"]
    return result.sort_values(by='Importance', ascending=False).head(50)["Feature"]

