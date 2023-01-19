"""
    Author: Daniel Ayala Cantador
"""

# Importing libraries
import pandas as pd
import numpy as np

# Importing models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.tree import DecisionTreeRegressor

# Optimizing models
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Importing datasets
from sklearn.datasets import load_iris

# Importing metrics
from sklearn.metrics import accuracy_score, mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor

PREDICT_DATASETS = {
    'Iris': {
        'source': pd.DataFrame(data=np.c_[load_iris()['data'], load_iris()['target']],
                               columns=load_iris()['feature_names'] + ['target']),
        'objective': 'target',
        'size-test': .2
    },
    'Wine Quality': {
        'source': pd.read_csv('datasets/winequality-red.csv'),
        'objective': 'quality',
        'size-test': .2
    },
    'Titanic': {
        'source': pd.read_csv('datasets/titanic.csv'),
        'objective': 'Survived',
        'size-test': .3
    }
}

MODELS = {
    'KNN Weights Uniform': KNeighborsClassifier(n_neighbors=5, weights='uniform'),
    'KNN Weights Normal': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'Naive Bayes GaussianNB': GaussianNB(),
    'Naive Bayes MultinomialNB': MultinomialNB(),
    'Naive Bayes BernoulliNB': BernoulliNB(),
    'Naive Bayes ComplementNB': ComplementNB(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'DecisionTreeRegressor AdaBoostRegressor GridSearchCV': GridSearchCV(
        estimator=AdaBoostRegressor(DecisionTreeRegressor()),
        param_grid={
            'n_estimators': [8, 32, 64, 128, 256],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.25, 1]
        },
        cv=5),
    'DecisionTreeRegressor AdaBoostRegressor RandomizedSearchCV': RandomizedSearchCV(
        estimator=AdaBoostRegressor(DecisionTreeRegressor()),
        param_distributions={
            'n_estimators': [8, 32, 64, 128, 256],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.25, 1]
        },
        cv=5),
}


def process_datasets():
    split_datasets = {}

    for name_prediction, value_prediction in PREDICT_DATASETS.items():
        df = value_prediction['source']
        objetive = value_prediction['objective']

        df[objetive] = pd.Categorical(df[objetive])

        X = df.drop([objetive], axis=1)
        y = df[objetive]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=value_prediction['size-test'])

        split_datasets[name_prediction] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    return split_datasets


def fit_and_predict(split_datasets):
    predictions = {}

    for dataset_name, splits in split_datasets.items():
        predictions[dataset_name] = {}
        for model_name, model in MODELS.items():
            model.fit(splits['X_train'], splits['y_train'])
            y_pred = model.predict(splits['X_test'])

            predictions[dataset_name][model_name] = [
                accuracy_score(splits['y_test'], y_pred),
                mean_absolute_error(splits['y_test'], y_pred)
            ]

    return predictions


def show_results(predictions):
    for dataset_name, models in predictions.items():
        print(f'================== {dataset_name} ==================')
        print(pd.DataFrame.from_dict(models, orient='index', columns=['Accuracy Score', 'Mean Absolute Error']))
        print('=====================================================')


def main():
    split_datasets = process_datasets()
    predictions = fit_and_predict(split_datasets)
    show_results(predictions)


if "__main__" == __name__:
    main()
