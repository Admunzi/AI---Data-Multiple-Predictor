"""
    Author: Daniel Ayala Cantador
"""

# Importing libraries
import pandas as pd
import numpy as np

# Importing models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Optimizing models
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Importing datasets
from sklearn.datasets import load_iris

# Importing metrics
from sklearn.metrics import accuracy_score, mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

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
    'KNN': {
        'model': KNeighborsClassifier(),
        'type': 'classification',
        'cross_name': 'GridSearchCV',
        'params_cross': {
            'n_neighbors': np.arange(1, 25),
            'weights': ['uniform', 'distance'],
        },
        'cv_cross': 5
    },
    'Naive Bayes GaussianNB': {
        'model': GaussianNB(),
        'type': 'classification',
    },
    'Naive Bayes MultinomialNB': {
        'model': MultinomialNB(),
        'type': 'classification',
    },
    'Naive Bayes BernoulliNB': {
        'model': BernoulliNB(),
        'type': 'classification',
    },
    'Naive Bayes ComplementNB': {
        'model': ComplementNB(),
        'type': 'classification',
    },
    'DecisionTree Classifier': {
        'model': DecisionTreeClassifier(),
        'type': 'classification'
    },
    'DecisionTree Regressor': {
        'model': DecisionTreeRegressor(),
        'type': 'regression'
    },
    'DecisionTreeClassifier AdaBoostClassifier GridSearchCV': {
        'model': AdaBoostClassifier(DecisionTreeClassifier()),
        'type': 'classification',
        'cross_name': 'GridSearchCV',
        'params_cross': {
            'n_estimators': [8, 32, 64, 128, 256],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.25, 1]
        },
        'cv_cross': 5,
    },
    'DecisionTreeClassifier AdaBoostClassifier RandomizedSearchCV': {
        'model': AdaBoostClassifier(DecisionTreeClassifier()),
        'type': 'classification',
        'cross_name': 'RandomizedSearchCV',
        'params_cross': {
            'n_estimators': [8, 32, 64, 128, 256],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.25, 1]
        },
        'cv_cross': 5,
    }
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
        for model_name, model_dict in MODELS.items():
            model = model_dict['model']
            if 'cross_name' in model_dict:
                model = model_cross(model, model_dict)

            model.fit(splits['X_train'], splits['y_train'])
            y_pred = model.predict(splits['X_test'])

            create_metrics(dataset_name, model_dict, model_name, predictions, splits, y_pred)

    return predictions


def create_metrics(dataset_name, model_dict, model_name, predictions, splits, y_pred):
    # We want show too roc_auc_score
    if model_dict['type'] == 'classification':
        predictions[dataset_name][model_name] = [accuracy_score(splits['y_test'], y_pred), '']
    else:
        predictions[dataset_name][model_name] = ['', mean_absolute_error(splits['y_test'], y_pred)]


def model_cross(model, model_dict):
    cross_name = model_dict['cross_name']
    params_cross = model_dict['params_cross']
    cv_cross = model_dict['cv_cross']
    cross = None

    if cross_name == 'GridSearchCV':
        cross = GridSearchCV(model, params_cross, cv=cv_cross)
    elif cross_name == 'RandomizedSearchCV':
        cross = RandomizedSearchCV(model, params_cross, cv=cv_cross, n_iter=10)
    return cross


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
