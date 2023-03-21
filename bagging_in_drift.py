import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from skmultiflow.drift_detection import DDM, ADWIN
from xgboost import XGBClassifier

from bagging_class import BaggingClassifier
from mlsmote import MLSMOTE
from mlsol import MLSOL


def new_model(bagging):
    if bagging.name == 'random forest':
        return BaggingClassifier(n_estimators=5, model=RandomForestClassifier(n_estimators=10), name='random forest')
    elif bagging.name == 'svm':
        return BaggingClassifier(n_estimators=5, model=svm.SVC(decision_function_shape='ovo'), name='svm')
    elif bagging.name == 'XGB':
        return BaggingClassifier(n_estimators=5, model=XGBClassifier(), name='XGB')
    elif bagging.name == 'KNN':
        return BaggingClassifier(n_estimators=5, model=KNeighborsClassifier(n_neighbors=5), name='KNN')
    elif bagging.name == 'tree':
        return BaggingClassifier(n_estimators=5, name='tree')
    else:
        return BaggingClassifier(n_estimators=5, name='tree')


def learning_from_new_chunk(data, label, x_test, y_test):
    # Train the Bagging classifiers on the dataset
    worst_accuracy = 100.0
    worst_bagging = None
    for bagging in bagging_pool:
        bagging.fit(data, label)
        y_pred = bagging.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('accuracy of ', bagging.name, 'is: ', accuracy)
        if accuracy < worst_accuracy:
            worst_accuracy = accuracy
            worst_bagging = bagging
    print("Worst Bagging classifier accuracy:", worst_accuracy)
    bagging_pool.remove(worst_bagging)
    model = new_model(worst_bagging)
    bagging_pool.append(model)
    bagging_pool[-1].fit(data, label)
    for method in driftMethods:
        prediction = bagging_pool[-1].predict(x_test)
        for instance in prediction:
            method.add_element(instance)


bagging_pool = [
    BaggingClassifier(n_estimators=5, model=RandomForestClassifier(n_estimators=10), name='random forest'),
    BaggingClassifier(n_estimators=6, model=svm.SVC(decision_function_shape='ovo'), name='svm'),
    BaggingClassifier(n_estimators=5, model=svm.SVC(), name='svm'),
    BaggingClassifier(n_estimators=5, model=XGBClassifier(), name='XGB'),
    # BaggingClassifier(n_estimators=5, model=CategoricalNB()),
    BaggingClassifier(n_estimators=7, model=KNeighborsClassifier(n_neighbors=5), name='KNN'),
    BaggingClassifier(n_estimators=7, name='tree'),
]


# Generate a synthetic balanced classification dataset
def generate_balanced_data_streaming():
    return make_classification(n_samples=5000, n_features=10, n_informative=5, random_state=42, n_classes=5,
                               weights=[0.2, 0.2, 0.2, 0.2, 0.2])


# Generate a synthetic imbalanced classification dataset
def generate_imbalanced_data_streaming(i):
    return make_classification(n_samples=1000, n_features=10, n_informative=5,
                               n_redundant=0, n_classes=5,
                               weights=[0.9, 0.02, 0.04, 0.02, 0.02],
                               class_sep=0.8, random_state=42)


driftMethods = [
    DDM(),
    ADWIN(delta=0.2),
]


def reset_model():
    for method in driftMethods:
        method.reset()
    bagging_pool = [
        BaggingClassifier(n_estimators=5, model=RandomForestClassifier(n_estimators=10), name='random forest'),
        BaggingClassifier(n_estimators=6, model=svm.SVC(decision_function_shape='ovo'), name='svm'),
        BaggingClassifier(n_estimators=5, model=svm.SVC(), name='svm'),
        BaggingClassifier(n_estimators=5, model=XGBClassifier(), name='XGB'),
        # BaggingClassifier(n_estimators=5, model=CategoricalNB()),
        BaggingClassifier(n_estimators=7, model=KNeighborsClassifier(n_neighbors=5), name='KNN'),
        BaggingClassifier(n_estimators=7, ),
    ]


bestBagging = bagging_pool[-1]


def drift_by_mlsol(data, label):
    for method in driftMethods:
        have_drift = 0
        print("===========================[ by using", method.get_info(), ']=========================')
        prediction = bestBagging.predict(data)
        for i in range(len(prediction)):
            method.add_element(i)
            if method.detected_change():
                have_drift += 1
                print('Change detected at index', i)

        if have_drift > 0:
            data, label = MLSOL().fit_resample(data, label)
            x, x_test, y, y_test = train_test_split(data, label, test_size=0.2)
            learning_from_new_chunk(x, y, x_test, y_test)
            method.reset()
            for l in label:
                method.add_element(l)


def drift_by_mlsmote(data, label):
    for method in driftMethods:
        have_drift = 0
        print("===========================[ by using", method.get_info(), ']=========================')
        prediction = bestBagging.predict(data)
        for i in range(len(prediction)):
            method.add_element(i)
            if method.detected_change():
                have_drift += 1
        if have_drift > 0:
            print('Change detected')

            data = pd.DataFrame(data)
            label = pd.get_dummies(label, prefix='class')
            data, label = MLSMOTE(data, label, 50)
            data=data.to_numpy()
            label=np.array(label)
            x, x_test, y, y_test = train_test_split(data, label, test_size=0.2)
            learning_from_new_chunk(x, y, x_test, y_test)
            method.reset()
            for l in label:
                method.add_element(l)


# first chunk
x, y = generate_balanced_data_streaming()
x, x_test, y, y_test = train_test_split(x, y, test_size=0.2)

learning_from_new_chunk(x, y, x_test, y_test)

# second chunk
for chunk_index in range(10):
    x, y = generate_imbalanced_data_streaming(chunk_index)
    drift_by_mlsmote(x, y)
