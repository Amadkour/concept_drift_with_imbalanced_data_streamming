from collections import Counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class BaggingClassifier:
    def __init__(self, n_estimators=10, max_depth=5, sample_size=1.0, model=DecisionTreeClassifier(max_depth=5),
                 name='tree'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.model = model
        self.name = name
        self.estimators = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Randomly sample data with replacement
            sample_indices = np.random.choice(len(X), size=int(self.sample_size * len(X)), replace=True)
            X_sample = X[ np.asarray(sample_indices,int)]
            y_sample = y[sample_indices]

            # Train a decision tree on the sampled data
            tree = self.model
            tree.fit(X_sample, y_sample)

            # Add the tree to the list of estimators
            self.estimators.append(tree)

    def predict(self, X):
        predictions = np.zeros((len(X), self.n_estimators))
        for i, estimator in enumerate(self.estimators):
            try:
                predictions[:, i] = estimator.predict(X)
            except:
                pred = estimator.predict(X)
                for i in range(len(predictions)):
                    np.append(predictions[i], pred[i])
        return np.array([Counter(row).most_common(1)[0][0] for row in predictions])
