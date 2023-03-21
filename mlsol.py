import numpy as np
from sklearn.neighbors import NearestNeighbors

from mlsmote import create_dataset, get_minority_instace

import numpy as np
from sklearn.neighbors import NearestNeighbors

import numpy as np
from sklearn.neighbors import NearestNeighbors

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class MLSOL:
    def __init__(self, k=5, metric="euclidean"):
        self.sampling_strategy_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.k = k
        self.metric = metric

    def fit_resample(self, X, y):
        X, y = check_array(X), np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.sampling_strategy_ = {self.classes_[i]: np.sum(y == self.classes_[i]) for i in range(self.n_classes_)}

        max_imb = max(self.sampling_strategy_.values()) / np.mean(list(self.sampling_strategy_.values()))
        nn_indices = self._find_neighbors(X, y, max_imb)
        synth_X, synth_y = self._generate_samples(X, y.astype(int), nn_indices)

        X_resampled = np.concatenate([X, synth_X], axis=0)
        y_resampled = np.concatenate([y, synth_y], axis=0)

        return X_resampled, y_resampled

    def _find_neighbors(self, X, y, max_imb):
        self.knn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        self.knn.fit(X)

        distances, indices = self.knn.kneighbors(X)
        neighbor_classes = y[indices[:, 1:]]
        neighbor_dists = distances[:, 1:]

        def compute_prob_dist(distances):
            weights = 1.0 / (distances + 0.000000001)
            return weights / np.sum(weights)

        prob_dists = np.apply_along_axis(compute_prob_dist, axis=1, arr=neighbor_dists)
        synth_samples = []

        for i, prob_dist in enumerate(prob_dists):
            if np.max(prob_dist) > 0 and (self.sampling_strategy_[y[i]] / len(X)) < max_imb:
                synth_sample = np.zeros(X.shape[1])

                for j in range(X.shape[1]):
                    prob_dist_norm = prob_dist / np.sum(prob_dist)

                    synth_sample[j] = np.random.choice(X[:len(prob_dist_norm), j], p=prob_dist_norm)
                synth_samples.append(synth_sample)
        return self.knn.kneighbors(np.asarray(synth_samples))[1]

    def _generate_samples(self, X, y, nn_indices):
        synth_X = np.zeros((len(nn_indices), X.shape[1]))
        synth_y = np.zeros(len(nn_indices), dtype=np.int)

        for i, nn in enumerate(nn_indices):
            nn_classes = y[nn]
            nn_classes = nn_classes.astype(int)  # convert to integers

            nn_count = np.bincount(nn_classes, minlength=len(self.classes_))
            nn_count[y[i]] -= 1

            if nn_count[y[i]] == 0:
                nn_count[np.random.choice(self.classes_)] += 1

            max_class = np.argmax(nn_count)
            max_class_count = nn_count[max_class]
            noise = np.random.normal(scale=0.1, size=(X.shape[1],))

            synth_sample = X[i, :] + noise

            for j in range(X.shape[1]):
                diff = X[nn[max_class_count:], j] - X[i, j]
                synth_sample[j] += np.random.choice(diff) if len(diff) > 0 else 0

            synth_X[i, :] = synth_sample
            synth_y[i] = max_class

            self.sampling_strategy_[max_class] += 1

        return synth_X, synth_y


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Generate a synthetic imbalanced dataset
    X, y = make_classification(n_samples=1000, n_features=10, weights=[0.9, 0.1])

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    n_samples, n_features = X_train.shape
    X_reshaped = X_train.reshape(n_samples, n_features)
    # Create an instance of the MLSOL class with desired parameters
    mlsol = MLSOL(k=5)
    # Resample the training data using MLSOL
    X_resampled, y_resampled = mlsol.fit_resample(X_train, y_train)
