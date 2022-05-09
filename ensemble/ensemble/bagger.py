import numpy as np

from sklearn.tree import DecisionTreeClassifier
from TCH import TCH
from .sampler import FeatureSampler, ObjectSampler


class Bagger:
    def __init__(self, base_estimator, object_sampler, feature_sampler, n_estimators=10, **params):
        """
        n_estimators : int
            number of base estimators
        base_estimator : class
            class for base_estimator with fit(), predict() and predict_proba() methods
        feature_sampler : instance of FeatureSampler
        object_sampler : instance of ObjectSampler
        n_estimators : int
            number of base_estimators
        params : kwargs
            params for base_estimator initialization
        """
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.feature_sampler = feature_sampler
        self.object_sampler = object_sampler
        self.estimators = []
        self.indices = []
        self.params = params

        self.n_classes = None

    def fit(self, X, y, verbose, optim_method):
        """
        for i in range(self.n_estimators):
            1) select random objects and answers for train
            2) select random indices of features for current estimator
            3) fit base_estimator (don't forget to remain only selected features)
            4) save base_estimator (self.estimators) and feature indices (self.indices)

        NOTE that self.base_estimator is class and you should init it with
        self.base_estimator(**self.params) before fitting
        """
        self.n_classes = len(np.unique(y))
        self.estimators = []
        self.indices = []

        for est_num in range(self.n_estimators):
            print(f'\n\nFITTING ESTIMATOR NUMBER {est_num + 1} of {self.n_estimators}\n\n')
            ind = self.object_sampler.sample_indices(X.shape[0])

            x, ynew = X[ind], y[ind]
            ind_features = self.feature_sampler.sample_indices(X.shape[1])
            x = x[:, ind_features]
            base_est = self.base_estimator(**self.params)

            self.estimators.append(base_est.fit(x, ynew, verbose=verbose, optim_method=optim_method))
            self.indices.append(ind_features)

        return self

    def predict_proba(self, X):
        """
        Returns
        -------
        probas : numpy ndarrays of shape (n_objects, n_classes)

        Calculate mean value of all probas from base_estimators
        Don't forget, that each estimator has its own feature indices for prediction
        """
        if not (0 < len(self.estimators) == len(self.indices)):
            raise RuntimeError('Bagger is not fitted', (len(self.estimators), len(self.indices)))

        probas = []
        for ind_feature, estimator in zip(self.indices, self.estimators):
            y_proba = estimator.predict_proba(X[:, ind_feature])
            probas.append(y_proba)
        return np.mean(probas, axis=0)

        """
        if self.n_classes:
            probas = np.zeros((X.shape[0], self.n_classes))
        else:
            y_pred_test = self.estimators[0].predict_proba(X[:, self.indices[0]])
            probas = np.zeros_like(y_pred_test)

        for ind_feature, estimator in zip(self.indices, self.estimators):
            y_proba = estimator.predict_proba(X[:, ind_feature])
            probas += y_proba

        return probas / self.n_estimators
        """

    def predict(self, X):
        """
        Returns
        -------
        predictions : numpy ndarrays of shape (n_objects, )
        """
        return np.argmax(self.predict_proba(X), axis=1)


class RandomForestClassifier(Bagger):
    def __init__(self, n_estimators=30, max_objects_samples=0.9, max_features_samples=0.8,
                 N=200, random_state=None, **params): # max_depth=None, min_samples_leaf=1
        base_estimator = TCH
        object_sampler = ObjectSampler(max_samples=max_objects_samples, random_state=random_state)
        feature_sampler = FeatureSampler(max_samples=max_features_samples, random_state=random_state)

        super().__init__(
            base_estimator=base_estimator,
            object_sampler=object_sampler,
            feature_sampler=feature_sampler,
            n_estimators=n_estimators,
            # max_depth=max_depth,
            # min_samples_leaf=min_samples_leaf,
            N=N,
            **params
        )
