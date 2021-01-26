import numpy as np

"""Bernoulli Naive Bayes Class"""


class Bernoulli_NB(object):
    def __init__(self, alpha=1.0, binarize=0.7):
        self.alpha = alpha
        self.binarize = binarize

    def fit(self, X, y):
        X = self._binarize_X(X)
        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.class_log_prior_ = [
            np.log(len(i) / count_sample) for i in separated]
        count = np.array([np.array(i).sum(axis=0)
                          for i in separated]) + self.alpha
        smoothing = 2 * self.alpha
        n_doc = np.array([len(i) + smoothing for i in separated])
        self.feature_prob_ = count / n_doc[np.newaxis].T
        self.feature_prob_ = self.feature_prob_.astype('float64')
        np.savetxt("DiceFeatures.csv", self.feature_prob_.T, delimiter=",")
        return self

    def predict_log_proba(self, X):
        X = self._binarize_X(X)
        return [(np.log(self.feature_prob_) * x + np.log(1 - self.feature_prob_) * np.abs(x - 1)).sum(axis=1) + self.class_log_prior_ for x in X]

    def predict(self, X):
        X = self._binarize_X(X)
        return np.argmax(self.predict_log_proba(X), axis=1)

    def _binarize_X(self, X):
        return np.where(X >= self.binarize, 1, 0) if self.binarize != None else X


"""Multinomial Naive Bayes Class"""


class Multi_NB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):

        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.class_log_prior_ = [
            np.log(len(i) / count_sample) for i in separated]
        count = np.array([np.array(i).sum(axis=0)
                          for i in separated]) + self.alpha
        self.feature_log_prob = np.log(count / count.sum(axis=1)[np.newaxis].T)
        self.feature_log_prob = self.feature_log_prob.astype('float64')
        return self

    def predict_log_proba(self, X):
        return [(self.feature_log_prob * x).sum(axis=1) + self.class_log_prior_ for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)
