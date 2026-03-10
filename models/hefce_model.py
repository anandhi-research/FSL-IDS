import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

class HEFCE:

    def __init__(self):

        self.model1 = GradientBoostingClassifier(n_estimators=200)
        self.model2 = MLPClassifier(hidden_layer_sizes=(64,32))
        self.meta = LogisticRegression()

    def fit(self, X, y):

        self.model1.fit(X, y)
        self.model2.fit(X, y)

        p1 = self.model1.predict_proba(X)
        p2 = self.model2.predict_proba(X)

        stacked = np.hstack((p1, p2))

        self.meta.fit(stacked, y)

    def predict(self, X):

        p1 = self.model1.predict_proba(X)
        p2 = self.model2.predict_proba(X)

        stacked = np.hstack((p1, p2))

        return self.meta.predict(stacked)
