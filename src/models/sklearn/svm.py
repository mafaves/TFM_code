from sklearn.svm import SVC
import numpy as np


PARAM_GRID = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'degree': [2, 3, 4, 5],
    'class_weight': ['balanced', None]
}


class SVMClassifier:
    """
    Support Vector Machine classifier for voice-based Parkinson's detection.

    Wrapper around sklearn's SVC with cross-validation support.

    Example:
        >>> from src.models.sklearn import SVMClassifier
        >>> model = SVMClassifier(kernel='rbf', C=1.0)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        C=1.0,
        kernel='rbf',
        gamma='scale',
        degree=3,
        class_weight='balanced',
        probability=False,
        random_state=42,
        cache_size=1000
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.class_weight = class_weight
        self.probability = probability
        self.random_state = random_state
        self.cache_size = cache_size

        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state,
            cache_size=cache_size
        )

        self._fitted = False

    def fit(self, X, y):
        """Fit the model to training data."""
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

    def get_params(self):
        """Get model parameters."""
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'class_weight': self.class_weight,
            'probability': self.probability,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def get_model(self):
        """Get the underlying sklearn model."""
        return self.model