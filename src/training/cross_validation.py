import numpy as np
from sklearn.model_selection import StratifiedGroupKFold as SGKF


class StratifiedGroupKFold:
    """
    Stratified Group K-Fold cross-validator.

    Ensures:
    1. No patient appears in both train and test sets (prevents data leakage)
    2. Proportion of classes is preserved in each fold

    This is CRITICAL for medical data with multiple recordings per patient.

    Example:
        >>> cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        >>> for train_idx, test_idx in cv.split(X, y, groups=patient_ids):
        >>>     # All samples from same patient are in either train OR test, never both
        >>>     X_train, X_test = X[train_idx], X[test_idx]
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    # def split(self, X, y, groups):
    #     """
    #     Generate train/test indices for each fold.

    #     Args:
    #         X (np.array or pd.DataFrame): Feature matrix (not used directly).
    #         y (np.array): Labels.
    #         groups (np.array): Group/patient IDs.

    #     Yields:
    #         tuple: (train_idx, test_idx) as numpy arrays.
    #     """
    #     unique_groups = np.unique(groups)
    #     group_labels = np.array([y[groups == g][0] for g in unique_groups])

    #     cv = SGKF(
    #         n_splits=self.n_splits,
    #         shuffle=self.shuffle,
    #         random_state=self.random_state
    #     )

    #     for train_groups, test_groups in cv.split(unique_groups, group_labels, groups=unique_groups):
    #         train_group_set = set(unique_groups[train_groups])
    #         test_group_set = set(unique_groups[test_groups])

    #         assert train_group_set.isdisjoint(test_group_set), \
    #             "Data leakage detected! Some patients appear in both train and test!"

    #         train_idx = np.where(np.isin(groups, train_group_set))[0]
    #         test_idx = np.where(np.isin(groups, test_group_set))[0]

    #         yield train_idx, test_idx

    def split(self, X, y, groups):
        cv = SGKF(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        for train_idx, test_idx in cv.split(X, y, groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])

            assert train_groups.isdisjoint(test_groups), \
                "Data leakage detected!"

            yield train_idx, test_idx

    def get_n_splits(self):
        """Get number of splits."""
        return self.n_splits


def cross_validate(
    model,
    X,
    y,
    patient_ids,
    n_splits=5,
    scaler=None,
    scoring='accuracy',
    inner_cv=3,
    inner_scorer='accuracy'
):
    """
    Perform nested cross-validation with patient-level separation.

    Outer loop: Model evaluation
    Inner loop: Hyperparameter tuning (GridSearchCV)

    Args:
        model: sklearn-compatible model.
        X (np.array): Feature matrix.
        y (np.array): Labels.
        patient_ids (np.array): Patient IDs for grouping.
        n_splits (int): Number of outer folds.
        scaler: Scaler (e.g., StandardScaler).
        scoring (str): Outer loop scoring metric.
        inner_cv (int): Number of inner CV folds.
        inner_scorer (str): Inner CV scoring metric.

    Returns:
        dict: Results including scores per fold and best parameters.
    """
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.preprocessing import StandardScaler

    cv = StratifiedGroupKFold(n_splits=n_splits)

    results = {
        'test_scores': [],
        'train_scores': [],
        'best_params': [],
        'models': []
    }

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, patient_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_patients = patient_ids[train_idx]
        test_patients = patient_ids[test_idx]

        assert set(train_patients).isdisjoint(set(test_patients)), \
            f"LEAKAGE in fold {fold}!"

        if scaler is not None:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        train_score = cross_val_score(
            model, X_train, y_train,
            cv=inner_cv, scoring=inner_scorer
        )
        results['train_scores'].append(train_score.mean())

        grid = GridSearchCV(
            model, {}, cv=inner_cv, scoring=inner_scorer
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        results['best_params'].append(grid.best_params_)

        test_score = cross_val_score(
            best_model, X_test, y_test,
            cv=3, scoring=scoring
        )
        results['test_scores'].append(test_score.mean())
        results['models'].append(best_model)

    return results