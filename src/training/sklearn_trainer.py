import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import json
import os
from collections import defaultdict


class SklearnTrainer:
    """
    Enhanced trainer for sklearn ML models with cross-validation.

    Features:
    - Patient-level train/test splitting (prevents data leakage)
    - Hyperparameter tuning with GridSearchCV on patient-level inner CV
    - Detailed logging of patient counts in each split
    - Segment-wise AND patient-wise predictions and metrics
    - Comprehensive result saving by exercise/feature_type

    Example:
        >>> from src.models.sklearn import SVMClassifier
        >>> from src.training import SklearnTrainer
        >>> trainer = SklearnTrainer(model=SVMClassifier())
        >>> results = trainer.train(X, y, patient_ids, exercise='vocal', feature_type='opensmile')
    """

    def __init__(
        self,
        model,
        param_grid=None,
        n_splits=5,
        inner_cv=3,
        scoring='accuracy',
        scale_features=True,
        random_state=42
    ):
        self.model = model
        self.param_grid = param_grid or {}
        self.n_splits = n_splits
        self.inner_cv = inner_cv
        self.scoring = scoring
        self.scale_features = scale_features
        self.random_state = random_state

        self.scaler_ = StandardScaler()
        self.best_model_ = None
        self.results_ = None

    def _compute_specificity(self, cm):
        """Compute specificity from confusion matrix."""
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0
        return None

    def _get_patient_counts(self, patient_labels):
        """Get patient counts per class."""
        unique_labels = np.unique(patient_labels)
        counts = {}
        for label in unique_labels:
            counts[int(label)] = int(sum(patient_labels == label))
        return counts

    def _get_patient_label_map(self, patient_ids, labels):
        """Create mapping from patient ID to label (assumes consistent labels per patient)."""
        patient_label_map = {}
        for pid, label in zip(patient_ids, labels):
            if pid not in patient_label_map:
                patient_label_map[pid] = label
        return patient_label_map

    def _aggregate_patient_predictions(self, patient_ids, y_true, y_pred_scores, threshold=0):
        """Aggregate segment predictions to patient level."""
        if y_pred_scores.ndim == 1:
            df = pd.DataFrame({
                'patient_id': patient_ids,
                'true_label': y_true,
                'score': y_pred_scores
            })
            patient_scores = df.groupby('patient_id')['score'].mean()
            patient_true = df.groupby('patient_id')['true_label'].first()
            patient_pred = (patient_scores > threshold).astype(int)
        else:
            df = pd.DataFrame({
                'patient_id': patient_ids,
                'true_label': y_true,
                'score': y_pred_scores[:, 1] if y_pred_scores.shape[1] > 1 else y_pred_scores[:, 0]
            })
            patient_scores = df.groupby('patient_id')['score'].mean()
            patient_true = df.groupby('patient_id')['true_label'].first()
            patient_pred = (patient_scores > threshold).astype(int)

        return patient_true, patient_pred, patient_scores

    def train(
        self,
        X,
        y,
        patient_ids,
        exercise='default',
        feature_type='default',
        save_dir=None,
        verbose=True,
        test_data=None
    ):
        """
        Train model with nested cross-validation.

        Args:
            X (np.array): Feature matrix.
            y (np.array): Labels.
            patient_ids (np.array): Patient IDs for group-level splitting.
            exercise (str): Exercise name for file organization.
            feature_type (str): Feature type for file organization.
            save_dir (str, optional): Base directory to save results.
            verbose (bool): Print progress.
            test_data (tuple, optional): (X_external, y_external, patient_ids_external)
                External patients (e.g., AC group) to test on each fold.
                Useful for seeing how ambiguous patients are classified.

        Returns:
            dict: Training results with segment and patient-level metrics.
        """
        from .cross_validation import StratifiedGroupKFold

        cv = StratifiedGroupKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)

        patient_label_map = self._get_patient_label_map(patient_ids, y)
        unique_patients = np.array(list(patient_label_map.keys()))
        patient_labels = np.array([patient_label_map[p] for p in unique_patients])

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training with {self.n_splits}-fold outer CV")
            print(f"Total unique patients: {len(unique_patients)}")
            print(f"Patients per class: {self._get_patient_counts(patient_labels)}")
            print(f"X shape: {X.shape}, y shape: {y.shape}")
            print(f"{'='*60}\n")

        fold_results = []
        all_segment_metrics = defaultdict(list)
        all_patient_metrics = defaultdict(list)

        full_segment_preds = []
        full_patient_preds = []

        seen_patients = []

        for fold, (train_patient_idx, test_patient_idx) in enumerate(
            cv.split(unique_patients, patient_labels, groups=unique_patients), 1
        ):
            train_patients = unique_patients[train_patient_idx]
            test_patients = unique_patients[test_patient_idx]

            train_patient_labels = patient_labels[train_patient_idx]
            test_patient_labels = patient_labels[test_patient_idx]

            train_counts = self._get_patient_counts(train_patient_labels)
            test_counts = self._get_patient_counts(test_patient_labels)

            if verbose:
                print(f"\n=== Fold {fold}/{self.n_splits} ===")
                print(f"  Outer - Train patients: {train_counts}")
                print(f"  Outer - Test patients: {test_counts}")

            assert set(train_patients).isdisjoint(set(test_patients)), \
                "DATA LEAKAGE DETECTED!"
            seen_patients.append(set(test_patients))
            for past in seen_patients[:-1]:
                assert set(test_patients).isdisjoint(past), "Overlap between test sets!"

            train_indices = np.where(np.isin(patient_ids, train_patients))[0]
            test_indices = np.where(np.isin(patient_ids, test_patients))[0]

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            if self.scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if self.param_grid:
                unique_train_patients = np.array(list(set(patient_ids[train_indices])))
                train_patient_label_map = self._get_patient_label_map(
                    patient_ids[train_indices], y_train
                )
                train_patient_labels = np.array([
                    train_patient_label_map[p] for p in unique_train_patients
                ])

                inner_cv = StratifiedGroupKFold(
                    n_splits=self.inner_cv,
                    random_state=self.random_state + fold
                )

                inner_cv_splits = []
                inner_train_patients_set = set(unique_train_patients)
                inner_train_labels_set = np.array([
                    train_patient_label_map[p] for p in unique_train_patients
                ])

                for inner_train_idx, inner_val_idx in inner_cv.split(
                    unique_train_patients, inner_train_labels_set,
                    groups=unique_train_patients
                ):
                    inner_train_pids = unique_train_patients[inner_train_idx]
                    inner_val_pids = unique_train_patients[inner_val_idx]

                    inner_train_indices = np.where(
                        np.isin(patient_ids[train_indices], inner_train_pids)
                    )[0]
                    inner_val_indices = np.where(
                        np.isin(patient_ids[train_indices], inner_val_pids)
                    )[0]

                    inner_cv_splits.append((inner_train_indices, inner_val_indices))

                    inner_train_labels = [train_patient_label_map[p] for p in inner_train_pids]
                    inner_val_labels = [train_patient_label_map[p] for p in inner_val_pids]

                    if verbose:
                        inner_train_counts = {
                            0: inner_train_labels.count(0),
                            1: inner_train_labels.count(1)
                        }
                        inner_val_counts = {
                            0: inner_val_labels.count(0),
                            1: inner_val_labels.count(1)
                        }
                        print(f"  Inner CV - Train: {inner_train_counts}, Val: {inner_val_counts}")

                grid = GridSearchCV(
                    self.model.__class__(**self.model.get_params())
                    if hasattr(self.model, 'get_params') else self.model,
                    self.param_grid,
                    cv=inner_cv_splits,
                    scoring=self.scoring,
                    n_jobs=-1
                )
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                model = self.model.__class__(**self.model.get_params()) \
                    if hasattr(self.model, 'get_params') else self.model
                model.fit(X_train, y_train)
                best_params = {}

            if verbose:
                print(f"  Best params: {best_params}")

            # Segment-wise predictions
            y_pred = model.predict(X_test)

            # Determine threshold: 0 for decision_function, 0.5 for probabilities
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                    y_scores = y_prob[:, 1]
                else:
                    y_scores = y_prob.ravel()
                threshold = 0.5
            elif hasattr(model, 'decision_function'):
                y_scores = model.decision_function(X_test)
                threshold = 0
            else:
                y_scores = y_pred
                threshold = 0.5

            seg_cm = confusion_matrix(y_test, y_pred)
            seg_specificity = self._compute_specificity(seg_cm)

            try:
                seg_auc = roc_auc_score(y_test, y_scores)
            except:
                seg_auc = None

            segment_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'specificity': seg_specificity,
                'auc': seg_auc,
                'confusion_matrix': seg_cm.tolist()
            }

            for k, v in segment_metrics.items():
                if k != 'confusion_matrix':
                    all_segment_metrics[k].append(v)

            # Patient-wise predictions
            patient_true, patient_pred, patient_scores = self._aggregate_patient_predictions(
                patient_ids[test_indices], y_test, y_scores,
                threshold=threshold
            )

            pat_cm = confusion_matrix(patient_true, patient_pred)
            pat_specificity = self._compute_specificity(pat_cm)

            try:
                pat_auc = roc_auc_score(patient_true, patient_scores)
            except:
                pat_auc = None

            patient_metrics = {
                'accuracy': accuracy_score(patient_true, patient_pred),
                'recall': recall_score(patient_true, patient_pred, zero_division=0),
                'f1': f1_score(patient_true, patient_pred, zero_division=0),
                'specificity': pat_specificity,
                'auc': pat_auc,
                'confusion_matrix': pat_cm.tolist()
            }

            for k, v in patient_metrics.items():
                if k != 'confusion_matrix':
                    all_patient_metrics[k].append(v)

            if verbose:
                print(f"  Segment-level - Acc: {segment_metrics['accuracy']:.3f}, "
                      f"AUC: {segment_metrics['auc']:.3f}")
                print(f"  Patient-level - Acc: {patient_metrics['accuracy']:.3f}, "
                      f"AUC: {patient_metrics['auc']:.3f}")

            # ==========================================
            # Test external patients (e.g., AC group)
            # ==========================================
            test_external_results = None
            if test_data is not None:
                X_test_ext, y_test_ext, patient_ids_ext = test_data

                if self.scale_features:
                    X_test_ext_scaled = scaler.transform(X_test_ext)
                else:
                    X_test_ext_scaled = X_test_ext

                y_pred_ext = model.predict(X_test_ext_scaled)

                if hasattr(model, 'predict_proba'):
                    y_prob_ext = model.predict_proba(X_test_ext_scaled)
                    if y_prob_ext.ndim > 1 and y_prob_ext.shape[1] > 1:
                        y_scores_ext = y_prob_ext[:, 1]
                    else:
                        y_scores_ext = y_prob_ext.ravel()
                elif hasattr(model, 'decision_function'):
                    y_scores_ext = model.decision_function(X_test_ext_scaled)
                else:
                    y_scores_ext = y_pred_ext

                # Patient-level aggregation for external
                patient_true_ext, patient_pred_ext, patient_scores_ext = \
                    self._aggregate_patient_predictions(
                        patient_ids_ext, y_test_ext, y_scores_ext,
                        threshold=threshold
                    )

                n_hc = int((patient_pred_ext == 0).sum())
                n_pd = int((patient_pred_ext == 1).sum())

                try:
                    ext_cm = confusion_matrix(patient_true_ext, patient_pred_ext)
                except:
                    ext_cm = [[0, 0], [0, 0]]

                try:
                    ext_auc = roc_auc_score(patient_true_ext, patient_scores_ext)
                except:
                    ext_auc = None

                test_external_results = {
                    'patient_id': patient_true_ext.index.tolist(),
                    'y_true': patient_true_ext.tolist(),
                    'y_pred': patient_pred_ext.tolist(),
                    'y_score': patient_scores_ext.tolist(),
                    'n_classified_HC': n_hc,
                    'n_classified_PD': n_pd,
                    'accuracy': accuracy_score(patient_true_ext, patient_pred_ext) if len(patient_true_ext) > 1 else 0,
                    'auc': ext_auc
                }

                if verbose:
                    print(f"  {'='*32}")
                    print(f"  External patients (AC) - Classified HC: {n_hc}/{len(patient_pred_ext)}"
                          f" | Classified PD: {n_pd}/{len(patient_pred_ext)}")
                    if ext_auc is not None:
                        print(f"  External patients (AC) - Accuracy: {test_external_results['accuracy']:.3f}"
                              f" | AUC: {ext_auc:.3f}")
                    print(f"  {'='*32}")

            fold_results.append({
                'fold': fold,
                'best_params': best_params,
                'segment_metrics': segment_metrics,
                'patient_metrics': patient_metrics,
                'segment_preds': {
                    'patient_id': patient_ids[test_indices],
                    'y_true': y_test,
                    'y_pred': y_pred,
                    'y_score': y_scores
                },
                'patient_preds': {
                    'patient_id': test_patients,
                    'y_true': patient_true.loc[test_patients].values,
                    'y_pred': patient_pred.loc[test_patients].values,
                    'y_score': patient_scores.loc[test_patients].values
                },
                'test_external': test_external_results
            })

            full_segment_preds.append({
                'exercise': exercise,
                'fold': fold,
                'patient_id': patient_ids[test_indices],
                'y_true': y_test,
                'y_pred': y_pred,
                'y_score': y_scores
            })

            full_patient_preds.append({
                'exercise': exercise,
                'fold': fold,
                'patient_id': test_patients,
                'y_true': patient_true.values,
                'y_pred': patient_pred.values,
                'y_score': patient_scores.values
            })

        self.results_ = {
            'exercise': exercise,
            'feature_type': feature_type,
            'n_folds': self.n_splits,
            'fold_results': fold_results,
            'segment_metrics': {
                'mean': {k: np.mean(v) for k, v in all_segment_metrics.items()},
                'std': {k: np.std(v) for k, v in all_segment_metrics.items()}
            },
            'patient_metrics': {
                'mean': {k: np.mean(v) for k, v in all_patient_metrics.items()},
                'std': {k: np.std(v) for k, v in all_patient_metrics.items()}
            }
        }

        if save_dir:
            self.save_results(save_dir)

        return self.results_

    def predict(self, X):
        """Predict using last trained model."""
        if self.best_model_ is None:
            raise ValueError("Model not trained yet!")
        return self.best_model_.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using last trained model."""
        if self.best_model_ is None:
            raise ValueError("Model not trained yet!")
        return self.best_model_.predict_proba(X)

    def save_results(self, save_dir):
        """Save training results to directory organized by exercise/feature_type."""
        exercise = self.results_.get('exercise', 'default')
        feature_type = self.results_.get('feature_type', 'default')

        output_dir = os.path.join(save_dir, exercise, feature_type)
        os.makedirs(output_dir, exist_ok=True)

        fold_results = self.results_['fold_results']

        seg_rows = []
        pat_rows = []
        ac_rows = []
        all_best_params = []

        for fr in fold_results:
            fold = fr['fold']

            seg_preds = fr['segment_preds']
            for i in range(len(seg_preds['patient_id'])):
                seg_rows.append({
                    'fold': fold,
                    'patient_id': seg_preds['patient_id'][i],
                    'y_true': int(seg_preds['y_true'][i]),
                    'y_pred': int(seg_preds['y_pred'][i]),
                    'y_score': float(seg_preds['y_score'][i])
                })

            pat_preds = fr['patient_preds']
            for i in range(len(pat_preds['patient_id'])):
                pat_rows.append({
                    'fold': fold,
                    'patient_id': pat_preds['patient_id'][i],
                    'y_true': int(pat_preds['y_true'][i]),
                    'y_pred': int(pat_preds['y_pred'][i]),
                    'y_score': float(pat_preds['y_score'][i])
                })

            # External test predictions (e.g., AC group)
            test_ext = fr.get('test_external')
            if test_ext is not None and test_ext.get('patient_id') is not None:
                for i in range(len(test_ext['patient_id'])):
                    ac_rows.append({
                        'fold': fold,
                        'patient_id': test_ext['patient_id'][i],
                        'y_true': int(test_ext['y_true'][i]),
                        'y_pred': int(test_ext['y_pred'][i]),
                        'y_score': float(test_ext['y_score'][i])
                    })

            all_best_params.append(fr.get('best_params', {}))

        pd.DataFrame(seg_rows).to_csv(
            os.path.join(output_dir, 'predictions_segment_wise.csv'),
            index=False
        )

        pd.DataFrame(pat_rows).to_csv(
            os.path.join(output_dir, 'predictions_patient_wise.csv'),
            index=False
        )

        if ac_rows:
            pd.DataFrame(ac_rows).to_csv(
                os.path.join(output_dir, 'predictions_ac.csv'),
                index=False
            )

        seg_metrics = self.results_['segment_metrics']
        pat_metrics = self.results_['patient_metrics']

        # Aggregate external test results (e.g., AC group)
        ext_accuracies = []
        ext_aucs = []
        ext_hc_counts = []
        ext_pd_counts = []
        for fr in fold_results:
            ext = fr.get('test_external')
            if ext:
                if ext.get('accuracy') is not None:
                    ext_accuracies.append(ext['accuracy'])
                if ext.get('auc') is not None:
                    ext_aucs.append(ext['auc'])
                ext_hc_counts.append(ext.get('n_classified_HC', 0))
                ext_pd_counts.append(ext.get('n_classified_PD', 0))

        external_summary = {}
        if ext_accuracies:
            external_summary = {
                'classified_HC_mean': float(np.mean(ext_hc_counts)),
                'classified_PD_mean': float(np.mean(ext_pd_counts)),
                'accuracy_mean': float(np.mean(ext_accuracies)),
                'accuracy_std': float(np.std(ext_accuracies)),
            }
            if ext_aucs:
                external_summary['auc_mean'] = float(np.mean(ext_aucs))
                external_summary['auc_std'] = float(np.std(ext_aucs))

        summary = {
            'exercise': exercise,
            'feature_type': feature_type,
            'n_folds': self.results_['n_folds'],
            'best_params_per_fold': all_best_params,
            'segment_level': {
                'mean': seg_metrics['mean'],
                'std': seg_metrics['std']
            },
            'patient_level': {
                'mean': pat_metrics['mean'],
                'std': pat_metrics['std']
            }
        }

        if external_summary:
            summary['external_test'] = external_summary

        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"RESULTS SUMMARY\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Exercise: {exercise}\n")
            f.write(f"Feature Type: {feature_type}\n")
            f.write(f"Number of Folds: {self.results_['n_folds']}\n\n")

            f.write(f"{'-'*60}\n")
            f.write(f"SEGMENT-LEVEL METRICS\n")
            f.write(f"{'-'*60}\n")
            for metric, values in seg_metrics['mean'].items():
                std = seg_metrics['std'][metric]
                f.write(f"{metric:<15}: {values:.3f} (+/- {std:.3f})\n")

            f.write(f"\n{'-'*60}\n")
            f.write(f"PATIENT-LEVEL METRICS\n")
            f.write(f"{'-'*60}\n")
            for metric, values in pat_metrics['mean'].items():
                std = pat_metrics['std'][metric]
                f.write(f"{metric:<15}: {values:.3f} (+/- {std:.3f})\n")

            f.write(f"\n{'='*60}\n")
            f.write(f"PER-FOLD RESULTS\n")
            f.write(f"{'='*60}\n\n")

            for fr in fold_results:
                fold = fr['fold']
                f.write(f"--- Fold {fold} ---\n")
                f.write(f"Best Parameters: {fr.get('best_params', {})}\n\n")

                seg = fr['segment_metrics']
                f.write("Segment-Level:\n")
                f.write(f"  Confusion Matrix: {seg['confusion_matrix']}\n")
                for k, v in seg.items():
                    if k != 'confusion_matrix':
                        f.write(f"  {k}: {v:.3f}\n")

                pat = fr['patient_metrics']
                f.write("Patient-Level:\n")
                f.write(f"  Confusion Matrix: {pat['confusion_matrix']}\n")
                for k, v in pat.items():
                    if k != 'confusion_matrix':
                        f.write(f"  {k}: {v:.3f}\n")

                # External test (AC group) per fold
                ext = fr.get('test_external')
                if ext and ext.get('patient_id') is not None:
                    f.write("External test (AC group):\n")
                    f.write(f"  Classified HC: {ext['n_classified_HC']} | "
                            f"Classified PD: {ext['n_classified_PD']}\n")
                    f.write(f"  Accuracy: {ext['accuracy']:.3f}\n")
                    if ext.get('auc') is not None:
                        f.write(f"  AUC: {ext['auc']:.3f}\n")

                f.write("\n")

        # External test summary section in results.txt
        has_external = any(fr.get('test_external') and fr['test_external'].get('patient_id') is not None
                          for fr in fold_results)
        if has_external:
            with open(os.path.join(output_dir, 'results.txt'), 'a') as f:
                f.write(f"{'='*60}\n")
                f.write(f"EXTERNAL TEST (AC GROUP) - SUMMARY\n")
                f.write(f"{'='*60}\n\n")

                ext_ac_counts = []
                ext_pd_counts = []
                ext_accs = []
                ext_auc_vals = []

                for fr in fold_results:
                    ext = fr.get('test_external')
                    if ext:
                        ext_ac_counts.append(ext['n_classified_HC'])
                        ext_pd_counts.append(ext['n_classified_PD'])
                        ext_accs.append(ext['accuracy'])
                        if ext.get('auc') is not None:
                            ext_auc_vals.append(ext['auc'])

                f.write(f"Classified HC (DaTSCAN-like): {np.mean(ext_ac_counts):.1f} ± {np.std(ext_ac_counts):.1f}\n")
                f.write(f"Classified PD (DaTSCAN-like): {np.mean(ext_pd_counts):.1f} ± {np.std(ext_pd_counts):.1f}\n")
                f.write(f"Accuracy: {np.mean(ext_accs):.3f} ± {np.std(ext_accs):.3f}\n")
                if ext_auc_vals:
                    f.write(f"AUC: {np.mean(ext_auc_vals):.3f} ± {np.std(ext_auc_vals):.3f}\n")
                f.write("\n")

        print(f"\nResults saved to: {output_dir}")
        print(f"  - predictions_segment_wise.csv")
        print(f"  - predictions_patient_wise.csv")
        if ac_rows:
            print(f"  - predictions_ac.csv")
        print(f"  - metrics.json")
        print(f"  - results.txt")