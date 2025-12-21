#!/usr/bin/env python3
"""
Machine learning for composition prediction
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class CompositionPredictor:
    """Predict composition changes from mediator binding"""

    def __init__(self, test_size=0.3, random_seed=42):
        """Initialize predictor

        Parameters
        ----------
        test_size : float, default 0.3
            Fraction of data for test set
        random_seed : int, default 42
            Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}

    def prepare_features(self, df, mediator_cols, target_col):
        """Prepare features and target for ML

        Parameters
        ----------
        df : pandas.DataFrame
            Input data
        mediator_cols : list of str
            Mediator feature columns
        target_col : str
            Target composition column

        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        # Select features and target
        X = df[mediator_cols].values
        y = df[target_col].values

        # Remove NaN
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]

        if len(X) == 0:
            raise ValueError("No valid data after removing NaN")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed
        )

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and compare

        Parameters
        ----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training targets
        X_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            Test targets

        Returns
        -------
        dict
            Model performance metrics
        """
        print("\n" + "="*70)
        print("TRAINING PREDICTION MODELS")
        print("="*70)

        models_to_try = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.01, max_iter=5000),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10,
                                                  random_state=self.random_seed),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                          random_state=self.random_seed)
        }

        results = {}

        print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples\n")
        print(f"{'Model':<20} {'Train R²':>10} {'Test R²':>10} {'MAE':>10} {'RMSE':>10}")
        print("-" * 70)

        for name, model in models_to_try.items():
            # Train
            model.fit(X_train, y_train)

            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'mae': mae,
                'rmse': rmse,
                'predictions': y_test_pred
            }

            print(f"{name:<20} {train_r2:>10.4f} {test_r2:>10.4f} {mae:>10.4f} {rmse:>10.4f}")

            # Store feature importance if available
            if hasattr(model, 'coef_'):
                self.feature_importance[name] = model.coef_
            elif hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_

        print("="*70)

        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        print(f"\n✓ Best model: {best_model_name} (Test R² = {results[best_model_name]['test_r2']:.4f})")

        self.models = results
        return results

    def cross_validate(self, X, y, model_name='Ridge', cv=5):
        """Perform cross-validation

        Parameters
        ----------
        X : numpy.ndarray
            Features
        y : numpy.ndarray
            Targets
        model_name : str, default 'Ridge'
            Model to use
        cv : int, default 5
            Number of CV folds

        Returns
        -------
        dict
            CV results
        """
        print(f"\nPerforming {cv}-fold cross-validation for {model_name}...")

        if model_name == 'Ridge':
            model = Ridge(alpha=1.0)
        elif model_name == 'Lasso':
            model = Lasso(alpha=0.01, max_iter=5000)
        elif model_name == 'RandomForest':
            model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                         random_state=self.random_seed)
        elif model_name == 'GradientBoosting':
            model = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                             random_state=self.random_seed)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

        print(f"  CV R² scores: {scores}")
        print(f"  Mean CV R²: {scores.mean():.4f} ± {scores.std():.4f}")

        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }

    def predict_composition_changes(self, df, mediator_cols, lipid_types):
        """Train models to predict composition changes for each lipid

        Parameters
        ----------
        df : pandas.DataFrame
            Input data
        mediator_cols : list of str
            Mediator feature columns
        lipid_types : list of str
            Lipid types to predict

        Returns
        -------
        dict
            Prediction results for each lipid
        """
        print("\n" + "="*70)
        print("COMPOSITION CHANGE PREDICTION")
        print("="*70)

        all_results = {}

        for lipid_type in lipid_types:
            target_col = f'{lipid_type}_ratio'

            if target_col not in df.columns:
                print(f"\nWARNING: {target_col} not found, skipping")
                continue

            print(f"\n{'='*70}")
            print(f"Predicting: {lipid_type}")
            print('='*70)

            try:
                # Prepare data
                X_train, X_test, y_train, y_test = self.prepare_features(
                    df, mediator_cols, target_col
                )

                # Train models
                results = self.train_models(X_train, y_train, X_test, y_test)

                all_results[lipid_type] = results

            except Exception as e:
                print(f"ERROR: {str(e)}")
                continue

        return all_results

    def cross_system_validation(self, df_train, df_test, mediator_cols, lipid_types):
        """Train on one system, test on another

        Parameters
        ----------
        df_train : pandas.DataFrame
            Training data (e.g., EphA2)
        df_test : pandas.DataFrame
            Test data (e.g., Notch)
        mediator_cols : list of str
            Mediator feature columns
        lipid_types : list of str
            Lipid types

        Returns
        -------
        dict
            Cross-system validation results
        """
        print("\n" + "="*70)
        print("CROSS-SYSTEM VALIDATION")
        print("="*70)
        print("Training on one system, testing on another...")

        results = {}

        for lipid_type in lipid_types:
            target_col = f'{lipid_type}_ratio'

            if target_col not in df_train.columns or target_col not in df_test.columns:
                continue

            print(f"\n{lipid_type}:")

            # Prepare training data
            X_train = df_train[mediator_cols].values
            y_train = df_train[target_col].values
            mask_train = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
            X_train = X_train[mask_train]
            y_train = y_train[mask_train]

            # Prepare test data
            X_test = df_test[mediator_cols].values
            y_test = df_test[target_col].values
            mask_test = np.isfinite(X_test).all(axis=1) & np.isfinite(y_test)
            X_test = X_test[mask_test]
            y_test = y_test[mask_test]

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train simple model
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)

            # Predict on test system
            y_pred = model.predict(X_test_scaled)

            # Metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            corr = np.corrcoef(y_test, y_pred)[0, 1]

            results[lipid_type] = {
                'r2': r2,
                'mae': mae,
                'correlation': corr,
                'n_train': len(X_train),
                'n_test': len(X_test)
            }

            print(f"  R² on test system: {r2:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Correlation: {corr:.4f}")

        print("="*70)
        return results
