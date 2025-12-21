#!/usr/bin/env python3
"""
Run time-series ML analysis for nanodomain dynamics prediction
"""

import os
import numpy as np
import pandas as pd
from .ml_timeseries import CompositionTimeSeriesPredictor
from visualization.plots_timeseries import (
    plot_training_history,
    plot_prediction_vs_actual,
    plot_time_series_example,
    plot_long_term_prediction,
    create_nanodomain_animation
)


def run_timeseries_ml_analysis(df, lipid_types, target_lipid, output_dir,
                               lookback=10, prediction_horizon=1,
                               model_type='lstm', test_size=0.2,
                               epochs=100, batch_size=32, lr=0.001):
    """Run complete time-series ML analysis

    Parameters
    ----------
    df : pandas.DataFrame
        Composition dataframe with columns: frame, protein, {lipid}_fraction, target_lipid_bound
    lipid_types : list of str
        Lipid types to predict
    target_lipid : str
        Target lipid name (GM3)
    output_dir : str
        Output directory
    lookback : int
        Number of past frames to use
    prediction_horizon : int
        Number of frames ahead to predict
    model_type : str
        'lstm', 'gru', or 'baseline'
    test_size : float
        Fraction for test set
    epochs : int
        Training epochs
    batch_size : int
        Batch size
    lr : float
        Learning rate

    Returns
    -------
    dict
        Results including model, metrics, predictions
    """
    print("\n" + "="*70)
    print("TIME-SERIES MACHINE LEARNING ANALYSIS")
    print("="*70)
    print(f"Model: {model_type.upper()}")
    print(f"Lookback: {lookback} frames")
    print(f"Prediction horizon: {prediction_horizon} frames")
    print("="*70)

    # Create output directory
    ml_dir = os.path.join(output_dir, 'timeseries_ml')
    os.makedirs(ml_dir, exist_ok=True)

    # Initialize predictor
    predictor = CompositionTimeSeriesPredictor(
        lookback=lookback,
        prediction_horizon=prediction_horizon,
        model_type=model_type
    )

    # Prepare sequences
    X, y, proteins, frames = predictor.prepare_sequences(df, lipid_types, target_lipid)

    if len(X) == 0:
        print("ERROR: No valid sequences created. Check data quality.")
        return None

    # Train/test split (by protein to avoid data leakage)
    unique_proteins = list(set(proteins))
    n_test = max(1, int(len(unique_proteins) * test_size))
    np.random.seed(42)
    test_proteins = set(np.random.choice(unique_proteins, n_test, replace=False))

    train_mask = np.array([p not in test_proteins for p in proteins])
    test_mask = ~train_mask

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"\nTrain/Test Split:")
    print(f"  Train: {len(X_train)} sequences ({len(unique_proteins) - n_test} proteins)")
    print(f"  Test:  {len(X_test)} sequences ({n_test} proteins)")

    # Further split train into train/val
    n_val = int(len(X_train) * 0.2)
    indices = np.random.permutation(len(X_train))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train_final = X_train[train_indices]
    y_train_final = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]

    print(f"  Validation: {len(X_val)} sequences")

    # Train model
    history = predictor.train(
        X_train_final, y_train_final,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=15
    )

    # Plot training history
    plot_training_history(history, os.path.join(ml_dir, 'training_history.png'))

    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70)

    results = predictor.evaluate(X_test, y_test)

    print(f"\nOverall Performance:")
    print(f"  MAE:  {results['mae']:.4f}")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  R²:   {results['r2']:.4f}")

    print(f"\nPer-Lipid Performance:")
    for i, lipid in enumerate(lipid_types):
        metrics = results['per_lipid'][f'lipid_{i}']
        print(f"  {lipid:6s}: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")

    # Plot predictions vs actual
    plot_prediction_vs_actual(
        y_test, results['predictions'], lipid_types,
        os.path.join(ml_dir, 'prediction_vs_actual.png')
    )

    # Example time-series prediction
    print("\n" + "="*70)
    print("GENERATING EXAMPLE PREDICTIONS")
    print("="*70)

    # Find a protein with sufficient data
    for protein_name in df['protein'].unique():
        protein_df = df[df['protein'] == protein_name].sort_values('frame').reset_index(drop=True)

        if len(protein_df) < lookback + 50:
            continue

        # Extract sequence
        feature_cols = []
        for lipid in lipid_types:
            feature_cols.append(f'{lipid}_fraction')
        if 'target_lipid_bound' in protein_df.columns:
            feature_cols.append('target_lipid_bound')
        if f'{target_lipid}_count' in protein_df.columns:
            feature_cols.append(f'{target_lipid}_count')

        start_idx = 0
        n_predict = min(50, len(protein_df) - lookback - 1)

        initial_seq = protein_df.loc[start_idx:start_idx+lookback-1, feature_cols].values

        # Autoregressive prediction
        predictions = predictor.predict_long_term(initial_seq, n_predict, lipid_types)

        # True values
        target_cols = [f'{lipid}_fraction' for lipid in lipid_types]
        true_vals = protein_df.loc[start_idx+lookback:start_idx+lookback+n_predict-1, target_cols].values

        # Binding states
        binding_states = protein_df.loc[start_idx:start_idx+lookback+n_predict-1, 'target_lipid_bound'].values

        # Plot
        full_true = np.vstack([
            protein_df.loc[start_idx:start_idx+lookback-1, target_cols].values,
            true_vals
        ])

        plot_time_series_example(
            full_true, predictions, lipid_types, binding_states,
            os.path.join(ml_dir, f'timeseries_example_{protein_name}.png'),
            n_lookback=lookback
        )

        break

    # Long-term prediction (1000 steps)
    print("\n" + "="*70)
    print("LONG-TERM PREDICTION (1000 steps)")
    print("="*70)

    # Use first available protein
    protein_name = df['protein'].unique()[0]
    protein_df = df[df['protein'] == protein_name].sort_values('frame').reset_index(drop=True)

    if len(protein_df) >= lookback:
        initial_seq = protein_df.loc[:lookback-1, feature_cols].values
        initial_comp = protein_df.loc[lookback-1, target_cols].values

        long_predictions = predictor.predict_long_term(initial_seq, 1000, lipid_types)

        plot_long_term_prediction(
            long_predictions, lipid_types,
            os.path.join(ml_dir, 'long_term_prediction.png'),
            initial_state=initial_comp,
            frame_step=prediction_horizon
        )

        # Create animation
        print("\nCreating animation...")
        create_nanodomain_animation(
            long_predictions, lipid_types, ml_dir,
            frame_step=prediction_horizon, fps=20
        )

    # Save model and results
    import pickle
    with open(os.path.join(ml_dir, 'predictor.pkl'), 'wb') as f:
        pickle.dump(predictor, f)

    with open(os.path.join(ml_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print("\n" + "="*70)
    print("TIME-SERIES ML ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {ml_dir}")

    return {
        'predictor': predictor,
        'results': results,
        'history': history,
        'X_test': X_test,
        'y_test': y_test
    }
