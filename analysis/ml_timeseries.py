#!/usr/bin/env python3
"""
Time-series machine learning for nanodomain dynamics prediction

Predicts future lipid composition from past composition and GM3 binding history
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Using simple baseline models only.")


class TimeSeriesDataset(Dataset):
    """Dataset for time-series composition data"""

    def __init__(self, X, y):
        """Initialize dataset

        Parameters
        ----------
        X : numpy.ndarray
            Input sequences (n_samples, sequence_length, n_features)
        y : numpy.ndarray
            Target values (n_samples, n_outputs)
        """
        self.X = torch.FloatTensor(X) if TORCH_AVAILABLE else X
        self.y = torch.FloatTensor(y) if TORCH_AVAILABLE else y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPredictor(nn.Module):
    """LSTM model for composition prediction"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3, dropout=0.2):
        """Initialize LSTM

        Parameters
        ----------
        input_size : int
            Number of input features
        hidden_size : int, default 64
            LSTM hidden size
        num_layers : int, default 2
            Number of LSTM layers
        output_size : int, default 3
            Number of output features (lipid types)
        dropout : float, default 0.2
            Dropout rate
        """
        super(LSTMPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input (batch_size, sequence_length, input_size)

        Returns
        -------
        torch.Tensor
            Output (batch_size, output_size)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Take last timestep
        last_output = lstm_out[:, -1, :]

        # Fully connected
        out = self.fc(last_output)

        return out


class GRUPredictor(nn.Module):
    """GRU model for composition prediction"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3, dropout=0.2):
        super(GRUPredictor, self).__init__()

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        out = self.fc(last_output)
        return out


class CompositionTimeSeriesPredictor:
    """Time-series predictor for lipid composition dynamics"""

    def __init__(self, lookback=10, prediction_horizon=1, model_type='lstm'):
        """Initialize predictor

        Parameters
        ----------
        lookback : int, default 10
            Number of past frames to use
        prediction_horizon : int, default 1
            Number of frames ahead to predict
        model_type : str, default 'lstm'
            'lstm', 'gru', or 'baseline'
        """
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None

    def prepare_sequences(self, df, lipid_types, target_lipid='DPG3'):
        """Prepare time-series sequences from dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            Composition data with 'frame', 'protein', composition columns, 'target_lipid_bound'
        lipid_types : list of str
            Lipid types to predict
        target_lipid : str
            Target lipid name

        Returns
        -------
        tuple
            (X, y, protein_names, frame_indices)
        """
        print("\n" + "="*70)
        print("PREPARING TIME-SERIES SEQUENCES")
        print("="*70)

        all_X = []
        all_y = []
        all_proteins = []
        all_frames = []

        # Process each protein separately
        for protein_name in sorted(df['protein'].unique()):
            protein_df = df[df['protein'] == protein_name].sort_values('frame').reset_index(drop=True)

            # Feature columns
            feature_cols = []
            for lipid in lipid_types:
                feature_cols.append(f'{lipid}_fraction')

            # Add target lipid binding state
            if 'target_lipid_bound' in protein_df.columns:
                feature_cols.append('target_lipid_bound')

            # Add target lipid count if available
            if f'{target_lipid}_count' in protein_df.columns:
                feature_cols.append(f'{target_lipid}_count')

            # Target columns (compositions to predict)
            target_cols = [f'{lipid}_fraction' for lipid in lipid_types]

            # Create sliding windows
            for i in range(len(protein_df) - self.lookback - self.prediction_horizon + 1):
                # Input: past lookback frames
                X_window = protein_df.loc[i:i+self.lookback-1, feature_cols].values.astype(np.float64)

                # Output: composition at prediction_horizon frames ahead
                y_target = protein_df.loc[i+self.lookback+self.prediction_horizon-1, target_cols].values.astype(np.float64)

                # Only include if no NaN
                if not (np.isnan(X_window).any() or np.isnan(y_target).any()):
                    all_X.append(X_window)
                    all_y.append(y_target)
                    all_proteins.append(protein_name)
                    all_frames.append(protein_df.loc[i+self.lookback+self.prediction_horizon-1, 'frame'])

        X = np.array(all_X)  # (n_samples, lookback, n_features)
        y = np.array(all_y)  # (n_samples, n_outputs)

        print(f"Created {len(X)} sequences from {len(df['protein'].unique())} proteins")
        print(f"Input shape: {X.shape} (samples, lookback={self.lookback}, features)")
        print(f"Output shape: {y.shape} (samples, lipid_types={len(lipid_types)})")
        print(f"Prediction horizon: {self.prediction_horizon} frames ahead")
        print("="*70)

        return X, y, all_proteins, all_frames

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32, lr=0.001, patience=10):
        """Train time-series model

        Parameters
        ----------
        X_train : numpy.ndarray
            Training sequences (n_samples, lookback, n_features)
        y_train : numpy.ndarray
            Training targets (n_samples, n_outputs)
        X_val : numpy.ndarray, optional
            Validation sequences
        y_val : numpy.ndarray, optional
            Validation targets
        epochs : int, default 100
            Number of training epochs
        batch_size : int, default 32
            Batch size
        lr : float, default 0.001
            Learning rate
        patience : int, default 10
            Early stopping patience

        Returns
        -------
        dict
            Training history
        """
        print("\n" + "="*70)
        print("TRAINING TIME-SERIES MODEL")
        print("="*70)
        print(f"Model type: {self.model_type.upper()}")

        if not TORCH_AVAILABLE and self.model_type in ['lstm', 'gru']:
            print("WARNING: PyTorch not available, using baseline model")
            self.model_type = 'baseline'

        # Reshape for scaling: (n_samples * lookback, n_features)
        n_samples, lookback, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)

        # Scale features
        X_train_scaled_2d = self.scaler_X.fit_transform(X_train_2d)
        X_train_scaled = X_train_scaled_2d.reshape(n_samples, lookback, n_features)

        # Scale targets
        y_train_scaled = self.scaler_y.fit_transform(y_train)

        if X_val is not None:
            n_val, _, _ = X_val.shape
            X_val_2d = X_val.reshape(-1, n_features)
            X_val_scaled_2d = self.scaler_X.transform(X_val_2d)
            X_val_scaled = X_val_scaled_2d.reshape(n_val, lookback, n_features)
            y_val_scaled = self.scaler_y.transform(y_val)

        if self.model_type == 'baseline':
            # Simple baseline: use last observation
            self.model = 'baseline'
            print("Using baseline model (last observation carried forward)")
            return {'train_loss': [0], 'val_loss': [0]}

        # Deep learning models
        n_outputs = y_train.shape[1]

        if self.model_type == 'lstm':
            self.model = LSTMPredictor(n_features, hidden_size=64, num_layers=2,
                                      output_size=n_outputs, dropout=0.2)
        elif self.model_type == 'gru':
            self.model = GRUPredictor(n_features, hidden_size=64, num_layers=2,
                                     output_size=n_outputs, dropout=0.2)

        self.model = self.model.to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Data loaders
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # Validation
            if X_val is not None:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_losses.append(loss.item())

                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.6f}")

        print("="*70)
        return history

    def predict(self, X):
        """Predict future compositions

        Parameters
        ----------
        X : numpy.ndarray
            Input sequences (n_samples, lookback, n_features)

        Returns
        -------
        numpy.ndarray
            Predictions (n_samples, n_outputs)
        """
        if self.model_type == 'baseline':
            # Baseline: return last observation's composition (first 3 features)
            return X[:, -1, :3]

        # Scale
        n_samples, lookback, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        X_scaled_2d = self.scaler_X.transform(X_2d)
        X_scaled = X_scaled_2d.reshape(n_samples, lookback, n_features)

        # Predict
        self.model.eval()
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions_scaled = outputs.cpu().numpy()

        # Inverse scale
        predictions = self.scaler_y.inverse_transform(predictions_scaled)

        return predictions

    def predict_long_term(self, initial_sequence, n_steps, lipid_types):
        """Autoregressive prediction for long-term dynamics

        Parameters
        ----------
        initial_sequence : numpy.ndarray
            Initial sequence (lookback, n_features)
        n_steps : int
            Number of future steps to predict
        lipid_types : list of str
            Lipid type names

        Returns
        -------
        numpy.ndarray
            Predicted compositions (n_steps, n_lipids)
        """
        print(f"\nPredicting {n_steps} steps ahead (autoregressive)...")

        predictions = []
        current_sequence = initial_sequence.copy()

        for step in range(n_steps):
            # Predict next step
            X_input = current_sequence.reshape(1, self.lookback, -1)
            y_pred = self.predict(X_input)[0]  # (n_lipids,)

            predictions.append(y_pred)

            # Update sequence: shift and append prediction
            # Assume first n_lipids features are compositions
            n_lipids = len(lipid_types)
            new_features = current_sequence[-1].copy()
            new_features[:n_lipids] = y_pred

            # Shift sequence
            current_sequence = np.vstack([current_sequence[1:], new_features])

            if (step + 1) % 100 == 0:
                print(f"  Predicted {step+1}/{n_steps} steps")

        return np.array(predictions)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance

        Parameters
        ----------
        X_test : numpy.ndarray
            Test sequences
        y_test : numpy.ndarray
            Test targets

        Returns
        -------
        dict
            Evaluation metrics
        """
        y_pred = self.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Per-lipid metrics
        n_lipids = y_test.shape[1]
        per_lipid_metrics = {}

        for i in range(n_lipids):
            per_lipid_metrics[f'lipid_{i}'] = {
                'mae': mean_absolute_error(y_test[:, i], y_pred[:, i]),
                'r2': r2_score(y_test[:, i], y_pred[:, i])
            }

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'per_lipid': per_lipid_metrics,
            'predictions': y_pred
        }
