import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
from tcn import TCN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_real_data():
    """
    Load real CO₂ data from the GAMS indoor dataset.
    """
    # Load data
    url = "http://raw.githubusercontent.com/twairball/gams-dataset/refs/heads/master/gams_indoor.csv"
    df = pd.read_csv(url, parse_dates=['ts'])
    
    # Sort by timestamp and reset index
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Select relevant columns and handle any missing values
    df = df[['ts', 'co2']].dropna()
    
    return df

def preprocess_data(data, lookback=12, horizon=12, train_split=0.7, val_split=0.2):
    """
    Preprocess data: normalize and create sequences.
    """
    # Normalize data
    scaler = MinMaxScaler()
    co2_normalized = scaler.fit_transform(data['co2'].values.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(len(co2_normalized) - lookback - horizon + 1):
        X.append(co2_normalized[i:i+lookback])
        y.append(co2_normalized[i+lookback:i+lookback+horizon])
    X, y = np.array(X), np.array(y)
    
    # Split data
    train_size = int(len(X) * train_split)
    val_size = int(len(X) * val_split)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler

def build_tcn_model(input_shape, output_length):
    """
    Build TCN model.
    """
    model = Sequential([
        TCN(64, kernel_size=3, dilations=[1, 2, 4, 8], 
            return_sequences=False, input_shape=input_shape),
        Dense(output_length)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_simple_rnn_model(input_shape, output_length):
    """
    Build SimpleRNN model.
    """
    model = Sequential([
        SimpleRNN(50, input_shape=input_shape),
        Dense(output_length)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_seq2seq_model(input_shape, output_length):
    """
    Build Sequence-to-Sequence model.
    """
    # Encoder
    encoder_inputs = Input(shape=input_shape, name='encoder_input')
    encoder = LSTM(50, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(output_length, 1), name='decoder_input')
    decoder_lstm = LSTM(50, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(1)
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model, data, model_name, epochs=100, batch_size=32):
    """
    Train model with early stopping.
    """
    (X_train, y_train), (X_val, y_val), _, _ = data
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # For Seq2Seq model, prepare decoder inputs
    if model_name == 'Seq2Seq':
        # Create zero inputs for decoder
        decoder_input_train = np.zeros((X_train.shape[0], y_train.shape[1], 1))
        decoder_input_val = np.zeros((X_val.shape[0], y_val.shape[1], 1))
        
        # Train with both encoder and decoder inputs
        history = model.fit(
            [X_train, decoder_input_train], y_train,
            validation_data=([X_val, decoder_input_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
    else:
        # Regular training for other models
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
    
    # Save model in the newer Keras format
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save(f'models/{model_name}.keras')
    
    return history

def is_seq2seq_model(model):
    """
    Helper function to determine if a model is a Sequence-to-Sequence model.
    """
    return isinstance(model, Model) and len(model.inputs) > 1

def evaluate_model(model, data, scaler, model_name):
    """
    Evaluate model performance.
    """
    _, _, (X_test, y_test), _ = data
    
    # Make predictions
    if model_name == 'Seq2Seq':  # Check by model name instead of structure
        decoder_input_test = np.zeros((X_test.shape[0], y_test.shape[1], 1))
        y_pred = model.predict([X_test, decoder_input_test])
    else:  # Other models
        y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2 = r2_score(y_test_orig, y_pred_orig)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'predictions': y_pred_orig,
        'actual': y_test_orig
    }

def recommend_ventilation(co2_predictions):
    """
    Generate ventilation recommendations based on predicted CO₂ levels.
    """
    recommendations = []
    for i, co2 in enumerate(co2_predictions):
        if co2 > 1500:
            recommendations.append(f"Open windows and turn on ventilation for 45 minutes")
        elif co2 > 1100:
            recommendations.append(f"Open windows for 30 minutes")
        elif co2 > 900:
            recommendations.append(f"Open windows for 15 minutes")
        else:
            recommendations.append("No action needed")
    return recommendations

def plot_results(actual, predictions, recommendations, model_name):
    """
    Plot predictions and recommendations.
    """
    plt.figure(figsize=(15, 8))
    plt.plot(actual, label='Actual', alpha=0.7)
    plt.plot(predictions, label='Predicted', alpha=0.7)
    
    # Add recommendation markers
    for i, rec in enumerate(recommendations):
        if rec != "No action needed":
            plt.axvline(x=i, color='r', alpha=0.2, linestyle='--')
    
    plt.title(f'CO₂ Predictions - {model_name}')
    plt.xlabel('Time Steps')
    plt.ylabel('CO₂ (ppm)')
    plt.legend()
    
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(f'plots/{model_name}_predictions.png')
    plt.close()

def main():
    # Load real data instead of generating synthetic data
    print("Loading real CO₂ data...")
    data = load_real_data()
    
    # Preprocess data
    print("Preprocessing data...")
    processed_data = preprocess_data(data)
    
    # Define models
    models = {
        'TCN': build_tcn_model((12, 1), 12),
        'SimpleRNN': build_simple_rnn_model((12, 1), 12),
        'Seq2Seq': build_seq2seq_model((12, 1), 12)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        history = train_model(model, processed_data, name)
        
        print(f"Evaluating {name} model...")
        results[name] = evaluate_model(model, processed_data, processed_data[-1], name)
        
        # Generate recommendations
        recommendations = recommend_ventilation(results[name]['predictions'])
        
        # Plot results
        plot_results(
            results[name]['actual'],
            results[name]['predictions'],
            recommendations,
            name
        )
        
        # Print metrics
        print(f"\n{name} Model Performance:")
        print(f"RMSE: {results[name]['rmse']:.2f} ppm")
        print(f"R²: {results[name]['r2']:.4f}")
        print(f"MAE: {results[name]['mae']:.2f} ppm")
        
        # Save predictions and recommendations
        if not os.path.exists('results'):
            os.makedirs('results')
        pd.DataFrame({
            'actual': results[name]['actual'].flatten(),
            'predicted': results[name]['predictions'].flatten(),
            'recommendations': recommendations
        }).to_csv(f'results/{name}_results.csv', index=False)

if __name__ == "__main__":
    main() 