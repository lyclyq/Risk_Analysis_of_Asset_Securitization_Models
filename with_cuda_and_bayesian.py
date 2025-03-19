import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf


try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
except ImportError:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Integer, Real


KAGGLE_DATASET = "wordsforthewise/lending-club"


def preprocess_data():
    print("...")
    chunks = pd.read_csv("data/accepted_2007_to_2018Q4.csv", chunksize=100000, low_memory=False)
    data = pd.concat(chunks)
    data = data[['loan_amnt', 'int_rate', 'fico_range_high', 'fico_range_low', 'annual_inc', 'dti', 'loan_status']]
    data = data[(data['loan_status'] == 'Fully Paid') | (data['loan_status'] == 'Charged Off')]
    data['loan_status'] = data['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
    data.fillna(data.mean(), inplace=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.drop(columns=['loan_status']))
    return data, scaled_data


def create_lstm_dataset(scaled_data, labels, time_steps=10):
    X, Y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i+time_steps])
        Y.append(labels.iloc[i+time_steps])
    return np.array(X), np.array(Y)


def build_lstm(optimizer='adam', lstm_units=50, dropout_rate=0.2):
    model = Sequential([
        LSTM(lstm_units, activation='relu', return_sequences=True, input_shape=(10, 6)),
        Dropout(dropout_rate),
        LSTM(lstm_units, activation='relu', return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def bayesian_optimization(X_train, Y_train):
    print("Running Bayesian Optimization for hyperparameter tuning...")

    model = KerasClassifier(build_fn=build_lstm, verbose=0)

    param_space = {
        'lstm_units': Integer(30, 100),
        'dropout_rate': Real(0.1, 0.5),
        'optimizer': ['adam', 'rmsprop'],
        'batch_size': Integer(16, 64),
        'epochs': Integer(10, 50)
    }

    bayes_search = BayesSearchCV(model, param_space, n_iter=10, cv=3, n_jobs=1)
    bayes_search.fit(X_train, Y_train)

    print("Best parameters found:", bayes_search.best_params_)
    return bayes_search.best_params_


def train_lstm(X_train, Y_train, X_test, Y_test, best_params):
    print("Training LSTM model with optimized parameters...")

    model = build_lstm(
        optimizer=best_params['optimizer'],
        lstm_units=best_params['lstm_units'],
        dropout_rate=best_params['dropout_rate']
    )

    history = model.fit(X_train, Y_train,
                        epochs=best_params['epochs'],
                        batch_size=best_params['batch_size'],
                        validation_data=(X_test, Y_test))
    return model, history


def evaluate_model(model, X_test, Y_test, history):
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(Y_test, predictions)
    print(f"Model accuracy: {acc:.2f}")

if __name__ == "__main__":
    raw_data, scaled_data = preprocess_data()
    X, Y = create_lstm_dataset(scaled_data, raw_data['loan_status'])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # bayesian
    best_params = bayesian_optimization(X_train, Y_train)

    # train
    model, history = train_lstm(X_train, Y_train, X_test, Y_test, best_params)

    evaluate_model(model, X_test, Y_test, history)
