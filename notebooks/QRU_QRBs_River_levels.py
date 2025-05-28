# -*- coding: utf-8 -*-
import os
import pandas as pd
import pennylane as qml
import torch
import numpy as np
import matplotlib.pyplot as plt

# Global parameters
depth = 5        # network depth (number of layers)
nb_epoch = 100   # number of training epochs
lr = 0.01        # learning rate
seq_length = 3   # sequence length (time series lag)
tolerance = 0.1  # relative tolerance for accuracy calculation

# Directory for saving results
results_dir = "QRU_QRBs_River_level"
os.makedirs(results_dir, exist_ok=True)

##############################################################################
# 1. Data preprocessing
##############################################################################
def preprocess_river_data(file_path):
    df = pd.read_csv(file_path, skiprows=2)
    df.columns = ['date', 'wlvalue', 'fvalue']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df.set_index('date', inplace=True)
    df['wlvalue_normalized'] = (
        (df['wlvalue'] - df['wlvalue'].min())
        / (df['wlvalue'].max() - df['wlvalue'].min())
        * (2 * np.pi)
    )
    return df[['wlvalue_normalized']]

def create_sequences(data, seq_length=3):
    X, y = [], []
    for i in range(len(data) - seq_length):
        wl_seq = data.iloc[i : i + seq_length, 0].values
        X.append(wl_seq)
        y.append(data.iloc[i + seq_length, 0])
    return np.array(X), np.array(y)

def split_data(data, test_size=0.2, seq_length=3):
    X, y = create_sequences(data, seq_length=seq_length)
    train_size = int((1 - test_size) * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    return X_train, y_train, X_test, y_test

##############################################################################
# 2. Definition of quantum circuits
##############################################################################

# -- 4 different devices:
dev_dru       = qml.device("default.qubit", wires=1)
dev_qrb_h_crx = qml.device("default.qubit", wires=2)
dev_qrb_param = qml.device("default.qubit", wires=2)
dev_qrb_multi = qml.device("default.qubit", wires=3)

# -- Simple DRU (uses 9 parameters: params[i][0..8])
@qml.qnode(dev_dru, interface="torch")
def quantum_circuit_dru(params, x):
    """
    DRU circuit: applies 3 blocks (RX, RY*x[j], RZ) per layer,
    for j = 0..2 (3 inputs).
    """
    for i in range(depth):
        # For x[0]
        qml.RX(params[i][0], wires=0)
        qml.RY(params[i][1] * x[0], wires=0)
        qml.RZ(params[i][2], wires=0)

        # For x[1]
        qml.RX(params[i][3], wires=0)
        qml.RY(params[i][4] * x[1], wires=0)
        qml.RZ(params[i][5], wires=0)

        # For x[2]
        qml.RX(params[i][6], wires=0)
        qml.RY(params[i][7] * x[2], wires=0)
        qml.RZ(params[i][8], wires=0)
    return qml.expval(qml.PauliZ(0))

# -- DRU + QRB_H_CRX (still uses 9 parameters, alpha set to π)
@qml.qnode(dev_qrb_h_crx, interface="torch")
def quantum_circuit_dru_qrb_h_crx(params, x, alpha=0.5):
    """
    Same DRU, then insert a small QRB_H_CRX block:
    - Hadamard(wires=1)
    - CRX(alpha * π, wires=[1,0])
    - Hadamard(wires=1)
    """
    for i in range(depth):
        # DRU gates
        qml.RX(params[i][0], wires=0)
        qml.RY(params[i][1] * x[0], wires=0)
        qml.RZ(params[i][2], wires=0)

        qml.RX(params[i][3], wires=0)
        qml.RY(params[i][4] * x[1], wires=0)
        qml.RZ(params[i][5], wires=0)

        qml.RX(params[i][6], wires=0)
        qml.RY(params[i][7] * x[2], wires=0)
        qml.RZ(params[i][8], wires=0)

        # QRB_H_CRX block
        qml.Hadamard(wires=1)
        qml.CRX(alpha * np.pi, wires=[1, 0])
        qml.Hadamard(wires=1)
    return qml.expval(qml.PauliZ(0))

# -- DRU + Parameterized QRB (uses 11 parameters)
@qml.qnode(dev_qrb_param, interface="torch")
def quantum_circuit_dru_qrb_param(params, x, alpha=0.5):
    """
    DRU (9 params) + 2 params for the parameterized block:
    qml.RX(alpha * params[i][9], wires=1)
    qml.CRX(alpha * params[i][10], wires=[1, 0])
    """
    for i in range(depth):
        # DRU gates
        qml.RX(params[i][0], wires=0)
        qml.RY(params[i][1] * x[0], wires=0)
        qml.RZ(params[i][2], wires=0)

        qml.RX(params[i][3], wires=0)
        qml.RY(params[i][4] * x[1], wires=0)
        qml.RZ(params[i][5], wires=0)

        qml.RX(params[i][6], wires=0)
        qml.RY(params[i][7] * x[2], wires=0)
        qml.RZ(params[i][8], wires=0)

        # Parameterized QRB
        qml.RX(alpha * params[i][9], wires=1)
        qml.CRX(alpha * params[i][10], wires=[1, 0])
    return qml.expval(qml.PauliZ(0))

# -- DRU + Multi-Qubit QRB (uses 11 parameters)
@qml.qnode(dev_qrb_multi, interface="torch")
def quantum_circuit_dru_qrb_multi(params, x):
    """
    DRU (9 params) + 2 params for interactions on 2 extra qubits:
    qml.CRX(params[i][9], wires=[2, 0])
    qml.CRY(params[i][10], wires=[1, 0])
    We assume 3 qubits: wire=0 for DRU, wires=1 and 2 for multi-qubit interactions.
    """
    for i in range(depth):
        # DRU gates
        qml.RX(params[i][0], wires=0)
        qml.RY(params[i][1] * x[0], wires=0)
        qml.RZ(params[i][2], wires=0)

        qml.RX(params[i][3], wires=0)
        qml.RY(params[i][4] * x[1], wires=0)
        qml.RZ(params[i][5], wires=0)

        qml.RX(params[i][6], wires=0)
        qml.RY(params[i][7] * x[2], wires=0)
        qml.RZ(params[i][8], wires=0)

        # Multi-qubit interactions
        qml.CRX(params[i][9], wires=[2, 0])
        qml.CRY(params[i][10], wires=[1, 0])
    return qml.expval(qml.PauliZ(0))

##############################################################################
# 3. Loss and accuracy functions
##############################################################################
def huber_loss(y_pred, y_true, delta=1.0):
    diff = torch.abs(y_pred - y_true)
    loss = torch.where(diff <= delta,
                       0.5 * diff**2,
                       delta * (diff - 0.5 * delta))
    return torch.mean(loss)

def prediction_accuracy(y_pred, y_true, tolerance=0.1):
    """
    Percentage of predictions within ±(tolerance * |y_true|).
    """
    within_tol = torch.abs(y_pred - y_true) <= (tolerance * torch.abs(y_true))
    return torch.mean(within_tol.float()) * 100

##############################################################################
# 4. Data preparation
##############################################################################
data = preprocess_river_data("river_level.csv")
X_train_np, y_train_np, X_test_np, y_test_np = split_data(data, seq_length=seq_length)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float64)
y_train = torch.tensor(y_train_np, dtype=torch.float64)
X_test  = torch.tensor(X_test_np,  dtype=torch.float64)
y_test  = torch.tensor(y_test_np,  dtype=torch.float64)

##############################################################################
# 5. Generic training loop for comparing all 4 circuits
##############################################################################
def train_model(model, params_init, X_train, y_train, nb_epoch=nb_epoch, lr=lr):
    """
    Train a given model (circuit) on (X_train, y_train),
    returns loss history, accura
