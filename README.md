# Quantum Flood Forecasting

This repository contains a Jupyter notebook implementing:
1. Classical LSTM baseline  
2. Quantum-enhanced QLSTM with QRU  

on daily river level and flow datasets for four Waipa River stations.

## Structure

- `data/` : raw CSV files (daily `wlvalue` and `flvalue`)  
- `notebooks/` : Jupyter notebook with exploration, modeling, and results  
- `README.md` : this file  

## Usage

```bash
# 1. Clone repo
git clone git@github.com:<username>/quantum-flood-forecasting.git
cd quantum-flood-forecasting

# 2. Install requirements
pip install -r requirements.txt

# 3. Launch notebook
jupyter lab notebooks/QRU-LSTM_Flood_Prediction.ipynb
