# Quantum River Forecasting and Insurance Prototype

This repository contains Jupyter notebooks exploring Quantum Re-Uploading Units (QRU) and hybrid quantum-classical models (QLSTM, QAOA) for hydrological forecasting and climate-risk insurance applications.

## 📂 Repository Structure

```
├── README.md
├── requirements.txt
├── simple_QRU_protype.ipynb
├── First_try_not_advantageaous_yet_18.05.ipynb
├── LSTM_QRU_parallele.ipynb
└── True_QLSTM_Schedule_sampl.ipynb
```

## 🚀 Installation

1. Clone the repo:  
   ```
   git clone https://github.com/YourOrg/quantum-river-forecasting.git
   cd quantum-river-forecasting
   ```
2. Create a Python 3.8+ virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```
3. Launch Jupyter Lab / Notebook:
```
    jupyter lab
```
## 📓 Notebooks
### 1. simple_QRU_protype.ipynb

Title: Quantum Data Re-Uploading Unit (QRU) for River Level Prediction
Description:
Implements a 1-qubit QRU to forecast 24–48 h ahead river‐level and rainfall exceedance using sliding-window regression. Benchmarks training accuracy and visualizes time-series predictions.

### 2. First_try_not_advantageaous_yet_18.05.ipynb

Title: 🌊 Hydrological Dataset Exploration
Description:
Loads and pre-processes rainfall & river‐level data (MetService, NIWA/LINZ), visualizes correlations and seasonal patterns, and runs an initial QRU prototype—highlighting areas for model improvement.

### 3. LSTM_QRU_parallele.ipynb

Description:
Compares classical LSTM and quantum QRU models in parallel on the same hydrological dataset. Evaluates forecasting performance, learning curves, and trade-offs in expressivity vs. trainability.

### 4. True_QLSTM_Schedule_sampl.ipynb

Description:
Implements a Quantum-LSTM (QLSTM) architecture with custom scheduling and sampling strategies for streaming data. Demonstrates hybrid circuit design and performance on sequence prediction.

##🔧 Requirements

    Python ≥ 3.8

    PennyLane

    PyTorch

    NumPy, pandas, scikit-learn

    matplotlib, seaborn

    (Optional) D-Wave Ocean SDK for QAOA experiments

Install with:
```
pip install pennylane torch numpy pandas scikit-learn matplotlib seaborn
# For D-Wave:
pip install dwave-ocean-sdk
```
## 📖 Usage

    Open the notebook you wish to explore.

    Run cells sequentially, ensuring your virtual environment is active.

    Modify hyperparameters (depth, window_size, learning_rate) at the top of each notebook.

    For quantum hardware experiments, configure API keys (e.g., D-Wave, IBM Q) in ~/.env or as environment variables.

## 🔗 References

    Pérez-Salinas et al., “Data Re-Uploading for a Universal Quantum Classifier,” Quantum, 2020.

    Barthe & Pérez-Salinas, “Gradients and Frequency Profiles of Quantum Re-Uploading Models,” arXiv:2408.XXXX, 2024.

    Schuld et al., “Circuit-centric Quantum Classifiers,” Physical Review A, 2020.

© 2025 Team DelphiQ
