# CMAPSS Turbofan Engine Predictive Maintenance

A machine learning project to predict the Remaining Useful Life (RUL) of turbofan jet engines using NASA's CMAPSS dataset. 
Built as a portfolio project demonstrating predictive maintenance methodology.

## Background

Predictive maintenance aims to predict equipment failure before it occurs using sensor data, enabling maintenance to be scheduled only when needed. 
This is more efficient than time-based preventive maintenance and safer than reactive maintenance.

This project predicts the number of flight cycles remaining before an engine fails using time-series sensor readings.
A maintenance engineer can use this output to schedule intervention at the right time, avoiding both unnecessary downtime and unexpected failures.

## Dataset

### NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS)

The dataset simulates turbofan engine degradation under different operating conditions and fault modes. 
Each engine runs from a healthy state until failure with 3 operational setting variables and 21 sensor readings recorded per cycle.

This project currently uses the FD001 subset which contains a single fault mode, single operating condition as a clean baseline 
before extending to the more complex FD003/FD004 subsets.

**Download:** https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Place the extracted files in a `CMAPSSData/` folder in the project root.

## Project Structure

```
CMAPSS-Predictive-Maintenance/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb
|   ├── 02_baseline_model.ipynb
|   └── 03_lstm_model.ipynb
├── data/
├── models/
├── results/
└── CMAPSSData/
```

## Methodology

### Exploratory Data Analysis
- Loaded and inspected FD001 training data
- Calculated RUL directly from cycle data
- Removed 7 zero-variance sensors through variance filtering
- Analyzed degradation trends across single and multiple engines
- Performed correlation analysis, removing sensor_14 (r=0.96 with sensor_9)
- Applied piecewise linear RUL cap at 125 cycles to focus model training on the relevant degradation zone

### XGBoost Baseline
- Loaded cleaned data from the initial exploratory data analysis (EDA)
- Engineered time-based features (rolling statistics, lag features)
- Prepared train/test split
- Trained XGBoost regressor
- Evaluated performance and analyzed results
- Calculated score based on NASA scoring function

### LSTM Model
- Loaded cleaned data from EDA
- Prepared 3D sequence arrays using a sliding window approach
- Normalized features
- Built and trained LSTM
- Evaluated against XGBoost baseline using identical metrics
