import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Turbofan Engine RUL Predictor",
    page_icon="✈️",
    layout="wide"
)

# LSTM Model Definition 
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size_1=64, hidden_size_2=32, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size_1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_size_1, hidden_size=hidden_size_2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size_2, 1)

        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)
        out2, _ = self.lstm2(out1)
        out2 = self.dropout2(out2)
        final_hidden = out2[:, -1, :]
        output = self.fc(final_hidden).squeeze(-1)
        return output

# Data and Model Loading
# @st.cache_data caches the result so these only run once rather than reloading every time the user clicks something

@st.cache_data
def load_data():
    """Load and prepare training and test data."""
    # Column names matching the raw NASA files
    column_names = [
        'engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
        'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
        'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
        'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
        'sensor_21'
    ]

    # Sensors kept after EDA variance and correlation analysis
    sensors_to_drop = [
        'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
        'sensor_14', 'sensor_16', 'sensor_18', 'sensor_19'
    ]

    # Load training data
    df_train = pd.read_csv(
        '../CMAPSSData/train_FD001.txt',
        sep=r'\s+', header=None, names=column_names
    )
    df_train = df_train.drop(columns=sensors_to_drop)
    df_train = df_train.sort_values(['engine_id', 'cycle']).reset_index(drop=True)

    # Calculate RUL
    max_cycles = df_train.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    df_train = df_train.merge(max_cycles, on='engine_id')
    df_train['RUL'] = df_train['max_cycle'] - df_train['cycle']
    df_train = df_train.drop(columns=['max_cycle'])
    df_train['RUL'] = df_train['RUL'].clip(upper=125)

    # Load test data
    df_test = pd.read_csv(
        '../CMAPSSData/test_FD001.txt',
        sep=r'\s+', header=None, names=column_names
    )
    df_test = df_test.drop(columns=sensors_to_drop)
    df_test = df_test.sort_values(['engine_id', 'cycle']).reset_index(drop=True)

    # Load true RUL labels
    df_rul = pd.read_csv(
        '../CMAPSSData/RUL_FD001.txt',
        header=None, names=['RUL']
    )
    df_rul['engine_id'] = df_rul.index + 1
    df_rul['RUL'] = df_rul['RUL'].clip(upper=125)

    return df_train, df_test, df_rul

@st.cache_resource
def load_models():
    """Load trained models and supporting files."""
    # XGBoost model
    xgb_model = joblib.load('../models/xgboost_baseline.pkl')

    # LSTM model
    device = torch.device('cpu')  # use CPU for deployment
    checkpoint = torch.load('../models/lstm_model.pt', map_location=device)
    config = checkpoint['model_config']
    lstm_model = LSTMModel(
        input_size=config['input_size'],
        hidden_size_1=config['hidden_size_1'],
        hidden_size_2=config['hidden_size_2'],
        dropout=config['dropout']
    )
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()

    # Scaler
    scaler = joblib.load('../models/lstm_scaler.pkl')

    # Conformal prediction parameters
    with open('../models/conformal_params.json') as f:
        conformal_params = json.load(f)

    return xgb_model, lstm_model, scaler, conformal_params, device

# Load everything
df_train, df_test, df_rul = load_data()
xgb_model, lstm_model, scaler, conformal_params, device = load_models()

# Feature columns used by both models
feature_cols = [
    'setting_1', 'setting_2', 'setting_3',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
    'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_15',
    'sensor_17', 'sensor_20', 'sensor_21'
]

SEQUENCE_LENGTH = 50
CAP = 125

tab_overview, tab_data, tab_prediction = st.tabs([
    "📊 Project Overview",
    "🔍 Data Explorer", 
    "🎯 RUL Prediction"
])

with tab_overview:

    # Header
    st.title("✈️ Turbofan Engine Remaining Useful Life Predictor")
    st.markdown("""
    Predictive maintenance dashboard using the NASA CMAPSS dataset. This tool predicts how many flight cycles remain before a turbofan engine
    requires maintenance, using sensor readings from the engine's operational history.

    Models: XGBoost baseline and LSTM sequence model with conformal prediction intervals
    """)
    st.divider()

    # Project Overview 
    st.header("Project Overview")

    # Load model comparison metrics from saved file
    metrics_df = pd.read_csv('../results/model_comparison.csv')
    xgb_row = metrics_df[metrics_df['model'] == 'XGBoost Tuned'].iloc[0]
    lstm_row = metrics_df[metrics_df['model'] == 'LSTM'].iloc[0]

    col_engine, col_sensor, col_rmse, col_nasa = st.columns(4)

    with col_engine:
        st.metric(
            label="Training Engines",
            value=str(df_train['engine_id'].nunique()),
            help="Number of engines in the training dataset"
        )

    with col_sensor:
        st.metric(
            label="Sensors Monitored",
            value=str(len([col for col in feature_cols if col.startswith('sensor')])),
            help="Sensors retained after variance and correlation analysis"
        )

    with col_rmse:
        rmse_change = lstm_row['rmse'] - xgb_row['rmse']
        st.metric(
            label="LSTM RMSE",
            value=f"{lstm_row['rmse']:.2f} cycles",
            delta=f"{rmse_change:+.2f} vs XGBoost",
            delta_color="inverse",
            help="Root mean squared error on the NASA test set. Lower is better."
        )

    with col_nasa:
        nasa_change = lstm_row['nasa_score'] - xgb_row['nasa_score']
        st.metric(
            label="NASA Score",
            value=f"{lstm_row['nasa_score']:.2f}",
            delta=f"{nasa_change:+.2f} vs XGBoost",
            delta_color="inverse",
            help="Asymmetric score penalizing late predictions. Lower is better."
        )

    st.divider()

    # Model comparison table
    st.subheader("Model Performance Comparison")

    comparison_data = {
        'Metric': ['RMSE (cycles)', 'MAE (cycles)', 'NASA Score',
                'RMSE % of mean RUL', 'MAE % of mean RUL',
                'Late Predictions', 'Early Predictions'],
        'XGBoost': [
            f"{xgb_row['rmse']:.2f}",
            f"{xgb_row['mae']:.2f}",
            f"{xgb_row['nasa_score']:.2f}",
            f"{xgb_row['rmse_pct_mean_rul']:.1f}%",
            f"{xgb_row['mae_pct_mean_rul']:.1f}%",
            f"{int(xgb_row['late_predictions'])}/100",
            f"{int(xgb_row['early_predictions'])}/100"
        ],
        'LSTM': [
            f"{lstm_row['rmse']:.2f}",
            f"{lstm_row['mae']:.2f}",
            f"{lstm_row['nasa_score']:.2f}",
            f"{lstm_row['rmse_pct_mean_rul']:.1f}%",
            f"{lstm_row['mae_pct_mean_rul']:.1f}%",
            f"{int(lstm_row['late_predictions'])}/100",
            f"{int(lstm_row['early_predictions'])}/100"
        ],
        'Change': [
            f"{lstm_row['rmse']-xgb_row['rmse']:+.2f}",
            f"{lstm_row['mae']-xgb_row['mae']:+.2f}",
            f"{lstm_row['nasa_score']-xgb_row['nasa_score']:+.2f}",
            f"{lstm_row['rmse_pct_mean_rul']-xgb_row['rmse_pct_mean_rul']:+.1f}%",
            f"{lstm_row['mae_pct_mean_rul']-xgb_row['mae_pct_mean_rul']:+.1f}%",
            f"{int(lstm_row['late_predictions'])-int(xgb_row['late_predictions']):+d}",
            f"{int(lstm_row['early_predictions'])-int(xgb_row['early_predictions']):+d}"
        ]
    }

    st.dataframe(
        pd.DataFrame(comparison_data),
        hide_index=True,
        width='stretch'
    )

    with st.expander("What do these metrics mean?"):
        st.markdown("""
        - **RMSE** - Average prediction error in flight cycles, penalizing large errors more heavily than small ones. Lower is better.
        - **MAE** - Average absolute prediction error in flight cycles. Lower is better.
        - **NASA Score** - Asymmetric scoring function from the original NASA research paper. 
                    Penalizes predicting too much remaining life (dangerous) more heavily than predicting too little (wasteful but safe).
                    Lower is better.
        - **Late predictions** - Model predicted more remaining life than actual. This is the operationally dangerous direction.
        - **Early predictions** - Model predicted less remaining life than actual. This is the operationally safe direction but may cause unnecessary maintenance.
        """)

    st.divider()

with tab_data:
    # Data Explorer
    st.header("Engine Data Explorer")
    st.markdown("""
    Select any training engine to explore its sensor readings over its full lifetime. Each engine runs from a healthy state until failure. 
    Observe how sensors degrade as the engine approaches the end of its useful life.
    """)

    # Training engine selector
    engine_ids = sorted(df_train['engine_id'].unique())
    selected_engine = st.selectbox(
        "Select training engine to explore",
        options=engine_ids,
        format_func=lambda x: f"Engine {x}"
    )

    # Get selected engine data
    engine_data = df_train[df_train['engine_id'] == selected_engine].sort_values('cycle')
    total_cycles = engine_data['cycle'].max()
    final_rul = engine_data['RUL'].min()

    # Engine summary metrics
    col_lifetime, col_degradation, col_percentile = st.columns(3)
    with col_lifetime:
        st.metric("Total Lifetime", f"{total_cycles} cycles")
    with col_degradation:
        # Cycles remaining when degradation becomes visible
        # Using the RUL cap as reference. Below 125 is the degradation zone
        degradation_start = engine_data[engine_data['RUL'] < 125]['cycle'].min()
        if pd.notna(degradation_start):
            st.metric("Degradation Zone Start",
                f"Cycle {degradation_start:.0f}",
                help="Cycle where RUL dropped below the 125-cycle cap, indicating the start of noticeable degradation in the engine's performance. " \
                "This is a key point in the engine's lifecycle, as it marks the transition from healthy operation to the phase where maintenance " \
                "planning should begin.")
    with col_percentile:
        # Percentile rank among all engines
        all_lifetimes = df_train.groupby('engine_id')['cycle'].max()
        percentile = (all_lifetimes < total_cycles).mean() * 100
        st.metric("Lifetime Percentile",
                f"{percentile:.0f}th",
                help="This engine lived longer than X% of training engines")

    # Sensor selector for degradation plot
    sensor_cols = [col for col in feature_cols if col.startswith('sensor')]
    selected_sensors = st.multiselect(
        "Select sensors to plot",
        options=sensor_cols,
        default=['sensor_2', 'sensor_3', 'sensor_4', 'sensor_11', 'sensor_17'],
        help="Select one or more sensors to visualize their degradation over time"
    )

    st.caption(
    "Default sensors selected based on feature importance analysis. Add or remove sensors to explore the full dataset."
    )

    if selected_sensors:
        fig, axes = plt.subplots(
            len(selected_sensors), 1,
            figsize=(12, 3 * len(selected_sensors)),
            sharex=True
        )

        # Handle single sensor case where axes isn't a list
        if len(selected_sensors) == 1:
            axes = [axes]

        for ax, sensor in zip(axes, selected_sensors):
            ax.plot(
                engine_data['cycle'],
                engine_data[sensor],
                color='steelblue',
                linewidth=1.2
            )
            ax.axvline(
                x=total_cycles,
                color='red',
                linestyle='--',
                alpha=0.7,
                label='Failure point'
            )

            # Degradation zone start
            degradation_start = engine_data[engine_data['RUL'] < 125]['cycle'].min()
            if pd.notna(degradation_start):
                ax.axvline(
                    x=degradation_start,
                    color='orange',
                    linestyle='--',
                    alpha=0.7,
                    label='Degradation zone start (RUL < 125)'
                )
                # Shade the degradation zone
                ax.axvspan(
                    degradation_start,
                    total_cycles,
                    alpha=0.05,
                    color='orange'
                )

            ax.set_ylabel(sensor)
            ax.set_title(f'{sensor} - Engine {selected_engine}')
            ax.legend(loc='upper left')

        axes[-1].set_xlabel('Cycle (Flight Number)')
        plt.suptitle(
            f'Engine {selected_engine} Sensor Degradation '
            f'({total_cycles} cycles until failure)',
            fontsize=13,
            y=1.01
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.info("Select at least one sensor to display the degradation plot.")

    # RUL over time plot
    st.subheader(f"Engine {selected_engine} - Remaining Useful Life Over Time")

    fig2, ax = plt.subplots(figsize=(12, 3))
    ax.plot(engine_data['cycle'], engine_data['RUL'],
            color='steelblue', linewidth=1.5)
    ax.fill_between(engine_data['cycle'], engine_data['RUL'],
                    alpha=0.2, color='steelblue')
    ax.axhline(y=30, color='red', linestyle='--',
            alpha=0.7, label='Critical threshold (30 cycles)')
    ax.axhline(y=60, color='orange', linestyle='--',
            alpha=0.7, label='Warning threshold (60 cycles)')
    ax.set_xlabel('Cycle (Flight Number)')
    ax.set_ylabel('Remaining Useful Life (cycles)')
    ax.set_title(f'Engine {selected_engine} - RUL Countdown to Failure')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.divider()

with tab_prediction:

    # RUL Prediction
    st.header("RUL Prediction")
    st.markdown("""
    Select a test engine to generate a Remaining Useful Life prediction. The true RUL is shown for comparison. 
    In a real deployment this would be unknown and the model's prediction would guide maintenance scheduling.
    """)

    col_engine, col_coverage = st.columns([2, 1])

    # Engine selector
    with col_engine:
        test_engine_ids = sorted(df_test['engine_id'].unique())
        selected_test_engine = st.selectbox(
            "Select Test Engine",
            options=test_engine_ids,
            format_func=lambda x: f"Engine {x}"
        )

    with col_coverage:
        # Coverage level selector
        coverage_level = st.selectbox(
            "Confidence Interval Level",
            options=[0.80, 0.90, 0.95],
            index=1,
            format_func=lambda x: f"{int(x*100)}%"
        )

    # Adjustable maintenance thresholds
    st.subheader("Maintenance Thresholds")
    st.markdown(
        "Set the cycle thresholds that trigger warning and critical alerts. "
        "These reflect operational scheduling requirements."
    )

    col_warn, col_crit = st.columns(2)

    # Warning threshold setting
    with col_warn:
        warning_threshold = st.slider(
            "Warning threshold (cycles)",
            min_value=20,
            max_value=100,
            value=60,
            step=5,
            help="Engines below this RUL are flagged for increased monitoring"
        )

    # Critical threshold setting
    with col_crit:
        critical_threshold = st.slider(
            "Critical threshold (cycles)",
            min_value=10,
            max_value=warning_threshold - 5,
            value=30,
            step=5,
            help="Engines below this RUL require immediate maintenance scheduling"
        )

    st.divider()

    # Preprocessing helpers
    def get_xgb_features(engine_data, feature_cols, window=30):
        """Recreate XGBoost rolling features for a single engine."""
        df = engine_data[feature_cols].copy()
        
        # Only apply rolling features to sensor columns
        # Settings columns were not rolled in the XGBoost notebook
        sensor_only_cols = [col for col in feature_cols if col.startswith('sensor')]
        
        for sensor in sensor_only_cols:
            df[f'{sensor}_mean_{window}'] = df[sensor].rolling(window, min_periods=window).mean()
            df[f'{sensor}_std_{window}']  = df[sensor].rolling(window, min_periods=window).std()
            df[f'{sensor}_lag_{window}']  = df[sensor].shift(window)
        
        df = df.dropna()
        
        if len(df) == 0:
            return None
            
        return df.iloc[[-1]]  # last row only

    def get_lstm_sequence(engine_data, feature_cols, scaler, sequence_length=50):
        """Prepare normalized sequence for LSTM inference."""
        features = engine_data[feature_cols].values
        features_scaled = scaler.transform(features)
        if len(features_scaled) >= sequence_length:
            seq = features_scaled[-sequence_length:]
        else:
            pad_length = sequence_length - len(features_scaled)
            padding = np.repeat(features_scaled[0:1], pad_length, axis=0)
            seq = np.vstack([padding, features_scaled])
        return torch.FloatTensor(seq).unsqueeze(0)  # add batch dimension

    def get_risk_level(rul_lower_bound, warning_threshold, critical_threshold):
        """
        Determine risk level based on the lower bound of the confidence interval.
        Using the lower bound rather than the point estimate means risk classification accounts for prediction uncertainty.
        """
        if rul_lower_bound <= critical_threshold:
            return "🔴 CRITICAL", "#ffcccc", (
                "The lower bound of the confidence interval falls within the critical zone. Immediate maintenance scheduling recommended."
            )
        elif rul_lower_bound <= warning_threshold:
            return "🟡 WARNING", "#fff3cc", (
                "The lower bound of the confidence interval falls within the warning zone. Increased monitoring recommended."
            )
        else:
            return "🟢 SAFE", "#ccffcc", (
                "Engine operating within normal parameters across the full confidence interval."
            )

    # Generate predictions 
    engine_test_data = df_test[
        df_test['engine_id'] == selected_test_engine
    ].sort_values('cycle')

    true_rul = df_rul[df_rul['engine_id'] == selected_test_engine]['RUL'].values[0]

    # XGBoost prediction
    xgb_features = get_xgb_features(engine_test_data, feature_cols)
    if len(xgb_features) > 0:
        xgb_pred = float(xgb_model.predict(xgb_features)[0])
        xgb_pred = np.clip(xgb_pred, 0, CAP)
    else:
        xgb_pred = None

    # LSTM prediction
    lstm_sequence = get_lstm_sequence(engine_test_data, feature_cols, scaler)
    with torch.no_grad():
        lstm_pred = float(lstm_model(lstm_sequence).item())
    lstm_pred = np.clip(lstm_pred, 0, CAP)

    # Conformal prediction interval
    q_hat = conformal_params['quantiles'][str(coverage_level)]
    interval_lower = max(0, lstm_pred - q_hat)
    interval_upper = min(CAP, lstm_pred + q_hat)

    # Risk assessment
    risk_label, risk_color, risk_message = get_risk_level(
        interval_lower, warning_threshold, critical_threshold
    )

    # Display results 
    st.subheader(f"Engine {selected_test_engine} - Prediction Results")

    # Risk indicator
    st.markdown(
        f"<h2 style='color:{risk_color}; text-align:center'>{risk_label}</h2>"
        f"<p style='text-align:center'>{risk_message}</p>",
        unsafe_allow_html=True
    )

    st.divider()

    # Prediction metrics
    col_rul, col_xgboost, col_lstm, col_confidence = st.columns(4)

    with col_rul:
        st.metric(
            label="True RUL",
            value=f"{true_rul:.0f} cycles",
            help="Actual remaining useful life - unknown in real deployment"
        )

    with col_xgboost:
        if xgb_pred is not None:
            xgb_error = xgb_pred - true_rul
            st.metric(
                label="XGBoost Prediction",
                value=f"{xgb_pred:.0f} cycles" if xgb_pred is not None else "N/A",
                delta=f"{xgb_error:+.0f} cycles error" if xgb_error is not None else None,
                delta_color="inverse"
            )
        else:
            st.metric(
                label="XGBoost Prediction",
                value="N/A",
                help="Insufficient data to compute rolling features for XGBoost prediction (needs 30+ cycles)"
            )

    with col_lstm:
        lstm_error = lstm_pred - true_rul
        st.metric(
            label="LSTM Prediction",
            value=f"{lstm_pred:.0f} cycles",
            delta=f"{lstm_error:+.0f} cycles error",
            delta_color="inverse"
        )

    with col_confidence:
        st.metric(
            label=f"{int(coverage_level*100)}% Confidence Interval",
            value=f"{interval_lower:.0f} – {interval_upper:.0f} cycles",
            help=f"True RUL falls within this range {int(coverage_level*100)}% of the time"
        )

    st.divider()

    # Prediction visualization
    fig, ax = plt.subplots(figsize=(10, 4))

    # Confidence interval bar
    ax.barh(
        ['LSTM Prediction'],
        [interval_upper - interval_lower],
        left=[interval_lower],
        color='steelblue',
        alpha=0.3,
        height=0.4,
        label=f'{int(coverage_level*100)}% confidence interval'
    )

    # Model predictions
    ax.scatter(
        [lstm_pred], ['LSTM Prediction'],
        color='steelblue', s=150, zorder=5, label='LSTM prediction'
    )
    if xgb_pred is not None:
        ax.scatter(
            [xgb_pred], ['LSTM Prediction'],
            color='navy', s=100, marker='D',
            zorder=5, label='XGBoost prediction', alpha=0.7
        )

    # True RUL
    ax.axvline(
        x=true_rul, color='red', linestyle='--',
        linewidth=2, label=f'True RUL ({true_rul:.0f} cycles)'
    )

    # Threshold zones
    ax.axvspan(0, critical_threshold, alpha=0.08, color='red', label='Critical zone')
    ax.axvspan(critical_threshold, warning_threshold, alpha=0.08,
            color='orange', label='Warning zone')

    ax.set_xlim(0, CAP + 5)
    ax.set_xlabel('Remaining Useful Life (cycles)')
    ax.set_title(f'Engine {selected_test_engine} - RUL Prediction with '
                f'{int(coverage_level*100)}% Confidence Interval')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Sensor history for selected test engine
    st.subheader(f"Engine {selected_test_engine} - Observed Sensor History")
    st.markdown(
        "Sensor readings observed before the prediction cutoff point. "
        "The model uses the most recent 50 cycles of this history to generate its prediction."
    )

    selected_test_sensors = st.multiselect(
        "Select sensors to display",
        options=sensor_cols,
        default=['sensor_2', 'sensor_4', 'sensor_11'],
        key="test_sensors"
    )

    if selected_test_sensors:
        fig3, axes = plt.subplots(
            len(selected_test_sensors), 1,
            figsize=(12, 3 * len(selected_test_sensors)),
            sharex=True
        )

        if len(selected_test_sensors) == 1:
            axes = [axes]

        for ax, sensor in zip(axes, selected_test_sensors):
            ax.plot(
                engine_test_data['cycle'],
                engine_test_data[sensor],
                color='steelblue', linewidth=1.2
            )
            # Highlight last 50 cycles used for prediction
            if len(engine_test_data) >= 50:
                last_50_start = engine_test_data['cycle'].iloc[-50]
                ax.axvspan(
                    last_50_start,
                    engine_test_data['cycle'].max(),
                    alpha=0.15, color='orange',
                    label='Last 50 cycles (used for prediction)'
                )
                ax.legend(loc='upper left', fontsize=8)
            ax.set_ylabel(sensor)
            ax.set_title(f'{sensor} - Test Engine {selected_test_engine}')

        axes[-1].set_xlabel('Cycle (Flight Number)')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    else:
        st.info("Select at least one sensor to display the history plot.")

st.divider()
st.caption(
    "Built using the NASA CMAPSS Aircraft Engine Simulator dataset."
)