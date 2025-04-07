import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Red Spider Mite Early Warning System",
    page_icon="🕷️",
    layout="wide"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stDateInput>div>input {
        border-radius: 8px;
        padding: 8px;
        border: 1px solid #ccc;
    }
    .stSlider>div>div>div {
        background-color: #4CAF50;
    }
    .stNumberInput>div>input {
        border-radius: 8px;
        padding: 8px;
        border: 1px solid #ccc;
    }
    .metric-box {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Configuration ---
ARTIFACTS_DIR = 'dashboard_artifacts'
HISTORY_LENGTH = 30
PRIMARY_PHENODE_URL = "https://phenode-link.com/"
FALLBACK_PHENODE_URL = "https://gist.githubusercontent.com/SiriChandanaGarimella/e478eee96f0391b1cb5e9552e57ad994/raw/f827adfd58c1f7afa6b3f9efd0254c904c9141a3/data.json"
PHENODE_USERNAME = "PHENODE_LINK_USERNAME"
PHENODE_PASSWORD = "PHENODE_LINK_PASSWORD"

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    """Loads the model, scaler, encoder, features, and initial history."""
    try:
        model = joblib.load(os.path.join(ARTIFACTS_DIR, 'xgb_model.joblib'))
        scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))
        label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, 'label_encoder.joblib'))
        features_list = joblib.load(os.path.join(ARTIFACTS_DIR, 'features.joblib'))
        try:
            history_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'recent_history.csv'), index_col=0, parse_dates=True)
            if not isinstance(history_df.index, pd.DatetimeIndex):
                history_df.index = pd.to_datetime(history_df.index)
            history_df = history_df[~history_df.index.duplicated(keep='last')]
        except FileNotFoundError:
            st.warning("History file not found. Creating minimal history.")
            required_base_features = ['temperature', 'humidity', 'rainfall', 'N', 'P', 'K', 'ph', 'label']
            default_data = {
                'temperature': [25.0], 'humidity': [60.0], 'rainfall': [0.0],
                'N': [75], 'P': [45], 'K': [42], 'ph': [7.0], 'label': ['unknown'],
                'Red_spider_mite_category': ['Low']
            }
            for col in required_base_features:
                if col not in default_data:
                    default_data[col] = [0]
            history_df = pd.DataFrame(default_data, index=[pd.to_datetime(datetime.now().date() - timedelta(days=1))])
            history_df.index.name = 'timestamp'
        history_df = history_df.iloc[-HISTORY_LENGTH:]
        return model, scaler, label_encoder, features_list, history_df
    except FileNotFoundError as e:
        st.error(f"Error loading critical artifacts: {e}. Cannot proceed.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading artifacts: {e}")
        st.stop()

model, scaler, label_encoder, FEATURES, initial_history = load_artifacts()
class_names = list(label_encoder.classes_)

# --- Data Loading Functions ---
def fetch_phenode_data(start_date, end_date):
    """Fetches real-time data from Phenode URL, trying primary URL first, then falling back to Gist."""
    try:
        # Convert start_date and end_date to datetime64[ns, UTC] for comparison
        start_date = pd.to_datetime(start_date).tz_localize('UTC')
        end_date = pd.to_datetime(end_date).tz_localize('UTC')

        # Try the primary Phenode URL with authentication
        try:
            response = requests.get(PRIMARY_PHENODE_URL, auth=(PHENODE_USERNAME, PHENODE_PASSWORD))
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                df['measurement_time'] = pd.to_datetime(df['measurement_time'])
                df['measurement_time'] = df['measurement_time'].dt.tz_convert('UTC')
                df = df[(df['measurement_time'] >= start_date) & (df['measurement_time'] <= end_date)]
                return df
            else:
                st.warning(f"Failed to fetch from primary Phenode URL: HTTP {response.status_code}. Trying fallback URL.")
        except Exception as e:
            st.warning(f"Error fetching from primary Phenode URL: {e}. Trying fallback URL.")

        # If primary URL fails, try the fallback Gist URL (no authentication needed)
        response = requests.get(FALLBACK_PHENODE_URL)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['measurement_time'] = pd.to_datetime(df['measurement_time'])
            df['measurement_time'] = df['measurement_time'].dt.tz_convert('UTC')
            df = df[(df['measurement_time'] >= start_date) & (df['measurement_time'] <= end_date)]
            return df
        else:
            st.error(f"Failed to fetch from fallback Phenode URL: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching Phenode data: {e}")
        return None

def load_historical_data():
    """Loads historical data from recent_history.csv or creates synthetic data if not available."""
    try:
        data = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'recent_history.csv'))
        if 'timestamp' not in data.columns:
            data['timestamp'] = pd.date_range(start='2024-01-01', end='2025-04-02', freq='D')[:len(data)]
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        required_cols = ['timestamp', 'temperature', 'humidity', 'rainfall', 'N', 'P', 'K', 'red_spider_count']
        for col in required_cols:
            if col not in data.columns:
                data[col] = np.random.uniform(20, 35, len(data)) if col == 'temperature' else \
                            np.random.uniform(50, 90, len(data)) if col == 'humidity' else \
                            np.random.uniform(0, 200, len(data)) if col == 'rainfall' else \
                            np.random.uniform(10, 120, len(data)) if col in ['N', 'P', 'K'] else \
                            np.random.uniform(50, 200, len(data))
        if 'risk_level' not in data.columns:
            X = scaler.transform(data[FEATURES])
            probs = model.predict_proba(X)
            data['risk_level'] = label_encoder.inverse_transform(np.argmax(probs, axis=1))
            data['confidence'] = np.max(probs, axis=1)
        return data
    except FileNotFoundError:
        st.warning("Historical data file not found. Generating synthetic data.")
        timestamps = pd.date_range(start='2024-01-01', end='2025-04-02', freq='D')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': np.random.uniform(20, 35, len(timestamps)),
            'humidity': np.random.uniform(50, 90, len(timestamps)),
            'rainfall': np.random.uniform(0, 200, len(timestamps)),
            'N': np.random.uniform(10, 120, len(timestamps)),
            'P': np.random.uniform(10, 120, len(timestamps)),
            'K': np.random.uniform(10, 120, len(timestamps)),
            'red_spider_count': np.random.uniform(50, 200, len(timestamps))
        })
        X = scaler.transform(data[FEATURES])
        probs = model.predict_proba(X)
        data['risk_level'] = label_encoder.inverse_transform(np.argmax(probs, axis=1))
        data['confidence'] = np.max(probs, axis=1)
        return data

def process_data(df):
    """Processes raw data into a format suitable for the model."""
    if df.empty:
        return pd.DataFrame()
    df_pivot = df.pivot_table(index='measurement_time', columns='metric', values='value', aggfunc='first')
    df_pivot.rename(columns={
        'air_temp': 'temperature',
        'humidity': 'humidity',
        'fallen_rain_mm': 'rainfall'
    }, inplace=True)
    # Convert columns to numeric, coercing errors to NaN
    df_pivot['temperature'] = pd.to_numeric(df_pivot.get('temperature', 25.0), errors='coerce').fillna(25.0)
    df_pivot['humidity'] = pd.to_numeric(df_pivot.get('humidity', 60.0), errors='coerce').fillna(60.0)
    df_pivot['rainfall'] = pd.to_numeric(df_pivot.get('rainfall', 0.0), errors='coerce').fillna(0.0)
    # Add static columns
    df_pivot['N'], df_pivot['P'], df_pivot['K'], df_pivot['ph'] = 75, 45, 42, 7.0
    df_pivot['label'] = 'unknown'
    return df_pivot

# --- Feature Engineering ---
def calculate_features(df):
    """Calculates lags and rolling features for the last row."""
    df = df.sort_index()
    if df.index.duplicated().any():
        dup_dates = df.index[df.index.duplicated()].unique().strftime('%Y-%m-%d')
        st.error(f"Duplicate date index in input to calculate_features: {dup_dates}. Aborting.")
        return pd.DataFrame()

    latest_idx = df.index[-1]
    required_base = ['temperature', 'humidity', 'rainfall']
    if not all(col in df.columns for col in required_base):
        st.error(f"Missing one of {required_base} for feature calculation.")
        return pd.DataFrame()

    temp_df = df.copy()
    lags = [1, 3, 7]
    windows = [3, 7, 14]
    drivers = ['temperature', 'humidity', 'rainfall']
    for feature in drivers:
        for lag in lags:
            temp_df[f'{feature}_lag_{lag}'] = temp_df[feature].shift(lag)
    for feature in drivers:
        for window in windows:
            roll = temp_df[feature].rolling(window=window, min_periods=1)
            temp_df[f'{feature}_roll_mean_{window}'] = roll.mean()
            if feature == 'temperature':
                temp_df[f'{feature}_roll_max_{window}'], temp_df[f'{feature}_roll_min_{window}'] = roll.max(), roll.min()
            if feature == 'humidity':
                temp_df[f'{feature}_roll_min_{window}'], temp_df[f'{feature}_roll_max_{window}'] = roll.min(), roll.max()
            if feature == 'rainfall':
                temp_df[f'{feature}_roll_sum_{window}'] = roll.sum()

    date_features = ['month', 'day_of_year', 'day_of_week', 'week_of_year']
    for feature in date_features:
        if feature not in temp_df.columns:
            if isinstance(temp_df.index, pd.DatetimeIndex):
                idx = temp_df.index
                if feature == 'month':
                    temp_df[feature] = idx.month
                elif feature == 'day_of_year':
                    temp_df[feature] = idx.dayofyear
                elif feature == 'day_of_week':
                    temp_df[feature] = idx.dayofweek
                elif feature == 'week_of_year':
                    temp_df[feature] = idx.isocalendar().week
            else:
                temp_df[feature] = 0

    missing_cols = [col for col in FEATURES if col not in temp_df.columns]
    if missing_cols:
        for col in missing_cols:
            temp_df[col] = 0

    try:
        last_row_features = temp_df.loc[[latest_idx]].reindex(columns=FEATURES)
        for col in last_row_features.columns[last_row_features.isnull().any()]:
            last_row_features[col] = last_row_features[col].fillna(0)
        if last_row_features.isnull().any().any():
            st.warning("NaNs persist after fillna(0). Check feature calculation logic.")
            last_row_features = last_row_features.fillna(0)
        return last_row_features
    except KeyError:
        st.error(f"KeyError on index '{latest_idx}' during reindex.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error finalizing features: {e}")
        return pd.DataFrame()

# --- Visualization & Other Helpers ---
def create_risk_gauge(current_risk, confidence):
    """Create a risk gauge visualization."""
    risk_map = {"Low": 1, "Medium": 2, "High": 3, "Severe": 4}
    risk_value = risk_map.get(current_risk, 0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={"text": f"Risk Level: {current_risk}"},
        gauge={
            "axis": {"range": [0, len(risk_map)]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 1], "color": "lightgreen"},
                {"range": [1, 2], "color": "gold"},
                {"range": [2, 3], "color": "salmon"},
                {"range": [3, 4], "color": "darkred"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": risk_value
            }
        }
    ))
    fig.add_annotation(x=0.5, y=0.1, text=f"Confidence: {confidence:.1%}", showarrow=False)
    return fig

def plot_environmental_factors(data):
    """Plot key environmental factors over time from history_df."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Temperature & Humidity", "Rainfall", "Soil Nutrients"),
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["temperature"], name="Temperature (°C)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["humidity"], name="Humidity (%)", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Bar(x=data["timestamp"], y=data["rainfall"], name="Rainfall (mm)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["N"], name="Nitrogen (N)"), row=3, col=1)
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["P"], name="Phosphorus (P)"), row=3, col=1)
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["K"], name="Potassium (K)"), row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Mite Favorable Temp", row=1, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Mite Favorable Humidity", row=1, col=1)
    fig.update_layout(height=700, title_text="Key Environmental Factors")
    return fig

def plot_risk_timeline(data):
    """Plot predicted risk level over time."""
    risk_map = {"Low": 1, "Medium": 2, "High": 3, "Severe": 4}
    data["risk_numeric"] = data["risk_level"].map(risk_map)
    fig = px.line(data, x="timestamp", y="risk_numeric", labels={"risk_numeric": "Risk Level", "timestamp": "Date"}, title="Risk Level Trend")
    fig.update_layout(yaxis=dict(tickmode="array", tickvals=[1, 2, 3, 4], ticktext=["Low", "Medium", "High", "Severe"]))
    fig.add_hrect(y0=0, y1=1.5, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=1.5, y1=2.5, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=2.5, y1=3.5, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=3.5, y1=4.5, fillcolor="darkred", opacity=0.1, line_width=0)
    return fig

def plot_mite_heatmap(data):
    """Create a simulated heatmap."""
    n_rows, n_cols = 5, 5
    grid_data = []
    latest_risk = data.iloc[-1]["risk_level"]
    risk_seed = {"Low": 0.2, "Medium": 0.5, "High": 0.8, "Severe": 1.0}.get(latest_risk, 0.5)
    max_distance = np.sqrt((n_rows / 2) ** 2 + (n_cols / 2) ** 2)
    for i in range(n_rows):
        for j in range(n_cols):
            distance_from_center = np.sqrt((i - n_rows / 2) ** 2 + (j - n_cols / 2) ** 2)
            normalized_distance = 1 - (distance_from_center / max_distance)
            risk_value = risk_seed * normalized_distance + np.random.normal(0, 0.1)
            risk_value = max(0, min(1, risk_value))
            grid_data.append({"row": i, "col": j, "risk": risk_value})
    grid_df = pd.DataFrame(grid_data)
    fig = px.imshow(
        grid_df.pivot(index="row", columns="col", values="risk"),
        color_continuous_scale="RdYlGn_r",
        labels=dict(color="Risk Index"),
        title="Field Risk Map (Latest)",
        aspect="equal"
    )
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(n_cols)), ticktext=[f"C{j+1}" for j in range(n_cols)]),
        yaxis=dict(tickmode="array", tickvals=list(range(n_rows)), ticktext=[f"R{i+1}" for i in range(n_rows)])
    )
    return fig

def create_action_recommendations(risk_level):
    """Generate action recommendations."""
    recommendations = {
        "Low": [
            "Continue regular monitoring (weekly inspections)",
            "Maintain beneficial predators (ladybugs, lacewings)",
            "Ensure proper irrigation to prevent water stress",
            "Document observations for future reference"
        ],
        "Medium": [
            "Increase monitoring frequency to every 3 days",
            "Release additional predatory mites as preventive control",
            "Check under leaf surfaces for early signs",
            "Prepare spray equipment and biological controls",
            "Avoid unnecessary nitrogen fertilization"
        ],
        "High": [
            "URGENT: Implement control measures immediately",
            "Apply approved miticides or biological controls",
            "Focus on hotspots identified in field risk map",
            "Consider targeted rather than whole-field application",
            "Monitor effectiveness after 2-3 days",
            "Alert neighboring farms of potential spread risk"
        ],
        "Severe": [
            "CRITICAL: Immediate action required",
            "Apply high-potency miticides immediately",
            "Isolate affected areas to prevent spread",
            "Deploy all available biological controls",
            "Monitor hourly and report to authorities",
            "Notify all nearby farms urgently"
        ]
    }
    return recommendations.get(risk_level, [])

def predict_risk(date, N, P, K, temperature, humidity):
    """Predict risk level for given inputs."""
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [6.5],
        'rainfall': [100]
    })
    for feature in FEATURES:
        if feature not in input_data.columns:
            input_data[feature] = 0
    X_scaled = scaler.transform(input_data[FEATURES])
    probs = model.predict_proba(X_scaled)
    risk_level = label_encoder.inverse_transform(np.argmax(probs, axis=1))[0]
    confidence = np.max(probs, axis=1)[0]
    return risk_level, confidence

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=True).encode('utf-8')

# --- Initialize Session State ---
def initialize_state(initial_history):
    """Initializes session state variables."""
    if 'history_df' not in st.session_state:
        st.session_state.history_df = initial_history.copy()
        if st.session_state.history_df.index.duplicated().any():
            st.warning("Duplicate dates found in initial history file, keeping last entry.")
            st.session_state.history_df = st.session_state.history_df[~st.session_state.history_df.index.duplicated(keep='last')]

    if 'data_source' not in st.session_state:
        st.session_state.data_source = "Synthetic"

    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = load_historical_data()

    if 'predicted_risk' not in st.session_state:
        st.session_state.predicted_risk = None
        st.session_state.predicted_confidence = None
        st.session_state.predicted_date = None

# --- Main App Function ---
def main():
    initialize_state(initial_history)
    st.title("🕷️ Red Spider Mite Early Warning System")
    st.markdown("#### Real-time monitoring and ML-based predictive alerts", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h2 style='color: #4CAF50;'>Data Source Selection</h2>", unsafe_allow_html=True)
        start_date = st.date_input("Start Date", value=pd.to_datetime("2024-07-03"), min_value=datetime(2024, 1, 1), max_value=datetime(2026, 12, 31))
        end_date = st.date_input("End Date", value=pd.to_datetime("2024-07-03"), min_value=datetime(2024, 1, 1), max_value=datetime(2026, 12, 31))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Synthetic Data", use_container_width=True):
                st.session_state.historical_data = load_historical_data()
                st.session_state.data_source = "Synthetic"
                st.success("Loaded synthetic data successfully.")
        with col2:
            if st.button("Load Phenode Data", use_container_width=True):
                phenode_data = fetch_phenode_data(start_date, end_date)
                if phenode_data is not None and not phenode_data.empty:
                    processed_phenode = process_data(phenode_data)
                    if not processed_phenode.empty:
                        processed_phenode['timestamp'] = processed_phenode.index
                        processed_phenode['red_spider_count'] = np.random.uniform(50, 200, len(processed_phenode))
                        X = scaler.transform(processed_phenode[FEATURES])
                        probs = model.predict_proba(X)
                        processed_phenode['risk_level'] = label_encoder.inverse_transform(np.argmax(probs, axis=1))
                        processed_phenode['confidence'] = np.max(probs, axis=1)
                        st.session_state.historical_data = processed_phenode
                        st.session_state.data_source = "Phenode"
                        st.success("Loaded Phenode data successfully.")
                    else:
                        st.warning("Processed Phenode data is empty. Switching to synthetic data.")
                        st.session_state.historical_data = load_historical_data()
                        st.session_state.data_source = "Synthetic"
                        st.info("Running on synthetic data due to issues with Phenode data.")
                else:
                    st.warning("Failed to fetch Phenode data. Switching to synthetic data.")
                    st.session_state.historical_data = load_historical_data()
                    st.session_state.data_source = "Synthetic"
                    st.info("Running on synthetic data due to issues with Phenode data.")

        st.markdown("---")
        st.markdown("<h2 style='color: #4CAF50;'>Predict Future Risk</h2>", unsafe_allow_html=True)
        future_date = st.date_input("Select Date for Prediction", value=datetime.today(), min_value=datetime(2024, 1, 1), max_value=datetime(2026, 12, 31))
        N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=50.0)
        P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=53.0)
        K = st.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=48.0)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=71.0)

        if st.button("Predict Risk", use_container_width=True):
            risk_level, confidence = predict_risk(future_date, N, P, K, temperature, humidity)
            st.session_state.predicted_risk = risk_level
            st.session_state.predicted_confidence = confidence
            st.session_state.predicted_date = future_date
            st.markdown(f"<p style='font-size: 16px;'><b>Predicted Risk for {future_date}:</b> {risk_level} (Confidence: {confidence:.1%})</p>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h3 style='color: #4CAF50;'>Current Conditions (Last Recorded)</h3>", unsafe_allow_html=True)
        latest_data = st.session_state.historical_data.iloc[-1]
        col1, col2 = st.columns(2)
        with col1:
            temp_value = float(latest_data.get('temperature', np.nan))
            st.markdown(f"<div class='metric-box'><b>Temp</b><br>{temp_value:.1f}°C</div>", unsafe_allow_html=True)
            rain_value = float(latest_data.get('rainfall', np.nan))
            st.markdown(f"<div class='metric-box'><b>Rain</b><br>{rain_value:.1f} mm</div>", unsafe_allow_html=True)
        with col2:
            humid_value = float(latest_data.get('humidity', np.nan))
            st.markdown(f"<div class='metric-box'><b>Humidity</b><br>{humid_value:.1f}%</div>", unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("Simulation Suggestions"):
            st.caption("Try these inputs for testing:")
            st.markdown("- **Heat/Dry:** Temp: 30-35°C, Humid: <55%")
            st.markdown("- **Mod Risk:** Temp: 25-28°C, Humid: 55-70%")
            st.markdown("- **Reduce Risk:** Temp: <20°C OR Humid: >80%")
        st.markdown("---")
        st.selectbox("Farm Location", ["Farm A", "Farm B", "Farm C"], key="farm_select")
        st.selectbox("Crop Type", ["Rice", "Maize", "Cotton", "Tomato"], key="crop_select")
        st.markdown("---")
        st.markdown("<h3 style='color: #4CAF50;'>Data Export</h3>", unsafe_allow_html=True)
        csv_data = convert_df_to_csv(st.session_state.historical_data)
        st.download_button(
            label="💾 Download History (CSV)",
            data=csv_data,
            file_name=f"mite_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
            use_container_width=True
        )
        with st.expander("ℹ️ About & Help"):
            st.markdown("**Use:** Select data source, input parameters, and predict risk.\n**Disclaimer:** Guidance tool. Combine with scouting.")

    # --- Main Dashboard Layout ---
    latest_data = st.session_state.historical_data.iloc[-1]
    col1, col2 = st.columns([2, 1.5])
    with col1:
        if st.session_state.predicted_risk:
            st.plotly_chart(create_risk_gauge(st.session_state.predicted_risk, st.session_state.predicted_confidence), use_container_width=True)
            st.write(f"Showing predicted risk for {st.session_state.predicted_date}")
        else:
            st.plotly_chart(create_risk_gauge(latest_data["risk_level"], latest_data["confidence"]), use_container_width=True)
            st.write(f"Showing latest historical risk (up to {latest_data['timestamp'].strftime('%Y-%m-%d')})")
    with col2:
        st.markdown("##### Recommended Actions")
        risk_to_show = st.session_state.predicted_risk if st.session_state.predicted_risk else latest_data["risk_level"]
        for rec in create_action_recommendations(risk_to_show):
            st.markdown(f"- {rec}")

    st.markdown("---")
    tab_titles = ["📈 Environment", "📉 Risk Trend", "🗺️ Field Map (Sim.)"]
    tab1, tab2, tab3 = st.tabs(tab_titles)
    with tab1:
        st.plotly_chart(plot_environmental_factors(st.session_state.historical_data), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_risk_timeline(st.session_state.historical_data), use_container_width=True)
    with tab3:
        st.plotly_chart(plot_mite_heatmap(st.session_state.historical_data), use_container_width=True)

    st.markdown("---")
    st.markdown("##### Recent Alerts & Notifications")
    alert_container = st.container(border=True)
    with alert_container:
        alerts_found = False
        if risk_to_show == "Severe":
            st.error("🚨 SEVERE RISK: Implement controls immediately.", icon="🚨")
            alerts_found = True
        elif risk_to_show == "High":
            st.warning("⚠️ HIGH RISK: Prepare interventions & monitor hotspots.", icon="⚠️")
            alerts_found = True
        elif risk_to_show == "Medium":
            st.warning("🟡 MEDIUM RISK: Increase scouting frequency.", icon="🟡")
            alerts_found = True
        st.info("ℹ️ Reminder: Check irrigation, avoid plant stress.", icon="💧")
        st.info("ℹ️ Beneficial populations stable (Simulated).", icon="🐞")
        if not alerts_found and risk_to_show != "Unknown":
            st.success("✅ Low Risk: Continue standard monitoring.", icon="✅")
        elif risk_to_show == "Unknown":
            st.info("Awaiting first prediction.")

if __name__ == "__main__":
    main()