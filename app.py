import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta, datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# import time # Removed for now

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Red Spider Mite Early Warning System",
    page_icon="🕷️",
    layout="wide"
)

# --- Configuration ---
ARTIFACTS_DIR = 'dashboard_artifacts'
HISTORY_LENGTH = 30

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
             st.warning(f"History file not found. Creating minimal history.")
             required_base_features = ['temperature', 'humidity', 'rainfall', 'N', 'P', 'K', 'ph', 'label']
             default_data = {'temperature': [25.0], 'humidity': [60.0], 'rainfall': [0.0],'N': [75], 'P': [45], 'K': [42], 'ph': [7.0], 'label': ['unknown'],'Red_spider_mite_category': ['Low']}
             for col in required_base_features:
                 if col not in default_data: default_data[col] = [0]
             history_df = pd.DataFrame(default_data, index=[pd.to_datetime(datetime.now().date() - timedelta(days=1))])
             history_df.index.name = 'timestamp'
        history_df = history_df.iloc[-HISTORY_LENGTH:]
        return model, scaler, label_encoder, features_list, history_df
    except FileNotFoundError as e: st.error(f"Error loading critical artifacts: {e}. Cannot proceed."); st.stop()
    except Exception as e: st.error(f"Unexpected error loading artifacts: {e}"); st.stop()

model, scaler, label_encoder, FEATURES, initial_history = load_artifacts()
class_names = list(label_encoder.classes_)

# --- Feature Engineering ---
def calculate_features(df):
    """Calculates lags and rolling features for the last row."""
    df = df.sort_index()
    # --- FIX: Enhanced duplicate check ---
    if df.index.duplicated().any():
        dup_dates = df.index[df.index.duplicated()].unique().strftime('%Y-%m-%d')
        st.error(f"Duplicate date index in input to calculate_features: {dup_dates}. Aborting.")
        return pd.DataFrame()

    latest_idx = df.index[-1]
    required_base = ['temperature', 'humidity', 'rainfall']
    if not all(col in df.columns for col in required_base):
        st.error(f"Missing one of {required_base} for feature calculation."); return pd.DataFrame()

    temp_df = df.copy()
    lags = [1, 3, 7]; windows = [3, 7, 14]; drivers = ['temperature', 'humidity', 'rainfall']
    for feature in drivers:
         for lag in lags: temp_df[f'{feature}_lag_{lag}'] = temp_df[feature].shift(lag) # shift handles short history with NaN
    for feature in drivers:
        for window in windows:
            roll = temp_df[feature].rolling(window=window, min_periods=1)
            temp_df[f'{feature}_roll_mean_{window}'] = roll.mean()
            if feature == 'temperature': temp_df[f'{feature}_roll_max_{window}'], temp_df[f'{feature}_roll_min_{window}'] = roll.max(), roll.min()
            if feature == 'humidity': temp_df[f'{feature}_roll_min_{window}'], temp_df[f'{feature}_roll_max_{window}'] = roll.min(), roll.max()
            if feature == 'rainfall': temp_df[f'{feature}_roll_sum_{window}'] = roll.sum()

    date_features = ['month', 'day_of_year', 'day_of_week', 'week_of_year']
    for feature in date_features:
         if feature not in temp_df.columns:
              if isinstance(temp_df.index, pd.DatetimeIndex):
                  idx = temp_df.index
                  if feature == 'month': temp_df[feature] = idx.month
                  elif feature == 'day_of_year': temp_df[feature] = idx.dayofyear
                  elif feature == 'day_of_week': temp_df[feature] = idx.dayofweek
                  elif feature == 'week_of_year': temp_df[feature] = idx.isocalendar().week
              else: temp_df[feature] = 0

    missing_cols = [col for col in FEATURES if col not in temp_df.columns]
    if missing_cols:
        # st.warning(f"Feature Calc: Missing: {missing_cols}. Filling with 0.") # Less verbose
        for col in missing_cols: temp_df[col] = 0

    try:
        # Reindex should be safe now if input index was unique
        last_row_features = temp_df.loc[[latest_idx]].reindex(columns=FEATURES)
        # Fill NaNs resulting from lags/rolls on short history
        for col in last_row_features.columns[last_row_features.isnull().any()]:
             last_row_features[col] = last_row_features[col].fillna(0) # Simple fill with 0 after lag/roll NaNs
        if last_row_features.isnull().any().any():
             st.warning("NaNs persist after fillna(0). Check feature calculation logic.")
             last_row_features = last_row_features.fillna(0) # Final catch-all
        return last_row_features
    except KeyError: st.error(f"KeyError on index '{latest_idx}' during reindex."); return pd.DataFrame()
    except Exception as e: st.error(f"Error finalizing features: {e}"); return pd.DataFrame()


# --- Visualization & Other Helpers ---
# create_risk_gauge, plot_environmental_factors, plot_risk_timeline, plot_mite_heatmap,
# create_action_recommendations, plot_contributing_factors, convert_df_to_csv
# (No changes needed in these from the previous version)
def create_risk_gauge(current_risk, latest_probabilities, class_names):
    """Create a risk gauge visualization."""
    risk_map = {name: i+1 for i, name in enumerate(class_names)}
    risk_value = risk_map.get(current_risk, 0)
    confidence = 0.0
    if latest_probabilities is not None and current_risk in class_names:
        try:
            risk_index = class_names.index(current_risk)
            if risk_index < len(latest_probabilities): confidence = latest_probabilities[risk_index]
        except (ValueError, IndexError): confidence = 0.0

    colors = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"] # Green, Yellow, Orange, Red
    steps, tickvals, ticktext = [], [], []
    for i, name in enumerate(class_names):
        steps.append({'range': [i, i+1], 'color': colors[i % len(colors)]})
        tickvals.append(i + 0.5); ticktext.append(name)

    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=risk_map.get(current_risk, 0) - 0.5 if risk_value > 0 else 0,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Current Predicted Risk: {current_risk}", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, len(class_names)], 'tickvals': tickvals, 'ticktext': ticktext},
            'bar': {'color': "rgba(0,0,0,0)", 'thickness':0},
            'steps': steps,
            'threshold': {
                'line': {'color': "black", 'width': 4}, 'thickness': 0.9,
                'value': risk_map.get(current_risk, 0) - 0.5 if risk_value > 0 else 0
            }
        }
    ))
    fig.add_annotation(x=0.5, y=0.05, text=f"Model Confidence: {confidence:.1%}", showarrow=False, font=dict(size=14))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20))
    return fig

def plot_environmental_factors(history_df):
    """Plot key environmental factors over time from history_df."""
    fig = make_subplots(rows=3, cols=1, subplot_titles=("Temperature & Humidity", "Rainfall", "Soil Nutrients (Example)"), shared_xaxes=True, vertical_spacing=0.1)
    if 'temperature' in history_df.columns: fig.add_trace(go.Scatter(x=history_df.index, y=history_df['temperature'], name="Temp (°C)"), row=1, col=1)
    if 'humidity' in history_df.columns: fig.add_trace(go.Scatter(x=history_df.index, y=history_df['humidity'], name="Humid (%)", line=dict(dash='dot')), row=1, col=1)
    if 'rainfall' in history_df.columns: fig.add_trace(go.Bar(x=history_df.index, y=history_df['rainfall'], name="Rain (mm)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df.get('N', pd.Series(index=history_df.index, dtype=float)), name="N"), row=3, col=1)
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df.get('P', pd.Series(index=history_df.index, dtype=float)), name="P"), row=3, col=1)
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df.get('K', pd.Series(index=history_df.index, dtype=float)), name="K"), row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="High Temp", row=1, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="blue", annotation_text="Low Humid", row=1, col=1)
    fig.update_layout(height=600, title_text="Key Environmental Factors Trend", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def plot_risk_timeline(history_df, predictions, class_names):
    """Plot predicted risk level over time."""
    risk_map = {name: i for i, name in enumerate(class_names)}
    valid_len = min(len(predictions), len(history_df))
    if len(predictions) != len(history_df): st.warning(f"Plotting risk timeline for last {valid_len} days.")
    predictions_plot, history_df_indices = predictions[-valid_len:], history_df.index[-valid_len:]
    risk_numeric = [risk_map.get(p, -1) for p in predictions_plot]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df_indices, y=risk_numeric, mode='lines+markers', name='Predicted Risk'))
    fig.update_layout(
        title='Predicted Risk Level Trend', xaxis_title='Date',
        yaxis=dict(title='Risk Level', tickmode='array', tickvals=list(risk_map.values()), ticktext=list(risk_map.keys()), range=[-0.5, len(class_names)-0.5]),
        hovermode="x unified")
    colors = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"]
    for i, name in enumerate(class_names):
        if name in risk_map: fig.add_shape(type="rect", xref="paper", yref="y", x0=0, y0=i-0.5, x1=1, y1=i+0.5, fillcolor=colors[i % len(colors)], opacity=0.15, layer="below", line_width=0)
    return fig

def plot_mite_heatmap(latest_prediction):
    """Create a *simulated* heatmap."""
    n_rows, n_cols = 5, 5; risk_seed_map = {'Low': 0.2, 'Medium': 0.4, 'High': 0.6, 'Severe': 0.8}; risk_seed = risk_seed_map.get(latest_prediction, 0.1)
    np.random.seed(42); heatmap_values = np.clip(np.random.rand(n_rows, n_cols) * 0.5 + (risk_seed * 0.5), 0, 1)
    fig = px.imshow(heatmap_values, color_continuous_scale='RdYlGn_r', labels=dict(color="Sim Risk"), title="Simulated Field Risk Map", aspect="equal")
    fig.update_layout(xaxis=dict(tickvals=list(range(n_cols)), ticktext=[f'S{i+1}' for i in range(n_cols)]), yaxis=dict(tickvals=list(range(n_rows)), ticktext=[f'R{i+1}' for i in range(n_rows)]))
    return fig

def create_action_recommendations(risk_level):
    """Generate action recommendations."""
    recommendations = {
        'Low': ["Continue regular monitoring (weekly).", "Maintain beneficials.", "Ensure proper irrigation."],
        'Medium': ["Increase monitoring (3-4 days).", "Check undersides of leaves.", "Consider releasing predators.", "Avoid plant stress."],
        'High': ["Prepare intervention.", "Identify hotspots.", "Consider spot treatments.", "Check resistance history."],
        'Severe': ["URGENT: Implement controls.", "Apply effective treatment.", "Evaluate effectiveness.", "Rotate modes of action.", "Notify neighbors."]
    }
    return recommendations.get(risk_level, ["No specific recommendations."])

def plot_contributing_factors(latest_row):
    """Create a *simulated* visualization of factors."""
    if latest_row is None or not isinstance(latest_row, pd.Series): return go.Figure().update_layout(title="Contributing Factors (Data Unavailable)")
    factors = {}; temp, humid, rain = latest_row.get('temperature'), latest_row.get('humidity'), latest_row.get('rainfall')
    if temp is not None: factors['High Temp'] = min(1, max(0, (temp - 20) / 15))
    if humid is not None: factors['Low Humidity'] = min(1, max(0, (70 - humid) / 40))
    if rain is not None: factors['Lack of Rain'] = 1 - min(1, rain / 10)
    factors['Plant Stress (Sim.)'] = np.random.uniform(0.1, 0.5) * factors.get('High Temp', 0.5)
    factors['Prev. Risk (Sim.)'] = np.random.uniform(0.2, 0.6)
    if not factors: return go.Figure().update_layout(title="Contributing Factors (Unavailable)")
    total = sum(factors.values()); normalized_factors = {k: v/total for k, v in factors.items()} if total > 0 else {}
    if not normalized_factors: return go.Figure().update_layout(title="Contributing Factors (Unavailable)")
    sorted_factors = dict(sorted(normalized_factors.items(), key=lambda item: item[1]))
    fig = go.Figure(go.Bar(x=list(sorted_factors.values()), y=list(sorted_factors.keys()), orientation='h'))
    fig.update_layout(title="Simulated Factors Influencing Risk", xaxis_title="Rel Influence (Sim.)", yaxis_title="Factor", height=300, margin=dict(l=150, r=20, t=50, b=30))
    return fig

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=True).encode('utf-8')

# --- Forecast Function ---
def display_forecast(history_df, model, scaler, features_list, label_encoder):
    """Display a 7-day forecast using the ML model iteratively."""
    forecast_results = []
    current_history_for_forecast = history_df.copy()
    if current_history_for_forecast.index.duplicated().any():
        st.warning("Removing duplicate dates from history before forecast.")
        current_history_for_forecast = current_history_for_forecast[~current_history_for_forecast.index.duplicated(keep='last')]

    st.markdown("---"); st.markdown("### Experimental 7-Day Risk Forecast")
    st.caption("Note: Assumes weather follows trends or manual input. Accuracy decreases further out.")
    if len(current_history_for_forecast) < 1: st.warning("Insufficient history."); return

    last_day = current_history_for_forecast.iloc[-1]
    second_last = current_history_for_forecast.iloc[-2] if len(current_history_for_forecast) > 1 else last_day
    last_temp, second_last_temp = last_day.get('temperature', 25.0), second_last.get('temperature', last_day.get('temperature', 25.0))
    last_humid, second_last_humid = last_day.get('humidity', 60.0), second_last.get('humidity', last_day.get('humidity', 60.0))
    last_rain = last_day.get('rainfall', 0.0)
    temp_trend, humid_trend = last_temp - second_last_temp, last_humid - second_last_humid
    rain_forecast_default = max(0.0, last_rain * 0.5)

    col1, col2 = st.columns([1, 5])
    with col1:
        adjust_weather = st.checkbox("Adjust Weather?", False, help="Override weather trend for forecast", key="adjust_weather_cb")
        ovr_temp = st.number_input("Avg Temp (°C)", value=last_temp + temp_trend, step=0.5, disabled=not adjust_weather, key="ovr_temp")
        ovr_humid = st.number_input("Avg Humidity (%)", value=max(20.0, min(100.0, last_humid + humid_trend)), step=1.0, disabled=not adjust_weather, key="ovr_humid")
        ovr_rain = st.number_input("Avg Rain (mm)", value=float(rain_forecast_default), min_value=0.0, step=1.0, format="%.1f", disabled=not adjust_weather, key="ovr_rain")

    with col2:
        forecast_placeholder = st.empty(); prog_bar = st.progress(0, "Generating forecast...")
        forecast_failed = False
        for i in range(7):
            if not isinstance(current_history_for_forecast.index, pd.DatetimeIndex):
                 st.error("Forecast Error: History index not DatetimeIndex."); forecast_failed = True; break
            next_pred_date = current_history_for_forecast.index.max() + timedelta(days=1)

            # --- FIX: Check *again* for duplicates right before adding ---
            if next_pred_date in current_history_for_forecast.index:
                 st.error(f"Forecast Error: Date {next_pred_date.strftime('%Y-%m-%d')} duplicate in sequence.")
                 forecast_results.append(("Duplicate", 0.0)); forecast_failed = True; break

            current_last_day_data = current_history_for_forecast.iloc[-1]
            if adjust_weather: forecast_temp, forecast_humid, forecast_rain = ovr_temp, ovr_humid, ovr_rain
            else:
                forecast_temp = current_last_day_data.get('temperature', 25.0) + temp_trend * (0.8**i)
                forecast_humid = max(20.0, min(100.0, current_last_day_data.get('humidity', 60.0) + humid_trend * (0.8**i)))
                forecast_rain = max(0.0, current_last_day_data.get('rainfall', 0.0) * (0.6**(i+1)))

            new_forecast_data = pd.Series(name=next_pred_date)
            new_forecast_data['temperature'], new_forecast_data['humidity'], new_forecast_data['rainfall'] = forecast_temp, forecast_humid, forecast_rain
            new_forecast_data['N'], new_forecast_data['P'], new_forecast_data['K'], new_forecast_data['ph'] = current_last_day_data.get('N', 75), current_last_day_data.get('P', 45), current_last_day_data.get('K', 42), current_last_day_data.get('ph', 7.0)
            new_forecast_data['label'] = current_last_day_data.get('label', 'unknown')
            new_forecast_data['month'], new_forecast_data['day_of_year'], new_forecast_data['day_of_week'] = next_pred_date.month, next_pred_date.dayofyear, next_pred_date.dayofweek
            new_forecast_data['week_of_year'] = next_pred_date.isocalendar().week

            # Pass a copy for calculation to avoid modifying the main forecast history prematurely
            temp_history_for_calc = pd.concat([current_history_for_forecast, pd.DataFrame(new_forecast_data).T])
            temp_history_for_calc.index = pd.to_datetime(temp_history_for_calc.index)

            try:
                # Calculate features based on the temporary history including the new day
                forecast_features_df = calculate_features(temp_history_for_calc)
                if not forecast_features_df.empty:
                    forecast_features_scaled = scaler.transform(forecast_features_df[FEATURES])
                    forecast_pred_encoded = model.predict(forecast_features_scaled)[0]
                    forecast_pred_proba = model.predict_proba(forecast_features_scaled)[0]
                    forecast_pred_decoded = label_encoder.inverse_transform([forecast_pred_encoded])[0]
                    confidence = forecast_pred_proba.max()
                    forecast_results.append((forecast_pred_decoded, confidence))

                    # Add the full data (inputs + features + prediction) for this forecast day
                    new_forecast_data_full = pd.concat([new_forecast_data, forecast_features_df.iloc[0]])
                    new_forecast_data_full['Red_spider_mite_category_predicted'] = forecast_pred_decoded

                    # --- FIX: Update history using .loc to avoid concat issues ---
                    # Assign the new row using its index (next_pred_date)
                    current_history_for_forecast.loc[next_pred_date] = new_forecast_data_full
                    # Ensure index remains datetime after using .loc
                    current_history_for_forecast.index = pd.to_datetime(current_history_for_forecast.index)
                    # Sort index just in case .loc messed up order (unlikely but safe)
                    current_history_for_forecast = current_history_for_forecast.sort_index()

                else: # Feature calculation failed
                    st.warning(f"Feature calculation failed for forecast day {i+1}."); forecast_results.append(("Error", 0.0)); forecast_failed = True; break
            except Exception as e:
                st.error(f"Prediction error on forecast day {i+1}: {e}"); forecast_results.append(("Error", 0.0)); forecast_failed = True; break
            prog_bar.progress((i + 1) / 7, f"Forecast day {i+1}/{7}...")

        if forecast_failed: prog_bar.progress(1.0, "Forecast stopped.")
        forecast_cols_list = forecast_placeholder.columns(7)
        start_date_for_labels = history_df.index.max()
        for i, (risk, conf) in enumerate(forecast_results):
             if i < len(forecast_cols_list):
                 with forecast_cols_list[i]:
                     day_name = (start_date_for_labels + timedelta(days=i+1)).strftime("%a %d")
                     st.markdown(f"**{day_name}**")
                     color = "grey" if risk in ["Error", "Duplicate"] else "#2ECC71" if risk == "Low" else "#F1C40F" if risk == "Medium" else "#E67E22" if risk == "High" else "#E74C3C"
                     st.markdown(f"<div style='padding:8px; border-radius:5px; background-color:{color}; color:white; text-align:center; font-size:small;'>{risk}<br>{conf:.0%}</div>", unsafe_allow_html=True)
        if not forecast_failed and len(forecast_results) == 7: prog_bar.progress(1.0, "Forecast complete.")

# --- Initialize Session State ---
def initialize_state(initial_history, model, scaler, features_list, label_encoder, class_names):
    """Initializes session state variables."""
    if 'history_df' not in st.session_state:
        st.session_state.history_df = initial_history.copy()
        if st.session_state.history_df.index.duplicated().any():
             st.warning("Duplicate dates found in initial history file, keeping last entry.")
             st.session_state.history_df = st.session_state.history_df[~st.session_state.history_df.index.duplicated(keep='last')]

    if 'predictions' not in st.session_state:
        if 'Red_spider_mite_category_predicted' in st.session_state.history_df.columns: pred_source_col = 'Red_spider_mite_category_predicted'
        elif 'Red_spider_mite_category' in st.session_state.history_df.columns: pred_source_col = 'Red_spider_mite_category'
        else: pred_source_col = None
        if pred_source_col and pred_source_col in st.session_state.history_df.columns:
            st.session_state.predictions = st.session_state.history_df[pred_source_col].tolist()
        else: st.session_state.predictions = ['Unknown'] * len(st.session_state.history_df)
        st.session_state.predictions = st.session_state.predictions[-HISTORY_LENGTH:]

    if 'latest_probabilities' not in st.session_state:
        try:
            # Make sure history isn't empty before calculating features
            if not st.session_state.history_df.empty:
                 initial_last_features_df = calculate_features(st.session_state.history_df)
                 if not initial_last_features_df.empty:
                     initial_last_features_scaled = scaler.transform(initial_last_features_df[FEATURES])
                     st.session_state.latest_probabilities = model.predict_proba(initial_last_features_scaled)[0]
                 else:
                     st.warning("Init probability calc failed (empty features). Defaulting."); st.session_state.latest_probabilities = np.array([1.0/len(class_names)] * len(class_names))
            else:
                 st.warning("History is empty, cannot calculate initial probabilities. Defaulting."); st.session_state.latest_probabilities = np.array([1.0/len(class_names)] * len(class_names))

        except Exception as e:
            st.warning(f"Init probability calc error: {e}. Defaulting."); st.session_state.latest_probabilities = np.array([1.0/len(class_names)] * len(class_names))

# --- Main App Function ---
def main():
    initialize_state(initial_history, model, scaler, FEATURES, label_encoder, class_names)
    st.title("🕷️ Red Spider Mite Early Warning System")
    st.markdown("#### Real-time monitoring and ML-based predictive alerts")

    with st.sidebar:
        st.header("Predict for Next Day")
        if st.session_state.history_df.empty:
            st.error("History data is empty. Cannot proceed."); st.stop()
        last_date = st.session_state.history_df.index.max()
        next_date = last_date + timedelta(days=1)
        st.write(f"Predicting for: {next_date.strftime('%Y-%m-%d')}")

        last_known_data = st.session_state.history_df.iloc[-1]
        default_temp, default_humid, default_rain = float(last_known_data.get('temperature', 25.0)), float(last_known_data.get('humidity', 60.0)), float(last_known_data.get('rainfall', 0.0))
        default_n, default_p, default_k, default_ph = float(last_known_data.get('N', 75)), float(last_known_data.get('P', 45)), float(last_known_data.get('K', 42)), float(last_known_data.get('ph', 7.0))

        new_temp = st.slider("Next Day Temp (°C)", -5.0, 45.0, default_temp, 0.1, key="next_temp")
        new_humid = st.slider("Next Day Humidity (%)", 10.0, 100.0, default_humid, 0.5, key="next_humid")
        new_rain = st.number_input("Next Day Rainfall (mm)", 0.0, 200.0, default_rain, 0.1, key="next_rain", format="%.1f")

        predict_button = st.button("▶️ Predict & Update History", key="predict_button", use_container_width=True)

        st.markdown("---"); st.markdown("##### Current Conditions (Last Recorded)")
        col1, col2 = st.columns(2); col1.metric("Temp", f"{last_known_data.get('temperature', np.nan):.1f}°C"); col1.metric("Rain", f"{last_known_data.get('rainfall', np.nan):.1f} mm"); col2.metric("Humidity", f"{last_known_data.get('humidity', np.nan):.1f}%")

        st.markdown("---")
        with st.expander("Simulation Suggestions"): st.caption("Try these inputs over several days:"); st.markdown("- **Heat/Dry:** Temp: 30-35°C, Humid: <55%, Rain: 0"); st.markdown("- **Mod Risk:** Temp: 25-28°C, Humid: 55-70%, Rain: 0"); st.markdown("- **Reduce Risk:** Temp: <20°C OR Humid: >80% OR Rain: >10mm")
        st.markdown("---"); st.selectbox("Farm Location", ["Farm A", "Farm B", "Farm C"], key="farm_select"); st.selectbox("Crop Type", ["Rice", "Maize", "Cotton", "Tomato"], key="crop_select")
        st.markdown("---"); st.markdown("##### Data Export"); csv_data = convert_df_to_csv(st.session_state.history_df)
        st.download_button(label="💾 Download History (CSV)", data=csv_data, file_name=f"mite_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime='text/csv', use_container_width=True)
        with st.expander("ℹ️ About & Help"): st.markdown("**Use:** Input weather, click Predict.\n**Disclaimer:** Guidance tool. Combine w/ scouting. Map/Factors simulated.")

    if predict_button:
        # --- FIX: Check date *before* creating any data ---
        if next_date in st.session_state.history_df.index:
             st.error(f"Prediction for {next_date.strftime('%Y-%m-%d')} already exists.")
        else:
            new_data = pd.Series(name=next_date); new_data['temperature'], new_data['humidity'], new_data['rainfall'] = new_temp, new_humid, new_rain
            new_data['N'], new_data['P'], new_data['K'], new_data['ph'] = default_n, default_p, default_k, default_ph
            new_data['label'] = last_known_data.get('label', 'unknown'); new_data['month'], new_data['day_of_year'], new_data['day_of_week'] = next_date.month, next_date.dayofyear, next_date.dayofweek
            new_data['week_of_year'] = next_date.isocalendar().week

            # Create temporary df for feature calc
            temp_history_for_calc = pd.concat([st.session_state.history_df, pd.DataFrame(new_data).T])
            temp_history_for_calc.index = pd.to_datetime(temp_history_for_calc.index)

            try:
                new_features_df = calculate_features(temp_history_for_calc) # calculate_features checks duplicates
                if not new_features_df.empty:
                    new_features_scaled = scaler.transform(new_features_df[FEATURES]); prediction_encoded = model.predict(new_features_scaled)[0]
                    current_prediction_proba = model.predict_proba(new_features_scaled)[0]; prediction_decoded = label_encoder.inverse_transform([prediction_encoded])[0]
                    new_data_full = pd.concat([new_data, new_features_df.iloc[0]]); new_data_full['Red_spider_mite_category_predicted'] = prediction_decoded

                    # --- FIX: Update using .loc ---
                    st.session_state.history_df.loc[next_date] = new_data_full
                    # Ensure index is datetime and sorted after update
                    st.session_state.history_df.index = pd.to_datetime(st.session_state.history_df.index)
                    st.session_state.history_df = st.session_state.history_df.sort_index()
                    # --- Ensure uniqueness (belt-and-braces) ---
                    st.session_state.history_df = st.session_state.history_df[~st.session_state.history_df.index.duplicated(keep='last')]
                    st.session_state.history_df = st.session_state.history_df.iloc[-HISTORY_LENGTH:] # Trim

                    st.session_state.predictions.append(prediction_decoded); st.session_state.predictions = st.session_state.predictions[-HISTORY_LENGTH:]
                    st.session_state.latest_probabilities = current_prediction_proba
                    st.success(f"Prediction for {next_date.strftime('%Y-%m-%d')} added!")
                    # --- FIX: Removed st.rerun() ---
                # else: Error message handled inside calculate_features
            except Exception as e: st.error(f"Error during prediction update: {e}"); st.exception(e)

    # --- Main Dashboard Layout ---
    # (Layout remains the same, relying on updated state)
    latest_prediction = st.session_state.predictions[-1] if st.session_state.predictions else "Unknown"
    latest_probabilities = st.session_state.get('latest_probabilities', None)
    col1, col2 = st.columns([2, 1.5]);
    with col1: st.plotly_chart(create_risk_gauge(latest_prediction, latest_probabilities, class_names), use_container_width=True)
    with col2:
        st.markdown("##### Recommended Actions")
        if latest_prediction != "Unknown":
            recommendations = create_action_recommendations(latest_prediction)
            if recommendations:
                for action in recommendations: # Use for loop
                    st.markdown(f"- {action}")
            else: st.markdown("No specific actions.")
        else: st.markdown("Prediction unavailable.")

    display_forecast(st.session_state.history_df, model, scaler, FEATURES, label_encoder)
    st.markdown("---")
    tab_titles = ["📈 Environment", "📉 Risk Trend", "🗺️ Field Map (Sim.)", "📊 Factors (Sim.)"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)
    with tab1: st.plotly_chart(plot_environmental_factors(st.session_state.history_df), use_container_width=True)
    with tab2: st.plotly_chart(plot_risk_timeline(st.session_state.history_df, st.session_state.predictions, class_names), use_container_width=True)
    with tab3: st.plotly_chart(plot_mite_heatmap(latest_prediction), use_container_width=True)
    with tab4: latest_row_data = st.session_state.history_df.iloc[-1] if not st.session_state.history_df.empty else None; st.plotly_chart(plot_contributing_factors(latest_row_data), use_container_width=True)
    st.markdown("---"); st.markdown("##### Recent Alerts & Notifications"); alert_container = st.container(border=True)
    with alert_container:
        alerts_found = False
        if latest_prediction == "Severe": st.error("🚨 SEVERE RISK: Implement controls immediately.", icon="🚨"); alerts_found = True
        elif latest_prediction == "High": st.warning("⚠️ HIGH RISK: Prepare interventions & monitor hotspots.", icon="⚠️"); alerts_found = True
        elif latest_prediction == "Medium": st.warning("🟡 MEDIUM RISK: Increase scouting frequency.", icon="🟡"); alerts_found = True
        st.info("ℹ️ Reminder: Check irrigation, avoid plant stress.", icon="💧"); st.info("ℹ️ Beneficial populations stable (Simulated).", icon="🐞")
        if not alerts_found and latest_prediction != "Unknown": st.success("✅ Low Risk: Continue standard monitoring.", icon="✅")
        elif latest_prediction == "Unknown": st.info("Awaiting first prediction.")

if __name__ == "__main__":
    main()