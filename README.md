# AI_capstone

# Red Spider Mite Early Warning System

## Description
This project develops a predictive system for forecasting Red Spider Mite (RSM) outbreaks in agricultural settings. It utilizes synthetic data generation, biologically-informed population simulation, machine learning modeling (XGBoost), and an interactive Streamlit dashboard to provide farmers with an early warning tool for effective pest management.

## Problem Statement
Red Spider Mites are significant agricultural pests causing substantial crop damage and economic loss. Traditional management is often reactive. This project aims to create a proactive system to predict outbreaks, enabling timely Integrated Pest Management (IPM) strategies, reducing pesticide reliance, and improving crop yields.

## Key Features
*   **Realistic Data Synthesis:** Generates long-term synthetic environmental and agricultural data incorporating seasonality, climate trends, oscillations, and realistic rainfall patterns.
*   **Enhanced Population Simulation:** Models RSM population dynamics based on temperature, humidity, density dependence, predation, pesticide application, and resistance development.
*   **Comprehensive Feature Engineering:** Creates lag, rolling window, calendar, interaction, and management-related features for robust model training.
*   **Machine Learning Prediction:** Employs an XGBoost classifier, trained on SMOTE-balanced data, to predict RSM risk categories (Low, Medium, High, Severe).
*   **Interactive Dashboard:** A Streamlit application visualizes current risk, environmental factors, risk trends, field maps (simulated), and provides actionable recommendations and forecasts.

## Methodology Overview
1.  Load and preprocess initial RSM data.
2.  Generate extensive synthetic environmental and agricultural data.
3.  Simulate RSM population counts using an enhanced biological model.
4.  Perform feature engineering to capture temporal patterns and interactions.
5.  Train an XGBoost classification model using stratified splitting and SMOTE for imbalance.
6.  Evaluate model performance on unseen test data.
7.  Develop a Streamlit dashboard for visualization and decision support.

## Technology Stack
*   **Python:** 3.8+
*   **Core Libraries:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn, XGBoost, imbalanced-learn (for SMOTE)
*   **Visualization:** Matplotlib, Plotly, Streamlit
*   **Optional:** Joblib (for model saving/loading)

## Setup & Installation
1.  Clone the repository:
    ```bash
    git clone [your-repo-link]
    cd [repository-name]
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
**Run the Dashboard:**
    ```bash
    streamlit run app.py
    ```

## Project Structure (Example)
AI_capstone/
├── dashboard_artifacts/
├── .gitignore
├── Data_generation.ipynb
├── Model_training.ipynb
├── README.md
├── app.py
├── full_red_spider_count.csv
├── red_spider_mite_forecast_data.csv
├── requirements.txt

## Future Work
*   Integration with real-time IoT sensor data.
*   Hyperparameter tuning and comparison with other ML models (LSTM, Random Forest).
*   Validation of the simulation model against real-world outbreak data.
*   Deployment to a cloud platform.
*   Incorporation of more detailed biological factors (e.g., mite life stages).


## Phenode Access
To enable real-time predictions, update your Phenode credentials in app.py
```bash
PHENODE_USERNAME = "PHENODE_LINK_USERNAME"
PHENODE_PASSWORD = "PHENODE_LINK_PASSWORD"
  ```