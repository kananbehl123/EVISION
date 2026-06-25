import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# --------------------------
# Load datasets
# --------------------------
@st.cache_data
def load_ev_data():
    df = pd.read_csv(r"C:\Users\HP\Desktop\streamlit\evdb_statewise.csv")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    return df

@st.cache_data
def load_charging_data():
    df = pd.read_csv(r"C:\Users\HP\Desktop\streamlit\ev_station_data_5000_rows dummy.csv")
    df.rename(columns={"State Name": "State"}, inplace=True)
    return df

df = load_ev_data()
charging_df = load_charging_data()

st.title(" India EV Analytics Dashboard")
st.subheader("Based on Monthly Vehicle Registration + Charging Infrastructure Data")

# --------------------------
# Sidebar Filters
# --------------------------
st.sidebar.header(" Filters")

# Prediction type selector
prediction_option = st.sidebar.radio(
    "Choose a prediction type:",
    [
        "1. Total EVs Sold per State",
        "2. EV Percentage using Fuel Mix",
        "3. EV Sales Forecast (Time Series)",
        "4. Identify States Lacking Charging Infrastructure"
    ]
)

# Filters for year, month, use type, category
years = sorted(df['Year'].dropna().unique())
months = sorted(df['Month_name'].dropna().unique())
use_types = df['Vehicle Use type'].dropna().unique()
categories = df['Vehicle Category'].dropna().unique()

selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)
selected_months = st.sidebar.multiselect("Select Month(s)", months, default=months)
selected_use_types = st.sidebar.multiselect("Select Vehicle Use Type(s)", use_types, default=use_types)
selected_categories = st.sidebar.multiselect("Select Vehicle Category(s)", categories, default=categories)

# Apply filters
filtered_df = df[
    (df['Year'].isin(selected_years)) &
    (df['Month_name'].isin(selected_months)) &
    (df['Vehicle Use type'].isin(selected_use_types)) &
    (df['Vehicle Category'].isin(selected_categories))
].copy()

# Add EV Total column
filtered_df['EV Total'] = (
    filtered_df['ELECTRIC(BOV)'] +
    filtered_df['PLUG-IN HYBRID EV'] +
    filtered_df['PURE EV'] +
    filtered_df['STRONG HYBRID EV']
)

# ========================
# PREDICTION 1
# ========================
if prediction_option == "1. Total EVs Sold per State":
    st.header(" Predict Total EVs Sold per State")

    ev_data = filtered_df.groupby('State')[['EV Total']].sum().reset_index()
    st.dataframe(ev_data)

    state_ev = filtered_df.groupby('State').sum(numeric_only=True).reset_index()

    if 'EV Total' in state_ev.columns and not state_ev['EV Total'].isnull().all():
        X = state_ev.drop(columns=['EV Total'])
        y = state_ev['EV Total']

        model_ev_total = RandomForestRegressor()
        model_ev_total.fit(X.select_dtypes(include='number'), y)

        preds = model_ev_total.predict(X.select_dtypes(include='number'))
        state_ev['Predicted EVs'] = preds

        st.markdown("### Actual vs Predicted EV Sales per State")
        st.dataframe(state_ev[['State', 'EV Total', 'Predicted EVs']])
    else:
        st.warning(" Not enough data to train EV total prediction model.")

# ========================
# PREDICTION 2
# ========================
elif prediction_option == "2. EV Percentage using Fuel Mix":
    st.header(" Predict EV % per State")

    if 'Total' in filtered_df.columns:
        filtered_df['Total Vehicles'] = filtered_df['Total']
        filtered_df['EV %'] = filtered_df['EV Total'] / filtered_df['Total Vehicles']

        fuel_features = [
            'CNG ONLY', 'DIESEL', 'PETROL', 'LPG ONLY',
            'ELECTRIC(BOV)', 'PLUG-IN HYBRID EV', 'PURE EV'
        ]

        grouped_fuel = filtered_df.groupby('State')[fuel_features + ['EV %']].sum().reset_index()

        X2 = grouped_fuel[fuel_features]
        y2 = grouped_fuel['EV %']

        model_ev_percent = RandomForestRegressor()
        model_ev_percent.fit(X2, y2)

        predicted_percent = model_ev_percent.predict(X2)
        grouped_fuel['Predicted EV %'] = predicted_percent

        st.markdown("### Actual vs Predicted EV Percentage per State")
        st.dataframe(grouped_fuel[['State', 'EV %', 'Predicted EV %']])
    else:
        st.warning(" 'Total' column missing in dataset. Cannot compute EV %.")

# ========================
# PREDICTION 3
# ========================
elif prediction_option == "3. EV Sales Forecast (Time Series)":
    st.header(" Forecast EV Sales Growth")

    state_option = st.selectbox("Select a state to forecast EV sales", filtered_df['State'].unique())

    df_state = filtered_df[filtered_df['State'] == state_option]
    df_state = df_state.groupby('Date')[['ELECTRIC(BOV)', 'PLUG-IN HYBRID EV', 'PURE EV', 'STRONG HYBRID EV']].sum()
    df_state['EV Total'] = df_state.sum(axis=1)
    df_state = df_state.reset_index()

    df_prophet = df_state[['Date', 'EV Total']].rename(columns={'Date': 'ds', 'EV Total': 'y'})

    if len(df_prophet) > 10:
        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        fig = px.line(forecast, x='ds', y='yhat', title=f"EV Sales Forecast for {state_option}")
        st.plotly_chart(fig)

        st.markdown("### Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
    else:
        st.warning("Not enough data points to forecast EV sales with Prophet.")

# ========================
# PREDICTION 4 – Identify Infrastructure Gaps
# ========================
elif prediction_option == "4. Identify States Lacking Charging Infrastructure":
    st.header(" Identify States Lacking EV Infrastructure")

    # Total EVs per state
    ev_statewise = filtered_df.groupby('State')[['EV Total']].sum().reset_index()

    # Merge EV data with charging data
    merged_df = pd.merge(ev_statewise, charging_df[['State', 'total-charging-stations']], on='State', how='inner')
    merged_df = merged_df[merged_df['total-charging-stations'] > 0]  # avoid division by 0

    # Compute EV load per station
    merged_df['EVs per Station'] = merged_df['EV Total'] / merged_df['total-charging-stations']

    # Target: 1 station per 500 EVs
    TARGET_EV_PER_STATION = 500
    merged_df['Required Stations'] = np.ceil(merged_df['EV Total'] / TARGET_EV_PER_STATION)

    # Calculate shortage (no negative values)
    merged_df['Station Shortage'] = merged_df['Required Stations'] - merged_df['total-charging-stations']
    merged_df['Station Shortage'] = merged_df['Station Shortage'].apply(lambda x: max(x, 0))

    # Display full table
    st.markdown("### EV Load and Required Charging Stations per State")
    st.dataframe(
        merged_df[['State', 'EV Total', 'total-charging-stations', 'EVs per Station', 'Required Stations', 'Station Shortage']]
        .sort_values(by='EVs per Station', ascending=False)
    )

    # Bar chart of EV Load
    fig1 = px.bar(
        merged_df.sort_values(by='EVs per Station', ascending=False),
        x='State',
        y='EVs per Station',
        title=" EV Load per Charging Station (Higher = More Urgent Need)",
        labels={'EVs per Station': 'EVs per Charging Station'}
    )
    st.plotly_chart(fig1)

    # Bar chart of Shortage
    fig2 = px.bar(
        merged_df.sort_values(by='Station Shortage', ascending=False),
        x='State',
        y='Station Shortage',
        title="Required Additional Charging Stations by State",
        labels={'Station Shortage': 'Required Additional Stations'}
    )
    st.plotly_chart(fig2)

    st.markdown(
        """
            - MADE BY - EDA VERSE
        """
    )
