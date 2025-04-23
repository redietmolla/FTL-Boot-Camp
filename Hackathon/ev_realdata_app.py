import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from pulp import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Smart EV Charging Optimizer", layout="centered")

st.title("⚡ Smart EV Charging Optimizer")
st.markdown("Optimize your EV charging schedule using real electricity demand data.")

# -------------------------
# 1. Load Real Dataset
# -------------------------
@st.cache_data
def load_electricity_data():
    df = pd.read_csv("LD2011_2014.txt", sep=";", index_col=0, parse_dates=True, decimal=',')
    df = df.resample('1h').mean() # hourly average
 
    df['total_load'] = df.sum(axis=1)
    df = df[['total_load']].dropna()
    df['hour'] = df.index.hour
    df['day'] = (df.index - df.index.min()).days
    df = df.reset_index()
    return df

# -------------------------
# 2. Train ML Model (XGBoost)
# -------------------------
def train_model(df):
    # Filter to first 60 days
    df_train = df[df['day'] < 60].copy()

    # Scale total_load to simulate realistic price per kWh ($0.10 – $0.25)
    load_min = df_train['total_load'].min()
    load_max = df_train['total_load'].max()
    df_train['simulated_price'] = 0.10 + 0.15 * ((df_train['total_load'] - load_min) / (load_max - load_min))

    # Prepare features and targets
    X = df_train[['day', 'hour']]
    y = df_train['simulated_price']

    # Train/test split and model training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    return model


# -------------------------
# 3. Predict Prices
# -------------------------
def predict_prices(model):
    future_hours = np.array([18, 19, 20, 21, 22, 23])
    future_day = np.full_like(future_hours, 61)
    future_X = pd.DataFrame({'day': future_day, 'hour': future_hours})
    predicted_prices = model.predict(future_X)
    return list(range(18, 24)), predicted_prices

# -------------------------
# 4. Load + Predict
# -------------------------
@st.cache_data
def load_and_predict():
    df = load_electricity_data()
    model = train_model(df)
    return predict_prices(model)

available_hours, predicted_prices = load_and_predict()

# -------------------------
# 5. User Inputs
# -------------------------
st.markdown("### 🔌 Charging Settings")
battery_capacity = st.number_input("Battery Capacity (kWh)", value=60)
current_charge = st.number_input("Current Charge (kWh)", min_value=0.0, max_value=float(battery_capacity), value=20.0)

charging_rate = st.number_input("Charging Rate (kW/hour)", value=7.2)
battery_health_mode = st.checkbox("Optimize for Battery Health (limit to 80%)", value=True)

target_charge = battery_capacity * (0.80 if battery_health_mode else 1.0)
needed_charge = max(0, target_charge - current_charge)

# -------------------------
# 6. Optimization with PuLP
# -------------------------
charge_vars = LpVariable.dicts("ChargeAtHour", available_hours, 0, 1, cat='Continuous')
prob = LpProblem("EVChargingOptimization", LpMinimize)

prob += lpSum([charge_vars[h] * charging_rate * predicted_prices[i] for i, h in enumerate(available_hours)])
prob += lpSum([charge_vars[h] * charging_rate for h in available_hours]) >= needed_charge
prob.solve()

# -------------------------
# 7. Output
# -------------------------
charging_schedule = {h: charge_vars[h].varValue for h in available_hours}
total_energy = sum(charging_schedule[h] * charging_rate for h in available_hours)
final_charge = current_charge + total_energy
total_cost = sum(charging_schedule[h] * charging_rate * predicted_prices[i]
                 for i, h in enumerate(available_hours))
score = 100 if final_charge <= battery_capacity * 0.80 else 80 if final_charge <= battery_capacity * 0.90 else 60
co2_saved = total_energy * 0.5

st.markdown("### ⚙️ Optimized Charging Schedule")
for h in available_hours:
    if charging_schedule[h] > 0:
        st.write(f"• **{h}:00** → Charge for **{charging_schedule[h]:.2f} hour(s)**")

st.success(f"Final Battery Charge: {final_charge:.2f} kWh")
st.info(f"Total Charging Cost (proxy): ${total_cost:.2f}")
st.metric("🔋 Battery Health Score", f"{score}/100")
st.metric("🌍 Estimated CO₂ Saved", f"{co2_saved:.2f} kg")
st.metric("💰 Total Charging Cost (estimated)", f"${total_cost:.2f}")
st.caption("Estimated based on predicted energy prices ($/kWh)")


# -------------------------
# 8. Plot
# -------------------------
fig, ax = plt.subplots(figsize=(8, 4))
bars = [charging_schedule[h] * charging_rate for h in available_hours]
price_trend = [p * charging_rate for p in predicted_prices]
ax.bar([f"{h}:00" for h in available_hours], bars, color='skyblue', label='Energy Charged (kWh)')
ax.plot([f"{h}:00" for h in available_hours], price_trend, color='red', marker='o', label='Predicted Load (proxy $)')
ax.set_title("Charging Schedule Based on Real Load Data")
ax.set_xlabel("Hour")
ax.set_ylabel("Energy (kWh) / Load")
ax.grid(True)
ax.legend()
plt.tight_layout()
st.pyplot(fig)
