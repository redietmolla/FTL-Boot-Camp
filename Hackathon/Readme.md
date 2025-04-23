# ⚡ SmartChargeAI – EV Charging Optimization App

SmartChargeAI is an AI-powered web application that helps electric vehicle (EV) users optimize their charging schedule to:

- 🔋 Extend battery life
- 💰 Reduce electricity cost (based on demand)
- 🌍 Lower carbon emissions

The system uses **real hourly energy data** to predict future demand using **XGBoost**, and schedules charging using **linear programming** via PuLP.

---

## 🧠 Key Features

- 📊 Predicts electricity demand (as a proxy for price)
- 🔄 Optimizes charging hours for cost and battery health
- 💡 Shows final battery level, cost, and CO₂ saved
- 📈 Visualizes hourly charging plan

---

## 📁 Dataset

**ElectricityLoadDiagrams20112014**

- Source: UCI Machine Learning Repository
- File used: `LD2011_2014.txt`
- Resolution: Hourly energy load data from 2011 to 2014

Download and place it in your project directory:  
👉 [UCI Link](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)

---

## 🚀 How to Run the App

### 1. 🔧 Install Required Libraries

Make sure you have Python 3.8+ and run:

pip install streamlit pandas numpy xgboost pulp scikit-learn matplotlib seaborn

### 2. ▶️ Launch the App

Run the Streamlit app with:

streamlit run ev_realdata_app.py
