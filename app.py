import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Page layouts and global theme presentation
st.set_page_config(
    page_title="CarVal | Asset Valuation Engine", 
    layout="centered", 
    page_icon="📊"
)

# Main Title Presentation
st.title("📊 CarVal Asset Pricing Portal")
st.markdown("### **Used Car Valuation Engine**")

st.write(
    "This predictive tool utilizes an optimized machine learning ensemble "
    "to calculate real-time used car wholesale valuations based on historical "
    "market patterns, protecting trade-in transaction margins."
)
st.markdown("---")

# Pointing explicitly inside your models/ subfolder path
MODEL_PATH = os.path.join('models', 'car_price_production_model.joblib')
SCALER_PATH = os.path.join('models', 'car_price_scaler.joblib')

@st.cache_resource
def load_production_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

production_model = None
production_scaler = None

try:
    production_model, production_scaler = load_production_assets()
except FileNotFoundError:
    st.error(f"🚨 Structural Error: Verification failed. Could not locate artifact files at: '{MODEL_PATH}' or '{SCALER_PATH}'")

# Organize user inputs via neat column containers
col1, col2 = st.columns(2)

with col1:
    mileage = st.number_input("Vehicle Total Mileage (km)", min_value=0, max_value=600000, value=75000, step=1000)
    engine_v = st.number_input("Engine Displacement Volume (Liters)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
    brand = st.selectbox("Vehicle Manufacturer / Brand", ["Audi", "BMW", "Mercedes-Benz", "Mitsubishi", "Renault", "Toyota", "Volkswagen"])

with col2:
    body = st.selectbox("Chassis / Body Configuration", ["Crossover", "hatch", "other", "sedan", "vagon", "van"])
    fuel_type = st.selectbox("Engine Fuel Mechanism", ["Diesel", "Gas", "Other", "Petrol"])
    registered = st.selectbox("Valid License Registration?", ["Yes", "No"])

if st.button("Generate Asset Valuation", type="primary"):
    
    if production_model is None or production_scaler is None:
        st.error("Execution blocked. Loaded pipeline elements are empty.")
    else:
        # 1. SCALE FACTOR: Divide real mileage by 1000 to match training data scale (e.g., 75000 -> 75.0)
        scaled_mileage = mileage / 1000.0
        
        # 2. Base feature mapping using the correctly scaled mileage unit
        raw_feature_map = {
            'Mileage': scaled_mileage,
            'EngineV': engine_v,
            'Brand_BMW': int(brand == "BMW"),
            'Brand_Mercedes-Benz': int(brand == "Mercedes-Benz"),
            'Brand_Mitsubishi': int(brand == "Mitsubishi"),
            'Brand_Renault': int(brand == "Renault"),
            'Brand_Toyota': int(brand == "Toyota"),
            'Brand_Volkswagen': int(brand == "Volkswagen"),
            'Body_hatch': int(body == "hatch"),
            'Body_other': int(body == "other"),
            'Body_sedan': int(body == "sedan"),
            'Body_vagon': int(body == "vagon"),
            'Body_van': int(body == "van"),
            'Engine Type_Gas': int(fuel_type == "Gas"),
            'Engine Type_Other': int(fuel_type == "Other"),
            'Engine Type_Petrol': int(fuel_type == "Petrol"),
            'Registration_yes': int(registered == "Yes")
        }
        
        # 3. FIXED: Generate the exact interaction keys your scaler expects (No 'Log_' strings)
        raw_feature_map['With_Mileage'] = scaled_mileage  # Helper tag if matching training
        raw_feature_map['Mileage_x_BMW'] = scaled_mileage * raw_feature_map['Brand_BMW']
        raw_feature_map['Mileage_x_Mercedes'] = scaled_mileage * raw_feature_map['Brand_Mercedes-Benz']
        raw_feature_map['Mileage_x_Renault'] = scaled_mileage * raw_feature_map['Brand_Renault']
        
        # Convert dictionary to DataFrame
        df_user_input = pd.DataFrame([raw_feature_map])
        
        # 4. CRITICAL SAFEGUARD: Auto-sort data columns to match training order perfectly
        if hasattr(production_scaler, 'feature_names_in_'):
            # Filters and orders columns based exactly on the scaler's training memory layout
            valid_cols = [c for c in production_scaler.feature_names_in_ if c in df_user_input.columns]
            df_user_input = df_user_input[valid_cols]
        
        # 5. Core Pipeline Inference Execution
        try:
            scaled_user_input = production_scaler.transform(df_user_input)
            log_price_prediction = production_model.predict(scaled_user_input)
            calculated_market_value = np.exp(log_price_prediction)[0]
            
            st.markdown("---")
            st.success(f"### 🎯 Calculated Market Price Valuation: **${calculated_market_value:,.2f}**")
        except Exception as e:
            st.error(f"🚨 Matrix Transformation Error: {e}")