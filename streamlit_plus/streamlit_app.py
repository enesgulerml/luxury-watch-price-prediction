import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "final_pipeline.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_lgb_model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "Luxury watch.csv")

pipeline = joblib.load(PIPELINE_PATH)
model = joblib.load(MODEL_PATH)
df = pd.read_csv(CSV_PATH)

st.set_page_config(page_title="Luxury Watch Price Predictor", layout="wide", page_icon="⌚")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0D1B2A;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #00695C;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        transition: transform 0.1s;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        background-color: #a0002d;
    }

    div.stAlert {
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)



st.markdown(
    """
    <h1 style="
        color: #00695C;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        text-align: center;
    ">
        Luxury Watch Price Prediction
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("<p style='text-align: center;'>Enter the specifications of a watch and the model will give an estimate in USD.</p>",
            unsafe_allow_html=True)

st.write("---")

# Layout: Two Columns Form
col1, col2 = st.columns(2)

with col1:
    brand_options = df['Brand'].unique().tolist()
    brand = st.selectbox("Brand", brand_options)

    model_options = df['Model'].unique().tolist()
    model_name = st.selectbox("Model", model_options)

    case_material_options = df['Case Material'].unique().tolist()
    case_material = st.selectbox("Case Material", case_material_options)

    strap_material_options = df['Strap Material'].unique().tolist()
    strap_material = st.selectbox("Strap Material", strap_material_options)

    movement_type_options = df['Movement Type'].unique().tolist()
    movement_type = st.selectbox("Movement Type", movement_type_options)

    dial_color_options = df['Dial Color'].unique().tolist()
    dial_color = st.selectbox("Dial Color", dial_color_options)

with col2:
    crystal_material_options = df['Crystal Material'].unique().tolist()
    crystal_material = st.selectbox("Crystal Material", crystal_material_options)

    complications_options = df['Complications'].fillna("None").unique().tolist()
    complications = st.selectbox("Complications", complications_options)

    case_diameter = st.number_input("Case Diameter (mm)", min_value=20.0, max_value=60.0, value=40.0)
    case_thickness = st.number_input("Case Thickness (mm)", min_value=5.0, max_value=20.0, value=12.0)
    band_width = st.number_input("Band Width (mm)", min_value=10.0, max_value=30.0, value=20.0)
    water_resistance = st.number_input("Water Resistance (meters)", min_value=0, max_value=1000, value=300)
    power_reserve = st.number_input("Power Reserve (hours)", min_value=0, max_value=200, value=48)

# Prepare DataFrame for prediction
input_df = pd.DataFrame([{
    "Brand": brand,
    "Model": model_name,
    "Case Material": case_material,
    "Strap Material": strap_material,
    "Movement Type": movement_type,
    "Dial Color": dial_color,
    "Crystal Material": crystal_material,
    "Complications": complications,
    "Case Diameter (mm)": case_diameter,
    "Case Thickness (mm)": case_thickness,
    "Band Width (mm)": band_width,
    "Water Resistance": water_resistance,
    "Power Reserve": power_reserve
}])

st.write("---")

st.warning(
    "⚠️ This project is for educational purposes only and is not a real product. "
    "Price predictions should not be used for investment or purchasing decisions.",
)


if st.button("Guess the Price"):
    try:
        y_pred_log = pipeline.predict(input_df)

        y_pred_usd = np.expm1(y_pred_log)

        st.success(f"Estimated Price: **${y_pred_usd[0]:,.2f}**")

    except Exception as e:
        st.error(f"Error occurred: {e}")

# Footer
st.markdown("<p style='text-align: center; color: gray;'>Luxury Watch Price Predictor © 2025</p>",
            unsafe_allow_html=True)
