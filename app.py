import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import re
import numpy as np
import json

# ----------------------
# Load trained model
# ----------------------
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------
# Load dataset
# ----------------------
df = pd.read_csv("used_cars_data_updated.csv")

# ----------------------
# Helper cleaning funcs
# ----------------------
def extract_number(value):
    if pd.isna(value):
        return np.nan
    num = re.findall(r"[0-9.]+", str(value))
    return float(num[0]) if num else np.nan

# Clean numeric columns
df["Mileage"] = df["Mileage"].apply(extract_number)
df["Engine"] = df["Engine"].apply(extract_number)
df["Power"] = df["Power"].apply(extract_number)

# ----------------------
# Split Name -> Brand, Model, Variant
# ----------------------
name_split = df["Name"].str.split(" ", n=2, expand=True)
df["Brand"], df["Model"], df["Variant"] = name_split[0], name_split[1], name_split[2]

brands = sorted(df["Brand"].dropna().unique())
models_by_brand = df.groupby("Brand")["Model"].unique().apply(list).to_dict()
variants_by_model = df.groupby("Model")["Variant"].unique().apply(list).to_dict()
locations = sorted(df["Location"].dropna().unique())

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Car Price Predictor", layout="centered")

# ----------------------
# Inject CSS for beauty
# ----------------------
st.markdown(
    """
    <style>
    /* Background with image */
    [data-testid="stAppViewContainer"] {
        background: url("https://thebossmagazine.com/wp-content/uploads/2024/01/obi-pixel8propix-JIcR3-O8ko8-unsplash-scaled.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        position: relative;
    }

    /* Dark overlay */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.65);
        z-index: 0;
    }

    /* Keep app content above */
    .stApp {
        position: relative;
        z-index: 1;
    }

    /* Headings */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif !important;
        color: #ffffff !important;
        font-weight: bold !important;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
    }

    /* General text and labels */
    .stMarkdown, .stInfo, label, .stSelectbox, .stSlider, .stRadio, .stNumberInput, .stText {
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    /* Table styling */
    table {
        color: #ffffff !important;
        background-color: rgba(0,0,0,0.5) !important;
    }
    th {
        background-color: #222 !important;
        color: #00e676 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Used Car Price Prediction")
st.write("Select car details below to estimate its price (in Lakhs).")

# ----------------------
# Inputs
# ----------------------
location = st.selectbox("Location", locations)
year = st.slider("Year of Manufacture", min_value=2007, max_value=2025, value=2015, step=1)
km_driven = st.slider("Kilometers Driven", min_value=0, max_value=300000, value=50000, step=1000)
fuel_type = st.selectbox("Fuel Type", sorted(df["Fuel_Type"].dropna().unique()))
transmission = st.selectbox("Transmission", sorted(df["Transmission"].dropna().unique()))
owner_type = st.selectbox("Owner Type", sorted(df["Owner_Type"].dropna().unique()))

brand = st.selectbox("Brand", brands)
model_name = st.selectbox("Model", models_by_brand.get(brand, []))
variant = st.selectbox("Variant", variants_by_model.get(model_name, []))

# ----------------------
# Dynamic sliders
# ----------------------
filtered_df = df[
    (df["Brand"] == brand) &
    (df["Model"] == model_name) &
    (df["Variant"] == variant)
]

max_mileage = float(filtered_df["Mileage"].max()) if not filtered_df["Mileage"].isna().all() else 50.0
max_engine = int(filtered_df["Engine"].max()) if not filtered_df["Engine"].isna().all() else 5000
max_power = float(filtered_df["Power"].max()) if not filtered_df["Power"].isna().all() else 600.0

mileage = st.slider("Mileage (kmpl or km/kg)", 5.0, max_mileage, min(18.0, max_mileage), 0.1)
engine = st.slider("Engine (CC)", 600, max_engine, min(1200, max_engine), 100)
power = st.slider("Power (bhp)", 20.0, max_power, min(80.0, max_power), 1.0)

# Seats auto-filled
if not filtered_df["Seats"].isna().all():
    seats_val = int(filtered_df["Seats"].mode()[0])
else:
    seats_val = 5
st.success(f"Seats: **{seats_val}**")

# ----------------------
# Prediction
# ----------------------
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        "Location": [location],
        "Kilometers_Driven": [np.log1p(km_driven)],
        "Fuel_Type": [fuel_type],
        "Transmission": [transmission],
        "Owner_Type": [owner_type],
        "Mileage": [mileage],
        "Engine": [engine],
        "Power": [power],
        "Seats": [seats_val],
        "New_Price": [None],
        "Brand": [brand],
        "Model": [model_name],
        "Variant": [variant],
        "Car_Age": [2025 - year]
    })

    prediction_log = model.predict(input_df)[0]
    prediction = np.expm1(prediction_log)

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #00c853, #00e676);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            font-size: 36px;
            font-weight: 900;
            color: #ffffff;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
            box-shadow: 0px 6px 25px rgba(0,0,0,0.8);
            margin-top: 20px;
        ">
             Estimated Price: ₹ {prediction:.2f} Lakhs
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------
# Accuracy Visualization
# ----------------------
st.subheader("Model Accuracy")

try:
    with open("model_metrics.json", "r") as f:
        results = json.load(f)

    model_names = list(results.keys())
    r2_scores = [v["R2"] * 100 for v in results.values()]

    fig, ax = plt.subplots()
    ax.bar(model_names, r2_scores )
    for i, txt in enumerate(r2_scores):
        ax.text(i, r2_scores[i] + 1, f"{txt:.1f}%", ha='center', fontsize=10, color="black")
    ax.set_ylabel("R² Score (%)")
    ax.set_title("Model Accuracy (R² %)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    st.write("### Accuracy Table")
    metrics_df = pd.DataFrame(results).T
    metrics_df["R2"] = metrics_df["R2"] * 100
    st.table(metrics_df.round(2))

except FileNotFoundError:
    st.warning(" Run train_model.py first to generate model_metrics.json")
