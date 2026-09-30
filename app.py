import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Customer Personality Classification",
    page_icon="👥",
    layout="wide"
)


# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("model/customer_personality_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
feature_names = joblib.load("model/feature_names.pkl")


# -----------------------------
# Title
# -----------------------------

st.title("Customer Personality Classification")

st.write(
    "Enter customer information to predict the customer's personality segment."
)

st.divider()


# -----------------------------
# Customer Information
# -----------------------------

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

with col2:
    income = st.number_input(
        "Income",
        min_value=0,
        value=50000
    )

with col3:
    mnt_wines = st.number_input(
        "Wine Spending",
        min_value=0,
        value=200
    )


col4, col5, col6 = st.columns(3)

with col4:
    mnt_fruits = st.number_input(
        "Fruit Spending",
        min_value=0,
        value=30
    )

with col5:
    mnt_meat = st.number_input(
        "Meat Products Spending",
        min_value=0,
        value=70
    )

with col6:
    mnt_fish = st.number_input(
        "Fish Products Spending",
        min_value=0,
        value=40
    )


col7, col8, col9 = st.columns(3)

with col7:
    mnt_sweet = st.number_input(
        "Sweet Products Spending",
        min_value=0,
        value=20
    )

with col8:
    mnt_gold = st.number_input(
        "Gold Products Spending",
        min_value=0,
        value=30
    )

with col9:
    num_deals = st.number_input(
        "Deals Purchases",
        min_value=0,
        value=3
    )


col10, col11, col12 = st.columns(3)

with col10:
    num_web = st.number_input(
        "Web Purchases",
        min_value=0,
        value=4
    )

with col11:
    num_catalog = st.number_input(
        "Catalog Purchases",
        min_value=0,
        value=5
    )

with col12:
    num_store = st.number_input(
        "Store Purchases",
        min_value=0,
        value=6
    )


num_visits = st.number_input(
    "Web Visits Per Month",
    min_value=0,
    value=4
)


# -----------------------------
# Prediction
# -----------------------------

st.divider()

if st.button("Predict Customer Segment", type="primary"):

    input_data = pd.DataFrame([[
        age,
        income,
        mnt_wines,
        mnt_fruits,
        mnt_meat,
        mnt_fish,
        mnt_sweet,
        mnt_gold,
        num_deals,
        num_web,
        num_catalog,
        num_store,
        num_visits
    ]], columns=feature_names)

    prediction = model.predict(input_data)

    predicted_class = label_encoder.inverse_transform(prediction)[0]

    st.success(
        f"Predicted Customer Segment: {predicted_class}"
    )

    st.subheader("Prediction Details")

    st.write(
        f"The customer has been classified as **{predicted_class}**."
    )