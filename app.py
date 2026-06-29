import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Real Estate Price Predictor", layout="wide", page_icon="🏠")

st.title("🏠 Real Estate Price Predictor")
st.write("This app predicts the real estate price per unit area based on house age, distance to MRT, and nearby convenience stores.")
st.markdown("---")

# --- DATA LOADING & TRAINING (Cached for Performance) ---
@st.cache_data
def load_and_train():
    # Load dataset (Make sure the path is correct or file is in same directory)
    # Using your path or fallback to local file if deployed
    try:
        df = pd.read_csv(r"D:\college\Real_Estate_Price_Predictor\Real-Estate-Price-Predictor\RealEstate.csv")
    except FileNotFoundError:
        df = pd.read_csv("RealEstate.csv") # Fallback if you keep it in the same folder
        
    df.drop(columns=["X5 latitude","X6 longitude","X1 transaction date"], inplace=True, errors='ignore')
    df.rename(columns={"X2 house age":"age","X3 distance to the nearest MRT station":"distance_nearest_mrt_station",
                        "X4 number of convenience stores":"convinience_stores",
                        "Y house price of unit area":"price_per_unit_area"}, inplace=True)
    
    X = df[["age","distance_nearest_mrt_station","convinience_stores"]]
    y = df["price_per_unit_area"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Scaling
    scx = StandardScaler()
    X_train_scaled = scx.fit_transform(X_train)
    X_test_scaled = scx.transform(X_test)
    
    # Model Training
    lm = linear_model.LinearRegression()
    lm.fit(X_train_scaled, y_train)
    
    test_predict = lm.predict(X_test_scaled)
    
    return df, scx, lm, X_test, y_test, test_predict

# Initialize components
dataset, scx, lm, x_test, y_test, test_predict = load_and_train()

# --- LAYOUT SPLIT: LEFT FOR INPUT, RIGHT FOR OUTPUT ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("🔧 Enter House Details")
    
    # Interactive UI Controls instead of terminal inputs
    Age = st.slider("Age of the House (in years)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
    nmrts = st.number_input("Distance to nearest MRT station (in meters)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)
    stores = st.randint = st.slider("Number of convenience stores nearby", min_value=0, max_value=15, value=4)
    
    st.markdown("###")
    if st.button("🔮 Predict Property Price", type="primary"):
        # Prediction logic
        custom_input = np.array([[Age, nmrts, stores]])
        scaled_input = scx.transform(custom_input)
        prediction = lm.predict(scaled_input)[0]
        
        # Displaying result beautifully
        st.success(f"### Predicted Price: $**{prediction:.2f}** per unit area/ ₹{95*prediction:.2f}")

with col2:
    st.header("📊 Model Performance & Insights")
    
    # Tabs for clean look
    tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Feature Weights", "Dataset Preview"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=y_test, y=test_predict, ax=ax, color="#1f77b4")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', linewidth=2)
        ax.set_xlabel("Actual price")
        ax.set_ylabel("Predicted price")
        ax.set_title("Actual vs Predicted Price")
        st.pyplot(fig)
        
    with tab2:
        cols = ["age", "distance_nearest_mrt_station", "convinience_stores"]
        weight = lm.coef_
        weightcols = pd.DataFrame({"Features": cols, "Weight": weight})
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Weight", y="Features", data=weightcols, palette="deep", ax=ax2)
        ax2.set_title("Feature Weights (Impact)")
        st.pyplot(fig2)
        
    with tab3:
        st.write("Quick glance at the dataset used:")
        st.dataframe(dataset.head(10))
