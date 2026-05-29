import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set page configuration with a custom title and beautiful car icon
st.set_page_config(
    page_title="Used Car Resale Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using glassmorphism and modern HSL colors
st.markdown("""
<style>
    /* Dark elegant gradient background for the whole page */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(9, 13, 26) 90.2%);
        color: #e2e8f0;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Elegant side panel style */
    section[data-testid="stSidebar"] {
        background-color: rgba(17, 24, 39, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Styled header card with glowing gradient text */
    .main-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.4) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    .main-header h1 {
        font-weight: 800;
        background: linear-gradient(90deg, #ff79c6, #f1fa8c, #bd93f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        font-size: 2.8rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 0;
    }
    
    /* Custom cards for displaying predictions */
    .prediction-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(4px);
    }
    .resale-value {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff79c6, #f1fa8c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 10px;
        margin-bottom: 10px;
        display: inline-block;
        letter-spacing: 1px;
    }
    
    /* Style for section headers */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
        border-bottom: 2px solid rgba(189, 147, 249, 0.3);
        padding-bottom: 5px;
        color: #bd93f9;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model payload
@st.cache_resource
def load_pipeline():
    model_path = os.path.join("models", "car_price_model.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model_payload = load_pipeline()

# Load used car dataset
@st.cache_data
def load_dataset():
    csv_path = os.path.join("data", "car_data.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

df_cars = load_dataset()

# Page Header
st.markdown("""
    <div class="main-header">
        <h1>🚗 Used Car Resale Value Estimator</h1>
        <p>A premium machine learning regression engine that calculates resale value of cars using high-fidelity Gradient Boosting Regressor algorithms.</p>
    </div>
""", unsafe_allow_html=True)

if model_payload is None:
    st.error("⚠️ Serialized model file not found. Please run 'python src/train.py' in your terminal first to train the regressors!")
    st.stop()

# ----------------- SIDEBAR - CONTROLS -----------------
st.sidebar.markdown("""
    <div style='text-align: center; padding-bottom: 10px;'>
        <h2 style='color: #bd93f9; margin-bottom: 5px;'>⚙️ Vehicle Features</h2>
        <p style='color: #64748b; font-size: 0.9rem;'>Adjust features to estimate the resale value.</p>
    </div>
""", unsafe_allow_html=True)

# Feature sliders
present_price = st.sidebar.slider("Current Showroom Price (Lakhs)", min_value=0.3, max_value=95.0, value=6.0, step=0.1)
kms_driven = st.sidebar.slider("Total Distance Driven (Kms)", min_value=500, max_value=500000, value=30000, step=500)
car_year = st.sidebar.slider("Manufacturing Year", min_value=2003, max_value=2026, value=2020, step=1)
car_age = 2026 - car_year

fuel_type = st.sidebar.selectbox("Fuel Type:", ["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("Seller Type:", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission Type:", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Previous Owners count:", [0, 1, 3])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model Parameters:**")
st.sidebar.markdown(f"**Algorithm:** {model_payload['model_name']}")
st.sidebar.markdown(f"**Dataset accuracy (R²):** {model_payload['metrics']['r2']:.2%}")

# ----------------- MAIN LAYOUT - TABS -----------------
tab_predict, tab_eda = st.tabs(["🔮 Resale Price Predictor", "📊 Dataset Explorer & Statistics"])

with tab_predict:
    col_pred, col_viz = st.columns([1, 1.2])
    
    with col_pred:
        st.markdown('<div class="section-title">🔮 Machine Learning Valuation</div>', unsafe_allow_html=True)
        
        # Prepare input dataframe
        input_df = pd.DataFrame([{
            'Present_Price': float(present_price),
            'Kms_Driven': int(kms_driven),
            'Fuel_Type': str(fuel_type),
            'Seller_Type': str(seller_type),
            'Transmission': str(transmission),
            'Owner': int(owner),
            'Car_Age': int(car_age)
        }])
        
        pipeline = model_payload["pipeline"]
        predicted_price = pipeline.predict(input_df)[0]
        predicted_price = max(0.0, predicted_price) # Resale price can't be negative
        
        # Calculate typical uncertainty range based on model MAE (0.56 Lakhs)
        mae = model_payload["metrics"]["mae"]
        min_range = max(0.0, predicted_price - mae)
        max_range = predicted_price + mae
        
        st.markdown(f"""
            <div class="prediction-card">
                <span style="font-size: 1.1rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Estimated Resale Price</span><br/>
                <div class="resale-value">
                     ₹ {predicted_price:.2f} Lakhs
                </div>
                <div style="margin-top: 15px; font-size: 0.95rem; color: #94a3b8; line-height: 1.5;">
                    Estimated Market Value Range:<br/>
                    <strong style="color: #f1fa8c; font-size: 1.1rem;">₹ {min_range:.2f} Lakhs - ₹ {max_range:.2f} Lakhs</strong><br/>
                    (Derived using average model deviation margin of ± {mae:.2f} Lakhs)
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Feature importances
        st.markdown('<div class="section-title">📊 Key Value Drivers</div>', unsafe_allow_html=True)
        importances = model_payload["feature_importances"]
        if importances:
            imp_df = pd.DataFrame({
                "Feature": [x.replace("num__", "").replace("cat__", "").replace("_", " ").title() for x in importances.keys()],
                "Importance": list(importances.values())
            }).sort_values("Importance", ascending=True)
            
            fig_imp, ax_imp = plt.subplots(figsize=(6, 3.2), facecolor='none')
            ax_imp.set_facecolor('none')
            
            # Highlight dominant feature
            colors = ['#818cf8' if x != imp_df['Importance'].max() else '#ff79c6' for x in imp_df['Importance']]
            
            bars = ax_imp.barh(imp_df["Feature"], imp_df["Importance"], color=colors, height=0.55, edgecolor='rgba(255, 255, 255, 0.1)')
            
            ax_imp.spines['top'].set_visible(False)
            ax_imp.spines['right'].set_visible(False)
            ax_imp.spines['bottom'].set_visible(False)
            ax_imp.spines['left'].set_color('#444')
            ax_imp.tick_params(colors='#e2e8f0', labelsize=9)
            ax_imp.set_xlim(0, max(imp_df["Importance"]) * 1.15)
            ax_imp.xaxis.grid(True, linestyle='--', alpha=0.1, color='#fff')
            
            for bar in bars:
                width = bar.get_width()
                ax_imp.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.1%}", 
                             ha='left', va='center', color='#e2e8f0', fontweight='bold', fontsize=9)
                             
            plt.tight_layout()
            st.pyplot(fig_imp)
        else:
            st.write("Feature importances not available.")
            
    with col_viz:
        st.markdown('<div class="section-title">✨ Price Mapping in Showroom vs. Resale Space</div>', unsafe_allow_html=True)
        st.write("Understand the pricing structure. Your current selected configuration is mapped as the **large gold star ⭐**.")
        
        if df_cars is not None:
            fig_scatter, ax_s = plt.subplots(figsize=(7.5, 5.0), facecolor='none')
            ax_s.set_facecolor('rgba(255, 255, 255, 0.01)')
            
            # Plot the actual dataset scatter
            sns.scatterplot(
                ax=ax_s,
                data=df_cars,
                x="Present_Price",
                y="Selling_Price",
                hue="Transmission",
                palette={'Manual': '#818cf8', 'Automatic': '#ff79c6'},
                alpha=0.6,
                s=60,
                edgecolor='rgba(0, 0, 0, 0.2)'
            )
            
            # Plot the user's predicted used car resale spot
            ax_s.scatter(
                [present_price], [predicted_price],
                color='#f1fa8c',
                marker='*',
                s=380,
                edgecolors='#fff',
                linewidths=1.5,
                label='Your Prediction',
                zorder=10
            )
            
            ax_s.set_title("Resale Price vs. Ex-Showroom Price", color='#e2e8f0', fontsize=11, fontweight='bold')
            ax_s.set_xlabel("Present Showroom Price (Lakhs)", color='#94a3b8', fontsize=9)
            ax_s.set_ylabel("Selling Resale Price (Lakhs)", color='#94a3b8', fontsize=9)
            ax_s.tick_params(colors='#888', labelsize=8)
            ax_s.legend(facecolor='#1e293b', edgecolor='rgba(255, 255, 255, 0.1)', labelcolor='#fff', fontsize=8)
            ax_s.spines['top'].set_visible(False)
            ax_s.spines['right'].set_visible(False)
            ax_s.spines['bottom'].set_color('#444')
            ax_s.spines['left'].set_color('#444')
            ax_s.grid(True, linestyle=':', alpha=0.1, color='#fff')
            
            plt.tight_layout()
            st.pyplot(fig_scatter)
        else:
            st.warning("Scatter plot could not be created because data file is missing.")

with tab_eda:
    st.markdown('<div class="section-title">📊 Dataset Explorer & Correlation Matrices</div>', unsafe_allow_html=True)
    
    if df_cars is not None:
        col_stats, col_corr = st.columns([1.2, 1])
        
        with col_stats:
            st.write("**used Car Resale Listings (Sample Table)**")
            st.dataframe(df_cars.head(10), height=250, use_container_width=True)
            
            st.write("**Descriptive Summary Statistics**")
            st.dataframe(df_cars.describe().T, use_container_width=True)
            
        with col_corr:
            st.write("**Correlation Matrix**")
            fig_corr, ax_co = plt.subplots(figsize=(6, 4.8), facecolor='none')
            ax_co.set_facecolor('rgba(0, 0, 0, 0.05)')
            
            numeric_df = df_cars.select_dtypes(include=[np.number])
            sns.heatmap(
                ax=ax_co,
                data=numeric_df.corr(),
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                linewidths=0.5,
                cbar=False
            )
            ax_co.set_title("Numeric Correlation Matrix", color='#e2e8f0', fontsize=12, fontweight='bold')
            ax_co.tick_params(colors='#888', labelsize=9)
            
            plt.tight_layout()
            st.pyplot(fig_corr)
    else:
        st.warning("Exploratory Data statistics cannot be shown because data file is missing.")
