import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set page configuration with a custom title and beautiful flower icon
st.set_page_config(
    page_title="Iris Flower Species Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using glassmorphism and modern HSL colors
st.markdown("""
<style>
    /* Dark elegant gradient background for the whole page */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(18, 18, 32) 0%, rgb(9, 9, 14) 90.2%);
        color: #e0e0e6;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Elegant side panel style */
    section[data-testid="stSidebar"] {
        background-color: rgba(22, 22, 38, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Styled header card with glowing gradient text */
    .main-header {
        background: linear-gradient(135deg, rgba(82, 36, 126, 0.2) 0%, rgba(36, 52, 126, 0.2) 100%);
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
        background: linear-gradient(90deg, #ff79c6, #8be9fd, #bd93f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        font-size: 2.8rem;
    }
    .main-header p {
        color: #a0a0b8;
        font-size: 1.1rem;
        margin-bottom: 0;
    }
    
    /* Custom cards for displaying predictions */
    .prediction-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(4px);
    }
    .species-badge {
        font-size: 2.2rem;
        font-weight: 700;
        padding: 10px 24px;
        border-radius: 50px;
        display: inline-block;
        margin-top: 15px;
        letter-spacing: 1px;
    }
    .setosa {
        background: linear-gradient(135deg, rgba(255, 121, 198, 0.2) 0%, rgba(255, 121, 198, 0.4) 100%);
        color: #ff79c6;
        border: 1px solid rgba(255, 121, 198, 0.5);
        box-shadow: 0 0 15px rgba(255, 121, 198, 0.2);
    }
    .versicolor {
        background: linear-gradient(135deg, rgba(139, 233, 253, 0.2) 0%, rgba(139, 233, 253, 0.4) 100%);
        color: #8be9fd;
        border: 1px solid rgba(139, 233, 253, 0.5);
        box-shadow: 0 0 15px rgba(139, 233, 253, 0.2);
    }
    .virginica {
        background: linear-gradient(135deg, rgba(189, 147, 249, 0.2) 0%, rgba(189, 147, 249, 0.4) 100%);
        color: #bd93f9;
        border: 1px solid rgba(189, 147, 249, 0.5);
        box-shadow: 0 0 15px rgba(189, 147, 249, 0.2);
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

# Helper function to load model and scaler
@st.cache_resource
def load_pipeline():
    model_path = os.path.join("models", "iris_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model_payload = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model_payload, scaler

model_payload, scaler = load_pipeline()

# Load the local dataset for visualizations
@st.cache_data
def load_dataset():
    csv_path = os.path.join("data", "iris_dataset.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

df_iris = load_dataset()

# Page Header
st.markdown("""
    <div class="main-header">
        <h1>🌸 Iris Flower Species Classifier</h1>
        <p>A premium machine learning dashboard that predicts the species of Iris flowers with stellar precision and visualizes your custom measurements in real-time.</p>
    </div>
""", unsafe_allow_html=True)

if model_payload is None:
    st.error("⚠️ Pre-trained model or scaler could not be loaded. Please run 'python src/train.py' first in your terminal to generate files!")
    st.stop()

# ----------------- SIDEBAR - INPUT CONTROLS -----------------
st.sidebar.markdown("""
    <div style='text-align: center; padding-bottom: 10px;'>
        <h2 style='color: #bd93f9; margin-bottom: 5px;'>🌸 Input Features</h2>
        <p style='color: #888; font-size: 0.9rem;'>Adjust the sliders to specify the measurements in centimeters (cm).</p>
    </div>
""", unsafe_allow_html=True)

# Sliders configured with the typical range of the Iris dataset features
sepal_length = st.sidebar.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1, step=0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.5, step=0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4, step=0.1)
petal_width = st.sidebar.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model Details:**")
st.sidebar.markdown(f"**Algorithm:** {model_payload['model_name']}")
st.sidebar.markdown(f"**Dataset size:** 150 instances")

# ----------------- MAIN LAYOUT - TABS -----------------
tab_predict, tab_eda = st.tabs(["🔮 Live Prediction & Space Mapping", "📊 Exploratory Data Analysis (EDA)"])

with tab_predict:
    col_pred, col_viz = st.columns([1, 1.2])
    
    with col_pred:
        st.markdown('<div class="section-title">🔮 Machine Learning Prediction</div>', unsafe_allow_html=True)
        
        # Prepare data vector
        input_vector = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_vector)
        
        # Model Prediction
        model = model_payload["model"]
        target_names = model_payload["target_names"]
        
        pred_idx = model.predict(input_scaled)[0]
        predicted_species = target_names[pred_idx].capitalize()
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Display Prediction result with clean styled badge
        badge_class = predicted_species.lower()
        st.markdown(f"""
            <div class="prediction-card">
                <span style="font-size: 1.1rem; color: #a0a0b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Predicted Species</span><br/>
                <div class="species-badge {badge_class}">
                     Iris {predicted_species}
                </div>
                <div style="margin-top: 15px; font-size: 0.95rem; color: #a0a0b8;">
                    Classification based on the custom user parameters.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Probabilities chart
        st.markdown('<div class="section-title">📊 Confidence Breakdown</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Species": [x.capitalize() for x in target_names],
            "Probability": probabilities
        })
        
        fig_prob, ax_prob = plt.subplots(figsize=(6, 2.5), facecolor='none')
        ax_prob.set_facecolor('none')
        
        colors = ['#ff79c6', '#8be9fd', '#bd93f9']
        bars = ax_prob.barh(prob_df["Species"], prob_df["Probability"], color=colors, height=0.45, edgecolor='rgba(255, 255, 255, 0.15)')
        
        # Customize styling of matplotlib chart to blend with dark page
        ax_prob.spines['top'].set_visible(False)
        ax_prob.spines['right'].set_visible(False)
        ax_prob.spines['bottom'].set_visible(False)
        ax_prob.spines['left'].set_color('#444')
        ax_prob.tick_params(colors='#e0e0e6', labelsize=10)
        ax_prob.set_xlim(0, 1.05)
        ax_prob.xaxis.grid(True, linestyle='--', alpha=0.1, color='#fff')
        
        for bar in bars:
            width = bar.get_width()
            ax_prob.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width:.1%}", 
                         ha='left', va='center', color='#e0e0e6', fontweight='bold', fontsize=10)
                         
        plt.tight_layout()
        st.pyplot(fig_prob)
        
    with col_viz:
        st.markdown('<div class="section-title">✨ Input Mapping in Feature Space</div>', unsafe_allow_html=True)
        st.write("See where your live inputs (represented by the **large golden star ⭐**) fall relative to the actual Iris flower training data distribution.")
        
        if df_iris is not None:
            # Let's create two subplots: Petal Scatter and Sepal Scatter
            fig_scatter, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor='none')
            
            # Subplot 1: Petal Length vs Width
            ax1.set_facecolor('rgba(255, 255, 255, 0.01)')
            colors_palette = {'setosa': '#ff79c6', 'versicolor': '#8be9fd', 'virginica': '#bd93f9'}
            
            sns.scatterplot(
                ax=ax1,
                data=df_iris,
                x="petal length (cm)",
                y="petal width (cm)",
                hue="species",
                palette=colors_palette,
                alpha=0.6,
                s=60,
                edgecolor='rgba(0, 0, 0, 0.2)'
            )
            # Plot the user inputs
            ax1.scatter(
                [petal_length], [petal_width],
                color='#ffb86c',
                marker='*',
                s=350,
                edgecolors='#fff',
                linewidths=1.5,
                label='Your Input',
                zorder=10
            )
            ax1.set_title("Petal Length vs Width", color='#e0e0e6', fontsize=11, fontweight='bold')
            ax1.set_xlabel("Petal Length (cm)", color='#a0a0b8', fontsize=9)
            ax1.set_ylabel("Petal Width (cm)", color='#a0a0b8', fontsize=9)
            ax1.tick_params(colors='#888', labelsize=8)
            ax1.legend(facecolor='#1e1e2e', edgecolor='rgba(255, 255, 255, 0.15)', labelcolor='#fff', fontsize=8)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_color('#444')
            ax1.spines['left'].set_color('#444')
            ax1.grid(True, linestyle=':', alpha=0.1, color='#fff')
            
            # Subplot 2: Sepal Length vs Width
            ax2.set_facecolor('rgba(255, 255, 255, 0.01)')
            sns.scatterplot(
                ax=ax2,
                data=df_iris,
                x="sepal length (cm)",
                y="sepal width (cm)",
                hue="species",
                palette=colors_palette,
                alpha=0.6,
                s=60,
                edgecolor='rgba(0, 0, 0, 0.2)',
                legend=False
            )
            # Plot the user inputs
            ax2.scatter(
                [sepal_length], [sepal_width],
                color='#ffb86c',
                marker='*',
                s=350,
                edgecolors='#fff',
                linewidths=1.5,
                zorder=10
            )
            ax2.set_title("Sepal Length vs Width", color='#e0e0e6', fontsize=11, fontweight='bold')
            ax2.set_xlabel("Sepal Length (cm)", color='#a0a0b8', fontsize=9)
            ax2.set_ylabel("Sepal Width (cm)", color='#a0a0b8', fontsize=9)
            ax2.tick_params(colors='#888', labelsize=8)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_color('#444')
            ax2.spines['left'].set_color('#444')
            ax2.grid(True, linestyle=':', alpha=0.1, color='#fff')
            
            plt.tight_layout()
            st.pyplot(fig_scatter)
        else:
            st.warning("Scatter plot could not be created because data file is missing.")

with tab_eda:
    st.markdown('<div class="section-title">📊 Dataset Explorer & Statistics</div>', unsafe_allow_html=True)
    
    if df_iris is not None:
        col_stats, col_dist = st.columns([1.2, 1])
        
        with col_stats:
            st.write("**Iris Flower Dataset (Full Table Preview)**")
            st.dataframe(df_iris.drop(columns=['target']).head(10), height=250, use_container_width=True)
            
            st.write("**Descriptive Summary Statistics**")
            st.dataframe(df_iris.drop(columns=['target', 'species']).describe().T, use_container_width=True)
            
        with col_dist:
            st.write("**Feature Distributions (Violin Analysis)**")
            selected_feature = st.selectbox("Select feature to view density distribution:", 
                                            ["petal length (cm)", "petal width (cm)", "sepal length (cm)", "sepal width (cm)"])
            
            fig_violin, ax_v = plt.subplots(figsize=(6, 4.3), facecolor='none')
            ax_v.set_facecolor('rgba(0, 0, 0, 0.05)')
            
            colors_palette = {'setosa': '#ff79c6', 'versicolor': '#8be9fd', 'virginica': '#bd93f9'}
            sns.violinplot(
                ax=ax_v,
                x="species",
                y=selected_feature,
                data=df_iris,
                palette=colors_palette,
                density_norm='count'
            )
            ax_v.set_title(f"Density Distribution of {selected_feature.title()}", color='#e0e0e6', fontsize=12, fontweight='bold')
            ax_v.set_xlabel("Iris Species", color='#a0a0b8', fontsize=10)
            ax_v.set_ylabel(selected_feature.title(), color='#a0a0b8', fontsize=10)
            ax_v.tick_params(colors='#888', labelsize=9)
            ax_v.spines['top'].set_visible(False)
            ax_v.spines['right'].set_visible(False)
            ax_v.spines['bottom'].set_color('#444')
            ax_v.spines['left'].set_color('#444')
            
            plt.tight_layout()
            st.pyplot(fig_violin)
    else:
        st.warning("Exploratory Data analysis statistics cannot be shown because data file is missing.")
