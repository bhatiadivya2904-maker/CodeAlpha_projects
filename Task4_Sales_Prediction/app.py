import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set page configuration with a custom title and beautiful line chart icon
st.set_page_config(
    page_title="High-Fidelity Sales Predictor & Spend Optimizer",
    page_icon="📈",
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
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
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
    .sales-value {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
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
        border-bottom: 2px solid rgba(129, 140, 248, 0.3);
        padding-bottom: 5px;
        color: #818cf8;
    }

    /* Elegant optimizer cards */
    .optimizer-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(129, 140, 248, 0.2);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model payload
@st.cache_resource
def load_pipeline():
    model_path = os.path.join("models", "sales_prediction_model.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model_payload = load_pipeline()

# Load advertising dataset
@st.cache_data
def load_dataset():
    csv_path = os.path.join("data", "advertising.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        return df
    return None

df_adv = load_dataset()

# Page Header
st.markdown("""
    <div class="main-header">
        <h1>📈 High-Fidelity Sales Predictor & Spend Optimizer</h1>
        <p>An advanced AI modeling engine that forecasts sales and optimizes marketing budgets across TV, Radio, and Newspaper channels to maximize ROI.</p>
    </div>
""", unsafe_allow_html=True)

if model_payload is None:
    st.error("⚠️ Serialized model file not found. Please run 'python src/train.py' in your terminal first to train the regressors!")
    st.stop()

# Extract model metadata
best_model_name = model_payload["model_name"]
metrics = model_payload["metrics"]
stats = model_payload["dataset_stats"]
elasticities = model_payload["linear_elasticities"]
importances = model_payload["feature_importances"]

# ----------------- SIDEBAR - CONTROLS -----------------
st.sidebar.markdown("""
    <div style='text-align: center; padding-bottom: 10px;'>
        <h2 style='color: #818cf8; margin-bottom: 5px;'>⚙️ Campaign Budgets</h2>
        <p style='color: #64748b; font-size: 0.9rem;'>Adjust budgets to forecast product sales.</p>
    </div>
""", unsafe_allow_html=True)

# Sliders configured to match or exceed slightly the dataset bounds
tv_budget = st.sidebar.slider(
    "📺 TV Budget ($ thousands)",
    min_value=0.0,
    max_value=350.0,
    value=float(stats["TV"]["mean"]),
    step=1.0,
    help="Advertising budget spent on television."
)
radio_budget = st.sidebar.slider(
    "📻 Radio Budget ($ thousands)",
    min_value=0.0,
    max_value=60.0,
    value=float(stats["Radio"]["mean"]),
    step=0.5,
    help="Advertising budget spent on radio."
)
news_budget = st.sidebar.slider(
    "📰 Newspaper Budget ($ thousands)",
    min_value=0.0,
    max_value=150.0,
    value=float(stats["Newspaper"]["mean"]),
    step=1.0,
    help="Advertising budget spent on print newspaper."
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**🤖 Predictive Model Info:**")
st.sidebar.markdown(f"**Best Algorithm:** `{best_model_name}`")
st.sidebar.markdown(f"**Accuracy (R² Score):** `{metrics['r2']:.2%}`")
st.sidebar.markdown(f"**Average Deviation (MAE):** `{metrics['mae']:.3f}`k units")

# ----------------- MAIN LAYOUT - TABS -----------------
tab_predict, tab_optimize, tab_eda, tab_benchmark = st.tabs([
    "🔮 Sales Predictor",
    "💰 Spend Optimizer",
    "📊 Exploratory Analytics",
    "⚡ Model Benchmarking"
])

# ----------------- TAB 1: PREDICTOR -----------------
with tab_predict:
    col_pred, col_viz = st.columns([1, 1.2])
    
    with col_pred:
        st.markdown('<div class="section-title">🔮 Machine Learning Forecast</div>', unsafe_allow_html=True)
        
        # Prepare input dataframe
        input_df = pd.DataFrame([{
            'TV': float(tv_budget),
            'Radio': float(radio_budget),
            'Newspaper': float(news_budget)
        }])
        
        pipeline = model_payload["pipeline"]
        predicted_sales = pipeline.predict(input_df)[0]
        predicted_sales = max(0.0, predicted_sales) # Sales cannot be negative
        
        # Typical uncertainty range based on model MAE
        mae = metrics["mae"]
        min_range = max(0.0, predicted_sales - mae)
        max_range = predicted_sales + mae
        
        st.markdown(f"""
            <div class="prediction-card">
                <span style="font-size: 1.1rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Predicted Sales Volume</span><br/>
                <div class="sales-value">
                     {predicted_sales:.2f}k Units
                </div>
                <div style="margin-top: 15px; font-size: 0.95rem; color: #94a3b8; line-height: 1.5;">
                    Estimated Sales Range (with MAE margin of ±{mae:.2f}k units):<br/>
                    <strong style="color: #38bdf8; font-size: 1.25rem;">{min_range:.2f}k - {max_range:.2f}k units</strong><br/>
                    <span style="color: #64748b; font-size: 0.85rem;">Estimated Revenue: <strong>${predicted_sales * 10.0:.2f}k</strong> (at standard $10 per unit value)</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Gauge interpretation
        sales_pct = (predicted_sales - stats["Sales"]["min"]) / (stats["Sales"]["max"] - stats["Sales"]["min"])
        sales_pct = min(max(sales_pct, 0.0), 1.0)
        
        st.markdown("<br/>", unsafe_allow_html=True)
        st.write("**Forecast Strength Indicator**")
        st.progress(sales_pct)
        
        if sales_pct < 0.33:
            st.info("📉 **Conservative Market Capture:** Budgets are low. Model indicates a standard baseline response.")
        elif sales_pct < 0.66:
            st.warning("📈 **Moderate Market Traction:** Healthy spending drives strong volume. Scaling further is viable.")
        else:
            st.success("🔥 **High Market Domination:** Aggressive spending unlocks near-peak saturation sales limits!")
            
    with col_viz:
        st.markdown('<div class="section-title">📊 Key Value Drivers (Relative Importance)</div>', unsafe_allow_html=True)
        st.write("Understand the predictive weight assigned to each channel. **TV** and **Radio** dominate conversion dynamics.")
        
        if importances:
            imp_df = pd.DataFrame({
                "Channel": list(importances.keys()),
                "Weight": list(importances.values())
            }).sort_values("Weight", ascending=True)
            
            fig_imp, ax_imp = plt.subplots(figsize=(7, 4), facecolor='none')
            ax_imp.set_facecolor('none')
            
            colors = ['#c084fc' if x == imp_df['Weight'].min() else ('#38bdf8' if x == imp_df['Weight'].max() else '#818cf8') for x in imp_df['Weight']]
            
            bars = ax_imp.barh(imp_df["Channel"], imp_df["Weight"], color=colors, height=0.5, edgecolor='rgba(255, 255, 255, 0.1)')
            
            ax_imp.spines['top'].set_visible(False)
            ax_imp.spines['right'].set_visible(False)
            ax_imp.spines['bottom'].set_visible(False)
            ax_imp.spines['left'].set_color('#444')
            ax_imp.tick_params(colors='#e2e8f0', labelsize=10)
            ax_imp.set_xlim(0, max(imp_df["Weight"]) * 1.15)
            ax_imp.xaxis.grid(True, linestyle='--', alpha=0.1, color='#fff')
            
            for bar in bars:
                width = bar.get_width()
                ax_imp.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.2%}", 
                             ha='left', va='center', color='#e2e8f0', fontweight='bold', fontsize=10)
                             
            plt.tight_layout()
            st.pyplot(fig_imp)
        else:
            st.write("Importance metrics not available.")

# ----------------- TAB 2: SPEND OPTIMIZER -----------------
with tab_optimize:
    st.markdown('<div class="section-title">💰 Smart Budget Spend Optimization</div>', unsafe_allow_html=True)
    st.write("""
        Maximize your sales yield by automatically distributing a fixed overall budget between the three media platforms.
        This algorithm calculates the **mathematically optimal spend profile** based on linear marketing elasticity coefficients.
    """)
    
    # User inputs their available marketing spend
    total_budget = st.number_input(
        "💵 Enter Total Target Marketing Budget ($ thousands)",
        min_value=10.0,
        max_value=500.0,
        value=150.0,
        step=5.0
    )
    
    # Constraints matching dataset boundaries
    st.markdown("##### Allocation Constraint Limits")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        max_tv = st.slider("Max TV Cap ($k)", 50.0, 350.0, 300.0)
    with col_c2:
        max_radio = st.slider("Max Radio Cap ($k)", 10.0, 60.0, 50.0)
    with col_c3:
        max_news = st.slider("Max Newspaper Cap ($k)", 10.0, 150.0, 115.0)

    # Greedy Bounded Linear Programming Solver
    # Objective: Maximize coefficients
    # Coefs are elasticities
    coef_dict = {
        "TV": elasticities["TV"],
        "Radio": elasticities["Radio"],
        "Newspaper": elasticities["Newspaper"]
    }
    
    # Sort channels by elasticity (rate of return)
    sorted_channels = sorted(coef_dict.items(), key=lambda x: x[1], reverse=True)
    
    limits = {
        "TV": max_tv,
        "Radio": max_radio,
        "Newspaper": max_news
    }
    
    # Solve greedy allocation
    allocated = {"TV": 0.0, "Radio": 0.0, "Newspaper": 0.0}
    remaining = total_budget
    
    for channel, coef in sorted_channels:
        cap = limits[channel]
        alloc = min(remaining, cap)
        allocated[channel] = alloc
        remaining -= alloc
        if remaining <= 0:
            break
            
    # Calculate baseline projection vs optimized projection
    # Standard baseline: Equal 1/3 split or proportional
    baseline_split = total_budget / 3.0
    baseline_tv = min(baseline_split, limits["TV"])
    baseline_radio = min(baseline_split, limits["Radio"])
    baseline_news = min(baseline_split, limits["Newspaper"])
    
    # Add any leftover from limits to other channels
    leftover = total_budget - (baseline_tv + baseline_radio + baseline_news)
    if leftover > 0:
        # greedily distribute leftover
        for c, _ in sorted_channels:
            avail = limits[c] - (baseline_tv if c=="TV" else (baseline_radio if c=="Radio" else baseline_news))
            add = min(leftover, avail)
            if c == "TV": baseline_tv += add
            elif c == "Radio": baseline_radio += add
            else: baseline_news += add
            leftover -= add
            if leftover <= 0: break
            
    # Evaluate in best model (Gradient Boosting)
    opt_df = pd.DataFrame([{
        "TV": allocated["TV"],
        "Radio": allocated["Radio"],
        "Newspaper": allocated["Newspaper"]
    }])
    base_df = pd.DataFrame([{
        "TV": baseline_tv,
        "Radio": baseline_radio,
        "Newspaper": baseline_news
    }])
    
    opt_sales = pipeline.predict(opt_df)[0]
    base_sales = pipeline.predict(base_df)[0]
    uplift = opt_sales - base_sales
    uplift_pct = (uplift / base_sales) if base_sales > 0 else 0.0

    col_res, col_chart = st.columns([1.1, 1])
    
    with col_res:
        st.markdown("<div class='optimizer-card'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Recommended Spend Strategy")
        
        # Display breakdown
        st.markdown(f"""
        * **📺 TV Allocation:** <strong style="color:#38bdf8">${allocated['TV']:.2f}k</strong> ({allocated['TV']/total_budget:.1%})
        * **📻 Radio Allocation:** <strong style="color:#818cf8">${allocated['Radio']:.2f}k</strong> ({allocated['Radio']/total_budget:.1%})
        * **📰 Newspaper Allocation:** <strong style="color:#c084fc">${allocated['Newspaper']:.2f}k</strong> ({allocated['Newspaper']/total_budget:.1%})
        """)
        
        # Display totals
        st.markdown(f"""
        <div style="margin-top:15px; border-top:1px solid rgba(255,255,255,0.1); padding-top:15px;">
            Predicted Optimized Sales: <strong style="color:#38bdf8; font-size:1.4rem;">{opt_sales:.2f}k units</strong><br/>
            Standard Baseline Sales (Even split): <span style="color:#94a3b8;">{base_sales:.2f}k units</span><br/>
            <span style="color:#22c55e; font-weight:700; font-size:1.15rem;">📈 Performance Uplift: +{uplift:.2f}k units (+{uplift_pct:.2%})</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.info(f"💡 **Strategic Advisory:** Radio yields the highest elasticity coefficient per unit spend (${elasticities['Radio']:.3f} sales units per $1k spend), followed by TV (${elasticities['TV']:.3f}). Newspaper yields negligible returns (${elasticities['Newspaper']:.3f}). The algorithm maxes out the high-yield Radio bucket first before scaling TV.")

    with col_chart:
        st.write("**Budget Comparison: Standard vs. Smart Optimized**")
        comparison_data = pd.DataFrame({
            "Channel": ["TV", "Radio", "Newspaper"] * 2,
            "Budget ($k)": [baseline_tv, baseline_radio, baseline_news, allocated["TV"], allocated["Radio"], allocated["Newspaper"]],
            "Strategy": ["Standard (Even)"] * 3 + ["AI Optimized"] * 3
        })
        
        fig_comp, ax_c = plt.subplots(figsize=(6, 4), facecolor='none')
        ax_c.set_facecolor('none')
        
        sns.barplot(
            ax=ax_c,
            data=comparison_data,
            x="Channel",
            y="Budget ($k)",
            hue="Strategy",
            palette={"Standard (Even)": "#64748b", "AI Optimized": "#818cf8"},
            edgecolor='rgba(255,255,255,0.1)'
        )
        ax_c.set_title("Budget Allocation Comparison", color='#e2e8f0', fontsize=11, fontweight='bold')
        ax_c.tick_params(colors='#888', labelsize=9)
        ax_c.set_xlabel("", color='#94a3b8')
        ax_c.set_ylabel("Budget ($ thousands)", color='#94a3b8', fontsize=9)
        ax_c.legend(facecolor='#1e293b', edgecolor='rgba(255, 255, 255, 0.1)', labelcolor='#fff', fontsize=8)
        ax_c.spines['top'].set_visible(False)
        ax_c.spines['right'].set_visible(False)
        ax_c.spines['bottom'].set_color('#444')
        ax_c.spines['left'].set_color('#444')
        
        plt.tight_layout()
        st.pyplot(fig_comp)

# ----------------- TAB 3: EXPLORATORY ANALYTICS -----------------
with tab_eda:
    st.markdown('<div class="section-title">📊 Dataset Explorer & Statistical Profiling</div>', unsafe_allow_html=True)
    
    if df_adv is not None:
        col_table, col_heatmap = st.columns([1.2, 1])
        
        with col_table:
            st.write("**Historical Campaigns (Sample Listings)**")
            st.dataframe(df_adv.head(10), height=250, use_container_width=True)
            
            st.write("**Descriptive Distribution Summary**")
            st.dataframe(df_adv.describe().T, use_container_width=True)
            
        with col_heatmap:
            st.write("**Correlation Matrix (Elasticity Indicators)**")
            fig_heatmap, ax_h = plt.subplots(figsize=(6, 4.5), facecolor='none')
            ax_h.set_facecolor('rgba(0,0,0,0.05)')
            
            sns.heatmap(
                ax=ax_h,
                data=df_adv.corr(),
                annot=True,
                fmt=".3f",
                cmap="Blues",
                linewidths=0.5,
                cbar=False
            )
            ax_h.set_title("Feature Correlations with Sales", color='#e2e8f0', fontsize=11, fontweight='bold')
            ax_h.tick_params(colors='#888', labelsize=9)
            
            plt.tight_layout()
            st.pyplot(fig_heatmap)
            
        st.markdown('<div class="section-title">📈 Media Channel Effectiveness Curves</div>', unsafe_allow_html=True)
        st.write("Visualizing the independent linear and non-linear trend curves between channel spend and sales outcomes.")
        
        fig_trends, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor='none')
        for idx, col in enumerate(["TV", "Radio", "Newspaper"]):
            ax_t = axes[idx]
            ax_t.set_facecolor('rgba(255,255,255,0.01)')
            
            sns.regplot(
                ax=ax_t,
                data=df_adv,
                x=col,
                y="Sales",
                scatter_kws={"color": "#38bdf8", "alpha": 0.4, "s": 20},
                line_kws={"color": "#818cf8", "linewidth": 2}
            )
            ax_t.set_title(f"{col} Spend vs. Sales", color='#e2e8f0', fontsize=10, fontweight='bold')
            ax_t.set_xlabel(f"{col} Budget ($ thousands)", color='#94a3b8', fontsize=8)
            ax_t.set_ylabel("Sales (thousand units)" if idx == 0 else "", color='#94a3b8', fontsize=8)
            ax_t.tick_params(colors='#888', labelsize=8)
            ax_t.spines['top'].set_visible(False)
            ax_t.spines['right'].set_visible(False)
            ax_t.spines['bottom'].set_color('#444')
            ax_t.spines['left'].set_color('#444')
            ax_t.grid(True, linestyle=':', alpha=0.1, color='#fff')
            
        plt.tight_layout()
        st.pyplot(fig_trends)
    else:
        st.warning("Exploratory metrics are unavailable because the CSV file was not found.")

# ----------------- TAB 4: BENCHMARKING -----------------
with tab_benchmark:
    st.markdown('<div class="section-title">⚡ Model Benchmarking & Performance Summary</div>', unsafe_allow_html=True)
    st.write("Compare the evaluation metrics of tested regressors. Non-linear ensemble models yield unmatched predictive accuracy.")
    
    # Comparison table data (extracted from train.py)
    benchmark_df = pd.DataFrame({
        "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest", "Gradient Boosting"],
        "R2 Score": [0.8994, 0.8988, 0.8983, 0.9813, 0.9831],
        "MAE": [1.4608, 1.4643, 1.4613, 0.6207, 0.6181],
        "RMSE": [1.7816, 1.7872, 1.7913, 0.7688, 0.7295]
    })
    
    col_bench, col_bench_chart = st.columns([1.1, 1])
    
    with col_bench:
        st.write("**Evaluation Statistics (Validation Set)**")
        # Format the R2 score beautifully
        formatted_df = benchmark_df.copy()
        formatted_df["R2 Score"] = formatted_df["R2 Score"].map(lambda x: f"{x:.2%}")
        st.dataframe(formatted_df, use_container_width=True)
        
        st.markdown("""
        > [!TIP]
        > **Why Gradient Boosting?**
        > The Gradient Boosting Regressor achieved the highest **$R^2$ Score of 98.31%**, reducing predictive error by over **50%** compared to traditional Linear Regression (from MAE 1.46 to 0.61). This captures synergistic interactions between TV and Radio advertising that linear models miss!
        """)
        
    with col_bench_chart:
        st.write("**Model Accuracy (R² Score Comparison)**")
        fig_bench, ax_b = plt.subplots(figsize=(6, 3.8), facecolor='none')
        ax_b.set_facecolor('none')
        
        sorted_bench = benchmark_df.sort_values("R2 Score", ascending=True)
        colors_bench = ['#38bdf8' if x == sorted_bench['R2 Score'].max() else '#64748b' for x in sorted_bench['R2 Score']]
        
        bars_b = ax_b.barh(sorted_bench["Model"], sorted_bench["R2 Score"], color=colors_bench, height=0.55, edgecolor='rgba(255,255,255,0.1)')
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)
        ax_b.spines['bottom'].set_visible(False)
        ax_b.spines['left'].set_color('#444')
        ax_b.tick_params(colors='#e2e8f0', labelsize=9)
        ax_b.set_xlim(0, 1.15)
        ax_b.xaxis.grid(True, linestyle='--', alpha=0.1, color='#fff')
        
        for bar in bars_b:
            width = bar.get_width()
            ax_b.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.2%}", 
                         ha='left', va='center', color='#e2e8f0', fontweight='bold', fontsize=9)
                         
        plt.tight_layout()
        st.pyplot(fig_bench)
