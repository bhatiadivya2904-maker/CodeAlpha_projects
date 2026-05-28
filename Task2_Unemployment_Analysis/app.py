import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configuration with a custom title and beautiful chart icon
st.set_page_config(
    page_title="Unemployment Rate Analysis Dashboard",
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
    
    /* Custom metric display cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(4px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
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
    
    /* Style policy card */
    .policy-card {
        background: rgba(129, 140, 248, 0.05);
        border-left: 4px solid #818cf8;
        padding: 15px;
        border-radius: 4px 12px 12px 4px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load datasets
@st.cache_data
def load_data():
    cleaned_path = os.path.join("data", "unemployment_cleaned.csv")
    india_path = os.path.join("data", "Unemployment in India.csv")
    
    df_cleaned = None
    df_india = None
    
    if os.path.exists(cleaned_path):
        df_cleaned = pd.read_csv(cleaned_path)
        df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
        
    if os.path.exists(india_path):
        df_india = pd.read_csv(india_path)
        # Strip columns
        df_india.columns = df_india.columns.str.strip()
        # Rename columns substring mapping
        rename_dict = {}
        for col in df_india.columns:
            if 'unemployment' in col.lower():
                rename_dict[col] = 'Unemployment_Rate'
            elif 'employed' in col.lower():
                rename_dict[col] = 'Employed'
            elif 'participation' in col.lower():
                rename_dict[col] = 'Labour_Participation_Rate'
        df_india = df_india.rename(columns=rename_dict)
        df_india = df_india.dropna(subset=['Region', 'Date'])
        df_india['Date'] = pd.to_datetime(df_india['Date'].str.strip(), dayfirst=True, errors='coerce')
        df_india = df_india.dropna(subset=['Date'])
        df_india['Area'] = df_india['Area'].astype(str).str.strip()
        
    return df_cleaned, df_india

df_cleaned, df_india = load_data()

# Page Header
st.markdown("""
    <div class="main-header">
        <h1>📈 Indian Unemployment Analysis Dashboard</h1>
        <p>A premium interactive portal exploring labor participation, economic trends, and the quantitative impact of the 2020 COVID-19 pandemic.</p>
    </div>
""", unsafe_allow_html=True)

if df_cleaned is None or df_india is None:
    st.error("⚠️ Datasets are missing. Please run 'python src/download_data.py' and 'python src/analyze.py' first to prepare files!")
    st.stop()

# ----------------- SIDEBAR - CONTROLS -----------------
st.sidebar.markdown("""
    <div style='text-align: center; padding-bottom: 10px;'>
        <h2 style='color: #818cf8; margin-bottom: 5px;'>⚙️ Dashboard Filters</h2>
        <p style='color: #64748b; font-size: 0.9rem;'>Customize the scope of states and dates.</p>
    </div>
""", unsafe_allow_html=True)

# State selection
states = sorted(df_cleaned['Region'].unique().tolist())
selected_state = st.sidebar.selectbox("Select State/Region:", ["All India"] + states)

# Filter dataframe based on state selection
if selected_state == "All India":
    df_state = df_cleaned.copy()
else:
    df_state = df_cleaned[df_cleaned['Region'] == selected_state].copy()

# Date sliders
min_date = df_cleaned['Date'].min().to_pydatetime()
max_date = df_cleaned['Date'].max().to_pydatetime()
date_range = st.sidebar.slider("Select Timeline Range:", min_value=min_date, max_value=max_date, 
                               value=(min_date, max_value), format="MMM YYYY")

# Apply date filter
df_state_filtered = df_state[(df_state['Date'] >= date_range[0]) & (df_state['Date'] <= date_range[1])]

st.sidebar.markdown("---")
st.sidebar.markdown("**Key Findings Summary:**")
st.sidebar.markdown("- **Strict Lockdown Peak:** Unemployment surged nationally to **23.76%** on average.")
st.sidebar.markdown("- **Highest Hit State:** Tripura & Haryana recorded averages above **25%**.")

# ----------------- MAIN LAYOUT - TABS -----------------
tab_trends, tab_covid, tab_regional = st.tabs([
    "📈 Interactive Trends & Metrics", 
    "🦠 COVID-19 Lockdown Impact", 
    "🏘️ Rural vs. Urban Job Divide"
])

# Define colors for dark theme consistency
colors_palette = ['#e15759', '#38bdf8', '#818cf8', '#fbbf24', '#34d399']

with tab_trends:
    # 1. Metric Cards Row
    avg_unemp = df_state_filtered['Unemployment_Rate'].mean()
    peak_unemp = df_state_filtered['Unemployment_Rate'].max()
    peak_row = df_state_filtered[df_state_filtered['Unemployment_Rate'] == peak_unemp]
    peak_time = peak_row['Date'].dt.strftime('%B %Y').values[0] if not peak_row.empty else "N/A"
    
    avg_participation = df_state_filtered['Labour_Participation_Rate'].mean()
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Average Unemployment Rate</div>
                <div class="metric-value" style="color: #38bdf8;">{avg_unemp:.2f}%</div>
                <div style="color: #64748b; font-size: 0.85rem;">Across selected period</div>
            </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Peak Unemployment Rate</div>
                <div class="metric-value" style="color: #f87171;">{peak_unemp:.2f}%</div>
                <div style="color: #64748b; font-size: 0.85rem;">Recorded in {peak_time}</div>
            </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Labor Participation Rate</div>
                <div class="metric-value" style="color: #34d399;">{avg_participation:.2f}%</div>
                <div style="color: #64748b; font-size: 0.85rem;">Active workforce size</div>
            </div>
        """, unsafe_allow_html=True)

    # 2. Plots Row
    st.markdown('<div class="section-title">📈 Labor & Job Market Trends Over Time</div>', unsafe_allow_html=True)
    col_t1, col_t2 = st.columns([1.2, 1])
    
    with col_t1:
        st.write(f"**Unemployment and Labor Participation Rate Timeline ({selected_state})**")
        fig_trend, ax_t = plt.subplots(figsize=(7.5, 4.2), facecolor='none')
        ax_t.set_facecolor('rgba(255, 255, 255, 0.01)')
        
        # Group by date for line plots
        time_data = df_state_filtered.groupby('Date')[['Unemployment_Rate', 'Labour_Participation_Rate']].mean().reset_index()
        time_data = time_data.sort_values('Date')
        
        ax_t.plot(time_data['Date'], time_data['Unemployment_Rate'], 
                  color='#f87171', marker='o', linewidth=2, label="Unemployment Rate (%)")
        ax_t.plot(time_data['Date'], time_data['Labour_Participation_Rate'], 
                  color='#34d399', marker='s', linewidth=2, label="Labour Participation (%)")
                  
        ax_t.set_title(f"Unemployment & Participation Trend - {selected_state}", color='#e2e8f0', fontsize=11, fontweight='bold')
        ax_t.tick_params(colors='#94a3b8', labelsize=8)
        ax_t.set_xlabel("Timeline Month", color='#94a3b8', fontsize=9)
        ax_t.set_ylabel("Percentage (%)", color='#94a3b8', fontsize=9)
        ax_t.legend(facecolor='#1e293b', edgecolor='rgba(255, 255, 255, 0.1)', labelcolor='#e2e8f0', fontsize=8)
        ax_t.spines['top'].set_visible(False)
        ax_t.spines['right'].set_visible(False)
        ax_t.spines['bottom'].set_color('#334155')
        ax_t.spines['left'].set_color('#334155')
        ax_t.grid(True, linestyle=':', alpha=0.1, color='#fff')
        
        st.pyplot(fig_trend)
        
    with col_t2:
        st.write("**Top 10 States by Highest Average Unemployment**")
        fig_bar, ax_b = plt.subplots(figsize=(6.2, 5.0), facecolor='none')
        ax_b.set_facecolor('rgba(0, 0, 0, 0.03)')
        
        state_order = df_cleaned.groupby('Region')['Unemployment_Rate'].mean().sort_values(ascending=False).head(10)
        
        sns.barplot(
            ax=ax_b,
            x=state_order.values,
            y=state_order.index,
            hue=state_order.index,
            palette="flare",
            legend=False
        )
        ax_b.set_title("States with Highest Unemployment (Average 2020)", color='#e2e8f0', fontsize=11, fontweight='bold')
        ax_b.tick_params(colors='#94a3b8', labelsize=8)
        ax_b.set_xlabel("Estimated Rate (%)", color='#94a3b8', fontsize=9)
        ax_b.set_ylabel("State", color='#94a3b8', fontsize=9)
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)
        ax_b.spines['bottom'].set_color('#334155')
        ax_b.spines['left'].set_color('#334155')
        
        st.pyplot(fig_bar)

with tab_covid:
    st.markdown('<div class="section-title">🦠 COVID-19 Lockdown Spike Investigation</div>', unsafe_allow_html=True)
    st.write("India implemented one of the strictest global lockdowns in late March 2020. This tab splits the data into three key economic segments to quantify this historical economic shock.")
    
    col_c1, col_c2 = st.columns([1, 1.2])
    
    # Pre-calculate periods
    df_state['Period'] = 'Normal'
    df_state.loc[(df_state['Date'] >= '2020-04-01') & (df_state['Date'] <= '2020-06-30'), 'Period'] = 'Lockdown Peak (Apr-Jun)'
    df_state.loc[(df_state['Date'] < '2020-04-01'), 'Period'] = 'Pre-Lockdown (Jan-Mar)'
    df_state.loc[(df_state['Date'] > '2020-06-30'), 'Period'] = 'Post-Lockdown (Jul-Nov)'
    
    period_avg = df_state.groupby('Period')[['Unemployment_Rate', 'Labour_Participation_Rate']].mean().reset_index()
    period_avg['sort_idx'] = [1, 0, 2] # Order logically
    period_sorted = period_avg.sort_values('sort_idx').drop(columns=['sort_idx'])
    
    with col_c1:
        st.write(f"**Timeline Stage Metrics Table ({selected_state})**")
        st.dataframe(period_sorted.rename(columns={
            'Period': 'Lockdown Phase',
            'Unemployment_Rate': 'Avg Unemployment (%)',
            'Labour_Participation_Rate': 'Avg Participation (%)'
        }), hide_index=True, use_container_width=True)
        
        # Display custom policy advice cards
        st.markdown('<div style="font-size: 1.15rem; font-weight:700; margin-top: 15px; margin-bottom:10px;">💡 Policy Recommendations</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="policy-card">
                <strong>1. Support Urban Vulnerable Informal Workforce</strong><br/>
                Lockdowns severely hit urban centers because retail, services, and construction completely froze. Emergency direct basic income transfers and cash benefits are vital during such freezes.
            </div>
            <div class="policy-card" style="border-left-color: #34d399;">
                <strong>2. Expand MGNREGA Job Guarantees</strong><br/>
                Rural employment guarantees (like MGNREGA) acted as a massive shock-absorber for returning urban migrant workers. Expanding similar policies to a permanent urban counterpart would build resilience.
            </div>
        """, unsafe_allow_html=True)
        
    with col_c2:
        st.write(f"**Unemployment Shock Bar Comparison ({selected_state})**")
        fig_cov, ax_c = plt.subplots(figsize=(6.5, 4.3), facecolor='none')
        ax_c.set_facecolor('rgba(0, 0, 0, 0.05)')
        
        bars = ax_c.barplot(
            data=period_sorted,
            x='Period',
            y='Unemployment_Rate',
            hue='Period',
            palette="coolwarm",
            legend=False
        )
        ax_c.set_title("Average Unemployment Rate by Lockdown Phase", color='#e2e8f0', fontsize=11, fontweight='bold')
        ax_c.tick_params(colors='#94a3b8', labelsize=8)
        ax_c.set_xlabel("Phase", color='#94a3b8', fontsize=9)
        ax_c.set_ylabel("Unemployment Rate (%)", color='#94a3b8', fontsize=9)
        
        # Add labels
        for idx, row in enumerate(period_sorted.itertuples()):
            ax_c.text(idx, row.Unemployment_Rate + 0.5, f"{row.Unemployment_Rate:.2f}%", 
                      ha='center', va='bottom', color='#e2e8f0', fontweight='bold', fontsize=9)
                      
        ax_c.spines['top'].set_visible(False)
        ax_c.spines['right'].set_visible(False)
        ax_c.spines['bottom'].set_color('#334155')
        ax_c.spines['left'].set_color('#334155')
        
        st.pyplot(fig_cov)

with tab_regional:
    st.markdown('<div class="section-title">🏘️ Rural vs. Urban Job Market Divide</div>', unsafe_allow_html=True)
    st.write("Using our first historical dataset, we can analyze the structural difference between job markets in rural areas (largely agricultural economies) and urban areas (service and industrial centers).")
    
    col_r1, col_r2 = st.columns([1, 1.2])
    
    with col_r1:
        st.write("**Key Observations:**")
        st.write("""
        * **Urban Instability:** Urban areas generally record higher baseline average unemployment rates than rural areas, reflecting the volatility of industrial job sectors.
        * **Rural Shock Absorbing:** Rural areas show a slightly wider spread but lower overall averages. This is because agricultural labor acts as a buffer—underemployed workers are still counted as occupied in family farming.
        """)
        
        # Calculate rural vs urban means
        st.write("**Average Historical Rates (Rural vs. Urban)**")
        rural_urban_mean = df_india.groupby('Area')['Unemployment_Rate'].mean().reset_index()
        st.dataframe(rural_urban_mean.rename(columns={
            'Area': 'Region Area Type',
            'Unemployment_Rate': 'Average Unemployment (%)'
        }), hide_index=True, use_container_width=True)
        
    with col_r2:
        st.write("**Rural vs. Urban Distribution Analysis**")
        fig_box, ax_box = plt.subplots(figsize=(6.5, 4.3), facecolor='none')
        ax_box.set_facecolor('rgba(0, 0, 0, 0.05)')
        
        sns.boxplot(
            ax=ax_box,
            data=df_india,
            x='Area',
            y='Unemployment_Rate',
            hue='Area',
            palette="Set2",
            legend=False
        )
        
        ax_box.set_title("Unemployment Rate Distribution: Rural vs. Urban Areas", color='#e2e8f0', fontsize=11, fontweight='bold')
        ax_box.tick_params(colors='#94a3b8', labelsize=8)
        ax_box.set_xlabel("Area Classification", color='#94a3b8', fontsize=9)
        ax_box.set_ylabel("Unemployment Rate (%)", color='#94a3b8', fontsize=9)
        ax_box.spines['top'].set_visible(False)
        ax_box.spines['right'].set_visible(False)
        ax_box.spines['bottom'].set_color('#334155')
        ax_box.spines['left'].set_color('#334155')
        
        st.pyplot(fig_box)
