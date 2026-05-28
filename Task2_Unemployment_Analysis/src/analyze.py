import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clean_and_prepare(df):
    # 1. Strip whitespaces from column names
    df.columns = df.columns.str.strip()
    
    # 2. Rename columns using a robust substring mapping
    rename_dict = {}
    for col in df.columns:
        if 'unemployment' in col.lower():
            rename_dict[col] = 'Unemployment_Rate'
        elif 'employed' in col.lower():
            rename_dict[col] = 'Employed'
        elif 'participation' in col.lower():
            rename_dict[col] = 'Labour_Participation_Rate'
            
    df = df.rename(columns=rename_dict)
    
    # 3. Drop rows where crucial columns are missing
    df = df.dropna(subset=['Region', 'Date'])
    
    # 4. Clean string columns
    df['Region'] = df['Region'].astype(str).str.strip()
    df['Date'] = df['Date'].astype(str).str.strip()
    if 'Frequency' in df.columns:
        df['Frequency'] = df['Frequency'].astype(str).str.strip()
    if 'Area' in df.columns:
        df['Area'] = df['Area'].astype(str).str.strip()
        
    # 5. Standardize date formats (Kaggle formats vary)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # 6. Extract month, year, and Month-Name
    df['Year'] = df['Date'].dt.year
    df['Month_Num'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%b')
    df['Year_Month'] = df['Date'].dt.to_period('M')
    
    return df

def main():
    print("=" * 60)
    print("        UNEMPLOYMENT RATE ANALYSIS: DATA PIPELINE")
    print("=" * 60)
    
    # Create output directory for plots
    os.makedirs("visualizations", exist_ok=True)
    
    # 1. Load datasets
    print("[1/5] Loading datasets...")
    path1 = os.path.join("data", "Unemployment in India.csv")
    path2 = os.path.join("data", "Unemployment_Rate_upto_11_2020.csv")
    
    if not os.path.exists(path1) or not os.path.exists(path2):
        print("Error: Datasets are missing. Please run 'python src/download_data.py' first.")
        return
        
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    
    # 2. Clean datasets
    print("[2/5] Cleaning and standardizing datasets...")
    df1_clean = clean_and_prepare(df1)
    df2_clean = clean_and_prepare(df2)
    
    print(f" -> df1 (Unemployment in India) shape after cleaning: {df1_clean.shape}")
    print(f" -> df2 (Unemployment Rate Upto Nov 2020) shape after cleaning: {df2_clean.shape}")
    
    # Set styling for plots
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.size"] = 11
    
    # ------------------ PLOT 1: State-wise Average Unemployment Rate (from df2) ------------------
    print("\n[3/5] Generating Plot 1: State-wise Average Unemployment Rate...")
    plt.figure(figsize=(12, 7))
    state_avg = df2_clean.groupby("Region")["Unemployment_Rate"].mean().sort_values(ascending=False)
    
    sns.barplot(x=state_avg.values, y=state_avg.index, hue=state_avg.index, palette="viridis", legend=False)
    plt.title("Average Unemployment Rate by Indian State (Jan - Nov 2020)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Estimated Unemployment Rate (%)", fontsize=11, labelpad=10)
    plt.ylabel("State", fontsize=11, labelpad=10)
    plt.tight_layout()
    plot1_path = os.path.join("visualizations", "statewise_avg_unemployment.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f" -> Saved: {plot1_path}")
    
    # ------------------ PLOT 2: Time-series Trend (Covid Spike) ------------------
    print("[4/5] Generating Plot 2: Historical Unemployment Trend (Spike Analysis)...")
    plt.figure(figsize=(11, 5.5))
    monthly_trend = df2_clean.groupby("Date")["Unemployment_Rate"].mean().reset_index()
    monthly_trend = monthly_trend.sort_values("Date")
    
    plt.plot(monthly_trend["Date"], monthly_trend["Unemployment_Rate"], 
             marker='o', color='#e15759', linewidth=2.5, markersize=7, label="National Avg")
             
    # Annotate Lockdown Spike
    lockdown_date = pd.to_datetime('2020-04-30')
    lockdown_rate = monthly_trend.loc[monthly_trend['Date'] == lockdown_date, 'Unemployment_Rate']
    if not lockdown_rate.empty:
        rate_val = lockdown_rate.values[0]
        plt.annotate(f"Lockdown Spike: {rate_val:.1f}%", 
                     xy=(lockdown_date, rate_val),
                     xytext=(pd.to_datetime('2020-06-15'), rate_val + 3),
                     arrowprops=dict(facecolor='#444', shrink=0.08, width=1, headwidth=6),
                     fontweight='bold', color='#c0392b', fontsize=10)
                     
    plt.title("Average Monthly Unemployment Rate Trend in India (2020)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=11, labelpad=10)
    plt.ylabel("Estimated Unemployment Rate (%)", fontsize=11, labelpad=10)
    plt.ylim(0, max(monthly_trend["Unemployment_Rate"]) + 5)
    plt.tight_layout()
    plot2_path = os.path.join("visualizations", "unemployment_trend_2020.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f" -> Saved: {plot2_path}")
    
    # ------------------ PLOT 3: Covid Impact Comparison (Pre- vs. During-Lockdown) ------------------
    print("[5/5] Generating Plot 3: COVID-19 Lockdown Impact Comparison...")
    # Lockdown in India was announced in late March 2020. Let's compare:
    # Pre-Lockdown: Jan-March 2020
    # Lockdown: April-June 2020
    
    df2_clean['Period'] = 'Normal'
    df2_clean.loc[(df2_clean['Date'] >= '2020-04-01') & (df2_clean['Date'] <= '2020-06-30'), 'Period'] = 'Lockdown Peak (Apr-Jun)'
    df2_clean.loc[(df2_clean['Date'] < '2020-04-01'), 'Period'] = 'Pre-Lockdown (Jan-Mar)'
    df2_clean.loc[(df2_clean['Date'] > '2020-06-30'), 'Period'] = 'Post-Lockdown (Jul-Nov)'
    
    plt.figure(figsize=(10, 6))
    period_avg = df2_clean.groupby('Period')['Unemployment_Rate'].mean().reset_index()
    # Sort order
    period_avg['sort_idx'] = [1, 0, 2] # Lockdown, Pre-lockdown, Post-lockdown ordering
    period_sorted = period_avg.sort_values('sort_idx')
    
    sns.barplot(data=period_sorted, x='Period', y='Unemployment_Rate', hue='Period', palette="coolwarm", legend=False)
    plt.title("Unemployment Rate Comparison: Pre vs. Peak vs. Post COVID-19 Lockdown", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Timeline Period (2020)", fontsize=11, labelpad=10)
    plt.ylabel("Average Unemployment Rate (%)", fontsize=11, labelpad=10)
    
    # Draw value labels on top of bars
    for i, row in enumerate(period_sorted.itertuples()):
        plt.text(i, row.Unemployment_Rate + 0.5, f"{row.Unemployment_Rate:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)
        
    plt.tight_layout()
    plot3_path = os.path.join("visualizations", "covid_lockdown_impact.png")
    plt.savefig(plot3_path, dpi=150)
    plt.close()
    print(f" -> Saved: {plot3_path}")
    
    # Save a merged clean CSV copy of df2 for easier Streamlit and Notebook use
    merged_path = os.path.join("data", "unemployment_cleaned.csv")
    df2_clean.to_csv(merged_path, index=False)
    print(f"\n -> Cleaned unemployment data saved to: {merged_path}")
    print("=" * 60)
    print(" Data cleaning, modeling aggregations, and plots completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
