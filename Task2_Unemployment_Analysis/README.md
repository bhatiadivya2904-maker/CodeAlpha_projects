# 📈 Unemployment Analysis with Python

Welcome to the **Unemployment Analysis** project, developed as part of **Task 2** of the **CodeAlpha Data Science Internship**.

This project implements a highly structured and visually rich data analysis pipeline to inspect, clean, and visualize unemployment statistics across Indian states during the critical year of 2020. Specifically, it quantifies the massive, historical labor shock caused by the **COVID-19 pandemic** and lockdowns.

---

## 🌟 Key Insights & Features

1. **Quantified COVID-19 Lockdown Shock:** 
   Our analysis segments the year 2020 into three phases:
   * **Pre-Lockdown (Jan - Mar):** Baseline unemployment averaged **10.32%**.
   * **Lockdown Peak (Apr - Jun):** Unemployment skyrocketed to a staggering national average of **23.76%**.
   * **Post-Lockdown (Jul - Nov):** Phased business reopening saw a gradual recovery to **9.17%**.
2. **Exploratory Data Analysis (EDA):** A detailed Jupyter Notebook outlining data cleaning, monthly time-series spikes, and regional variations.
3. **Rural vs. Urban Divide:** Analysis reveals that urban areas experienced higher baseline and volatile unemployment averages than rural zones due to direct restrictions on services and retail.
4. **Interactive Dashboard:** A premium, dark-mode **Streamlit Web Application** featuring:
   * **Dynamic State Filtering:** Select any state to immediately focus the metrics.
   * **Covid-19 Timeline Segmenter:** Quantifies changes in labor participation.
   * **Custom Interactive Plots:** Highly customized line charts, box plots, and bar comparisons.
   * **Policy Insights Tab:** Features economic and social policy recommendations.

---

## 📁 Repository Structure

```text
Task2_Unemployment_Analysis/
├── requirements.txt            # System dependencies
├── app.py                      # Premium Streamlit web application
├── README.md                   # Project documentation
├── data/
│   ├── Unemployment in India.csv              # Raw rural/urban dataset
│   ├── Unemployment_Rate_upto_11_2020.csv     # Raw monthly state-level dataset
│   └── unemployment_cleaned.csv               # Standardized combined dataset
├── notebooks/
│   └── unemployment_analysis.ipynb            # Interactive Jupyter Notebook
├── src/
│   ├── download_data.py                       # Automated URL downloader
│   └── analyze.py                             # Clean, prepare, and plot pipeline
└── visualizations/                            # Exported high-fidelity charts
    ├── covid_lockdown_impact.png
    ├── statewise_avg_unemployment.png
    └── unemployment_trend_2020.png
```

---

## ⚡ Setup & Installation

### 1. Install Dependencies
Ensure you are in the task directory and run:
```bash
pip install -r requirements.txt
```

### 2. Prepare Data (Optional)
The datasets are already pre-downloaded and cleaned. However, to re-run the downloader and data preparation pipelines:
```bash
# Fetch raw CSVs from GitHub mirrors
python src/download_data.py

# Clean data, save merged copy, and export static charts
python src/analyze.py
```

---

## 🚀 Running the Project

### 1. Launching the Jupyter Notebook
To run the detailed EDA notebook:
```bash
jupyter notebook notebooks/unemployment_analysis.ipynb
```

### 2. Running the Streamlit Web Application
To launch the interactive dashboard in your browser:
```bash
streamlit run app.py
```

---

## 💡 Policy Recommendations from the Data
Based on the massive spikes in urban unemployment and the shock absorbing role of agricultural economies:
* **Urban Informal Support:** Immediate, targeted direct cash and food transfers are essential to sustain the informal workforce during systemic economic lockouts.
* **Urban Job Guarantees:** Implementing an urban counterpart to MGNREGA (Rural Employment Guarantee) is highly recommended to protect vulnerable industrial and retail workers in metropolitan areas.
