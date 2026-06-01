# 📈 Sales Prediction using Python

Welcome to the **Sales Prediction** project, developed as part of **Task 4** of the **CodeAlpha Data Science Internship**. 

This repository implements a highly sophisticated and visually stunning end-to-end Machine Learning regression and optimization pipeline to forecast sales volume based on marketing campaign investments across three core mediums: **TV**, **Radio**, and **Newspaper**.

---

## 🌟 Key Features

1. **Jupyter Notebook (`sales_prediction_eda.ipynb`):** A complete data science workflow from scratch featuring descriptive statistics, missing value assertions, multi-variable correlation checks, trend curves, and detailed model evaluation.
2. **Robust Preprocessing & Feature Pipeline:**
   - **Clean Processing:** Automatic removal of redundant index columns (`Unnamed: 0`).
   - **Scaling Pipeline:** Clean implementation of `StandardScaler` to uniform numeric ranges for proper gradient tracking and regularized estimation.
3. **Multi-Regressor Pipeline:** Automated training, validation, and comparative performance evaluation of five regression models:
   - **Gradient Boosting Regressor** *(Selected Production Model - R²: 98.31%)*
   - **Random Forest Regressor** *(R²: 98.13%)*
   - **Linear Regression** *(R²: 89.94% - for elasticity insights)*
   - **Ridge Regression** *(R²: 89.88%)*
   - **Lasso Regression** *(R²: 89.83%)*
4. **Interactive Dashboard:** A premium, dark-mode, glassmorphic **Streamlit Web Application** featuring:
   - **Live Sales Forecast Calculator:** Adjust TV, Radio, and Newspaper budgets using smooth sliders and view real-time sales predictions, confidence bounds, and revenue estimates.
   - **AI Spend Optimizer:** Input a total marketing budget and mathematically optimize the allocation across channels to maximize sales (uses a greedy Bounded Linear Programming strategy).
   - **Exploratory Analytics:** Interactive correlation heatmaps and regression trend curves.
   - **Model Benchmarking:** Interactive bar charts comparing R² and error statistics across all trained models.
5. **Command-Line Predictor (CLI):** A command-line script supporting direct parameter arguments.

---

## 📁 Repository Structure

```text
Task4_Sales_Prediction/
├── requirements.txt            # System dependencies
├── app.py                      # Premium Streamlit web application
├── README.md                   # Project documentation
├── data/
│   └── advertising.csv         # Local copy of the Advertising dataset
├── models/
│   └── sales_prediction_model.pkl # Serialized production model package and metadata
├── notebooks/
│   └── sales_prediction_eda.ipynb # Detailed Jupyter Notebook for EDA & training
└── src/
    ├── download_data.py        # Automated dataset downloader
    ├── train.py                # Pipeline for model training, metrics benchmark, and serialization
    └── predict.py              # CLI inference prediction tool
```

---

## ⚡ Setup & Installation

### 1. Install Dependencies
Clone the repository, navigate to the folder, and run:
```bash
pip install -r requirements.txt
```

### 2. Prepare Data & Train Model (Optional)
The datasets are already pre-downloaded and trained. However, to run the pipeline from scratch:
```bash
# Download raw Advertising CSV dataset
python src/download_data.py

# Run comparative training and serialize model package
python src/train.py
```

---

## 🚀 Running the Project

### 1. Launching the Jupyter Notebook
To run the detailed EDA notebook:
```bash
jupyter notebook notebooks/sales_prediction_eda.ipynb
```

### 2. Running CLI Predictions
You can run the prediction script from your command-line terminal:
```bash
# Arguments: --tv <budget> --radio <budget> --newspaper <budget>
python src/predict.py --tv 150 --radio 30 --newspaper 20
```

### 3. Running the Streamlit Web Application
To spin up the beautiful, interactive dashboard in your browser:
```bash
streamlit run app.py
```

---

## 📈 Regressor Model Performance Benchmarks

During evaluation, five regression models were benchmarked on the test split (20% of the dataset):

| Regressor Algorithm | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting Regressor** | **98.31%** 🌟 | **0.6181** *(thousand units)* | **0.7295** |
| **Random Forest Regressor** | **98.13%** | **0.6207** | **0.7688** |
| **Linear Regression** | **89.94%** | **1.4608** | **1.7816** |
| **Ridge Regression** | **89.88%** | **1.4643** | **1.7872** |
| **Lasso Regression** | **89.83%** | **1.4613** | **1.7913** |

### Best Model Feature Drivers
According to the Gradient Boosting importances, two primary channels account for **99.35%** of predictive sales variance:
1. **TV Budget (`TV`):** `61.26%` (Volume multiplier)
2. **Radio Budget (`Radio`):** `38.09%` (Velocity multiplier)
3. **Newspaper Budget (`Newspaper`):** `0.65%` (Negligible impact)

### Elasticity Multipliers (ROI Return per $1k Spend)
Derived from our rescaled baseline Linear Regression coefficients:
- **Radio ROI:** **+0.1888** sales units per $1k spend
- **TV ROI:** **+0.0457** sales units per $1k spend
- **Newspaper ROI:** **+0.0010** sales units per $1k spend
