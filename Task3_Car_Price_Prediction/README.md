# 🚗 Car Price Prediction with Machine Learning

Welcome to the **Car Price Prediction** project, developed as part of **Task 3** of the **CodeAlpha Data Science Internship**. 

This repository implements a highly sophisticated and visually stunning end-to-end Machine Learning regression pipeline to estimate used car resale values based on various physical and commercial characteristics (such as showroom price, mileage, age, fuel type, transmission, and ownership count).

---

## 🌟 Key Features

1. **Jupyter Notebook (`car_price_prediction_eda.ipynb`):** A complete data science workflow from scratch featuring descriptive statistics, multi-variable correlation checks, time-series depreciation curves, feature transformations, and detailed model evaluation.
2. **Robust Preprocessing & Feature Engineering:**
   - **Depreciation calculation:** Transforms the calendar `Year` into `Car_Age` relative to 2026, which displays a much cleaner linear relationship with car resale depreciation.
   - **High-cardinality protection:** Drops unique `Car_Name` values to prevent model overfitting.
   - **Preprocessing Pipelines:** Clean implementation of `ColumnTransformer` applying standard scaling to numeric parameters and one-hot encoding (with `drop='first'`) to categorical features (`Fuel_Type`, `Seller_Type`, `Transmission`).
3. **Multi-Regressor Pipeline:** Automated training, validation, and comparative performance plotting of five regression models:
   - **Gradient Boosting Regressor** *(Selected Production Model)*
   - **Random Forest Regressor**
   - **Decision Tree Regressor**
   - **Ridge Regression**
   - **Linear Regression**
4. **Interactive Dashboard:** A premium, dark-mode, glassmorphic **Streamlit Web Application** featuring:
   - **Live Valuation Calculator:** Sliders and selection menus to compute estimated resale prices on-the-fly.
   - **Expected Pricing Margins:** Outlines range approximations based on actual model deviation statistics (Mean Absolute Error).
   - **Interactive Scatter Mapping:** Dynamically plots resale vs. showroom distributions and highlights the user's specific predicted vehicle configuration as a **large glowing gold star (⭐)**.
   - **Key Value Drivers:** Renders a horizontal bar chart displaying relative feature importances (e.g. current showroom price accounting for over **88%** of resale variance).
5. **Command-Line Predictor (CLI):** A command-line script supporting direct parameter arguments or interactive console prompts.

---

## 📁 Repository Structure

```text
Task3_Car_Price_Prediction/
├── requirements.txt            # System dependencies
├── app.py                      # Premium Streamlit web application
├── README.md                   # Project documentation
├── data/
│   └── car_data.csv            # Local copy of Used Car resale dataset
├── models/
│   └── car_price_model.pkl     # Serialized Gradient Boosting production model package
├── notebooks/
│   └── car_price_prediction_eda.ipynb # Detailed Jupyter Notebook for EDA & training
└── src/
    ├── download_data.py        # Automated used car dataset downloader
    ├── train.py                # Pipeline for model training, metrics benchmark, and serialization
    └── predict.py              # CLI/interactive inference prediction tool
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
# Download raw Used Car CSV dataset
python src/download_data.py

# Run comparative training and serialize model package
python src/train.py
```

---

## 🚀 Running the Project

### 1. Launching the Jupyter Notebook
To run the detailed EDA notebook:
```bash
jupyter notebook notebooks/car_price_prediction_eda.ipynb
```

### 2. Running CLI Resale Predictions
You can run the prediction script in **interactive mode**:
```bash
python src/predict.py
```
Or pass measurements directly as **command-line arguments** (Present_Price Kms_Driven Fuel_Type Seller_Type Transmission Owner Car_Age):
```bash
# Arguments: ShowroomPrice Kms Fuel Seller Transmission Owners Age
python src/predict.py 5.59 27000 Petrol Dealer Manual 0 5
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
| **Gradient Boosting Regressor** | **96.85%** 🌟 | **0.56 Lakhs** *(~56,000 INR)* | **0.85 Lakhs** |
| **Random Forest Regressor** | **96.25%** | **0.62 Lakhs** | **0.93 Lakhs** |
| **Decision Tree Regressor** | **91.07%** | **0.86 Lakhs** | **1.43 Lakhs** |
| **Ridge Regression** | **84.92%** | **1.21 Lakhs** | **1.86 Lakhs** |
| **Linear Regression** | **84.90%** | **1.22 Lakhs** | **1.87 Lakhs** |

### Best Model Feature Drivers
According to the Gradient Boosting importances, three primary factors account for **99.2%** of used car resale value:
1. **Showroom Price (`Present_Price`):** `88.17%` (Base value anchor)
2. **Car Age (`Car_Age`):** `8.43%` (Linear depreciation)
3. **Mileage (`Kms_Driven`):** `2.67%` (Wear and tear index)
