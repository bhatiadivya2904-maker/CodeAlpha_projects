# 🎓 CodeAlpha Data Science Internship Projects

This repository contains the projects and tasks completed during the **Data Science Internship** at **CodeAlpha** (June 2026).

Intern: **Divya Bhatia**  
Student ID: **CA/DF1/89051**  

---

## 📁 Repository Structure & Projects

This repository is organized into distinct directories for each internship task:

### 🌸 [Task 1: Iris Flower Classification](./Task1_Iris_Classification)
* **Goal:** Train a Machine Learning model to classify Iris flowers (Setosa, Versicolor, Virginica) based on sepal and petal measurements.
* **Accuracy Achieved:** **96.67%** (using a Support Vector Classifier).
* **Deliverables:**
  - Robust training script (`src/train.py`) and command-line predictor (`src/predict.py`).
  - Exploratory Data Analysis (EDA) Jupyter Notebook (`notebooks/iris_classification_eda.ipynb`).
  - Sleek, dark-mode, interactive **Streamlit Web Dashboard** (`app.py`).
  - Detailed task documentation.

### 📈 [Task 2: Unemployment Analysis with Python](./Task2_Unemployment_Analysis)
* **Goal:** Analyze and visualize state-wise unemployment rates, employment numbers, and labor participation in India across 2020.
* **Key Finding:** Quantified the massive economic shock of the COVID-19 lockdowns, where national unemployment skyrocketed from **10.32%** to a peak average of **23.76%** (April-June 2020).
* **Deliverables:**
  - Automated downloader script (`src/download_data.py`) and pipeline (`src/analyze.py`).
  - Multi-tab **Streamlit Interactive Dashboard** (`app.py`) for state-level filtering, timeline segment analysis, and rural vs. urban job market divide box plots.
  - Detailed Exploratory Data Analysis Jupyter Notebook (`notebooks/unemployment_analysis.ipynb`).
  - Policy insights and recommendations.

### 🚗 [Task 3: Car Price Prediction with Machine Learning](./Task3_Car_Price_Prediction)
* **Goal:** Develop a Machine Learning regression pipeline to predict used car resale values based on various physical and commercial characteristics.
* **Accuracy Achieved:** **96.85% $R^2$ Score** (using a Gradient Boosting Regressor).
* **Deliverables:**
  - Automated used car dataset downloader (`src/download_data.py`) and training script (`src/train.py`).
  - Command-line predictor (`src/predict.py`) for instant resale value predictions.
  - Detailed Exploratory Data Analysis Jupyter Notebook (`notebooks/car_price_prediction_eda.ipynb`).
  - Elegant, dark-mode **Streamlit Resale Value Calculator** (`app.py`) complete with live scatter price-mapping, uncertainty ranges, and horizontal driver importance plots.

### 📈 [Task 4: Sales Prediction using Python](./Task4_Sales_Prediction)
* **Goal:** Predict future product sales based on advertising spends across TV, Radio, and Newspaper channels, and deliver actionable marketing budget optimization.
* **Accuracy Achieved:** **98.31% $R^2$ Score** (using a Gradient Boosting Regressor).
* **Deliverables:**
  - Robust training script (`src/train.py`), downloader (`src/download_data.py`), and CLI predictor (`src/predict.py`).
  - Sleek, dark-mode **AI Sales Forecaster & Spend Optimizer Dashboard** (`app.py`) featuring real-time budget allocation recommendations and comparative performance plotting.
  - Detailed Exploratory Data Analysis Jupyter Notebook (`notebooks/sales_prediction_eda.ipynb`).

---

## 🛠️ Getting Started & Local Setup

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. You can check using:
```bash
python --version
```

### 2. Clone the Repository
```bash
git clone https://github.com/bhatiadivya2904-maker/CodeAlpha_projects.git
cd CodeAlpha_projects
```

### 3. Running the Dashboards
To run any tasks, navigate to its folder, install dependencies and launch Streamlit:
```bash
# Example for Task 3
cd Task3_Car_Price_Prediction
pip install -r requirements.txt
streamlit run app.py
```
For detailed execution steps, CLI commands, and instructions, see the task-specific README files.
