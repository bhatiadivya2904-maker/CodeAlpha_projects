# 🌸 Iris Flower Classification

Welcome to the **Iris Flower Classification** project, developed as part of **Task 1** of the **CodeAlpha Data Science Internship**. 

This repository implements a production-grade, end-to-end Machine Learning pipeline to classify Iris flowers into their respective species: **Setosa**, **Versicolor**, and **Virginica**, based on four physical measurements (Sepal Length, Sepal Width, Petal Length, and Petal Width).

---

## 🌟 Key Features

1. **Exploratory Data Analysis (EDA):** A detailed Jupyter Notebook featuring beautiful, modern, publication-ready statistical visualizations (correlation heatmaps, pairplots, and distribution density violin plots).
2. **Multi-Model Pipeline:** Automated model training, hyperparameter configuration, and comparative analysis of four machine learning classifiers (Logistic Regression, Support Vector Classifier, Random Forest, and k-Nearest Neighbors).
3. **Robust Serialization:** Clean serialization of both training models and dataset feature scaling parameters (`StandardScaler`) to ensure consistency during production inference.
4. **Interactive Dashboard:** A premium, dark-mode, glassmorphic **Streamlit Web Application** showcasing:
   - **Real-Time Classification:** Adjust sliders to predict species with instant probability breakdown bar charts.
   - **Dynamic Feature Mapping:** The application dynamically plots the training distribution and highlights the user's custom inputs as a **large golden star (⭐)** to show exactly where the sample lies in the feature space.
   - **Interactive EDA Browser:** Explore violin distribution plots and descriptive stats directly from the web interface.
5. **Command-Line Interface (CLI):** A handy, robust terminal command-line tool supporting direct or interactive input options.

---

## 📁 Repository Structure

```text
Task1_Iris_Classification/
├── requirements.txt            # System dependencies
├── app.py                      # Premium Streamlit web application
├── README.md                   # Project documentation
├── data/
│   └── iris_dataset.csv        # Local CSV copy of Iris dataset
├── models/
│   ├── iris_model.pkl          # Serialized production machine learning model (SVC)
│   └── scaler.pkl              # Serialized feature scaling weights
├── notebooks/
│   └── iris_classification_eda.ipynb  # Interactive Jupyter Notebook for EDA & Training
└── src/
    ├── train.py                # Model training and comparative evaluation script
    └── predict.py              # CLI/interactive prediction inference script
```

---

## ⚡ Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. You can check your version using:
```bash
python --version
```

### 2. Install Dependencies
Clone or copy this folder to your workspace, navigate to the folder, and run:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 1. Exploratory Data Analysis & Jupyter Notebook
To run the notebook and explore the visual data analysis:
```bash
jupyter notebook notebooks/iris_classification_eda.ipynb
```

### 2. Retraining the Machine Learning Model
You can easily re-run the comparative training script. It loads the dataset, trains candidate classifiers, prints performance metrics, and saves the best model:
```bash
python src/train.py
```

### 3. Running CLI Inference Predictions
You can run the prediction script in **interactive mode**:
```bash
python src/predict.py
```
Or pass measurements directly as **command-line arguments** (SepalLength SepalWidth PetalLength PetalWidth):
```bash
python src/predict.py 5.1 3.5 1.4 0.2
```

### 4. Running the Streamlit Web Application
To spin up the beautiful, interactive dashboard in your browser:
```bash
streamlit run app.py
```

---

## 📈 Model Performance & Comparative Results

During evaluation, four algorithms were benchmarked on the stratified test split (20% of the dataset):

| Classifier | Accuracy |
| :--- | :--- |
| **Support Vector Classifier (SVC)** | **96.67%** 🌟 *(Selected Production Model)* |
| **k-Nearest Neighbors (k-NN)** | **93.33%** |
| **Logistic Regression** | **93.33%** |
| **Random Forest Classifier** | **90.00%** |

### Best Model Performance (SVC)
- **Overall Accuracy:** `96.67%`
- **Class-wise Precision & Recall:**
  - `Iris Setosa`: Precision `1.00`, Recall `1.00`
  - `Iris Versicolor`: Precision `1.00`, Recall `0.90`
  - `Iris Virginica`: Precision `0.91`, Recall `1.00`
