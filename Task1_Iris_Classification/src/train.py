import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def main():
    print("=" * 60)
    print("           IRIS FLOWER CLASSIFICATION: MODEL TRAINING")
    print("=" * 60)

    # 1. Create directory structure if not exists
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 2. Load the dataset
    print("[1/6] Loading Iris dataset from Scikit-Learn...")
    iris = load_iris()
    
    # Create a nice Pandas DataFrame for local inspection and CSV save
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[x] for x in iris.target]
    df['target'] = iris.target
    
    csv_path = os.path.join("data", "iris_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f" -> Dataset saved locally to: {csv_path}")
    print(f" -> Dataset shape: {df.shape}")
    print(f" -> Target classes: {list(iris.target_names)}")
    
    # 3. Preprocessing
    print("\n[2/6] Preprocessing and splitting data...")
    X = iris.data
    y = iris.target
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Feature Scaling (Crucial for SVM, k-NN, and Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference/prediction pipeline
    scaler_path = os.path.join("models", "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f" -> Feature scaler saved to: {scaler_path}")

    # 4. Define candidate models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "Support Vector Classifier": SVC(probability=True, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }

    # 5. Train and evaluate all candidate models
    print("\n[3/6] Training and comparing candidate models...")
    best_accuracy = 0.0
    best_model_name = ""
    best_model = None
    results = []

    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": acc})
        print(f" -> {name}: Accuracy = {acc:.4f}")
        
        # Check if it is the best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    # Print clean comparison table
    print("\n" + "-" * 40)
    print(f"{'Model Name':<30} | {'Accuracy':<8}")
    print("-" * 40)
    for res in results:
        print(f"{res['Model']:<30} | {res['Accuracy']:.4%}")
    print("-" * 40)

    # 6. Detailed Evaluation of the Best Model
    print(f"\n[4/6] Selected Best Model: {best_model_name}")
    print(f"[5/6] Performing detailed evaluation on best model...")
    
    y_pred_best = best_model.predict(X_test_scaled)
    acc_best = accuracy_score(y_test, y_pred_best)
    
    print(f"\nOverall Test Accuracy: {acc_best:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=iris.target_names))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_best))

    # 7. Serialize the Best Model
    print("\n[6/6] Serializing and saving best model...")
    model_path = os.path.join("models", "iris_model.pkl")
    # We save a dictionary containing the model itself, target names, and feature names for easy use
    model_payload = {
        "model": best_model,
        "model_name": best_model_name,
        "target_names": list(iris.target_names),
        "feature_names": list(iris.feature_names)
    }
    joblib.dump(model_payload, model_path)
    print(f" -> Best model ({best_model_name}) successfully saved to: {model_path}")
    print("\nTraining complete! Ready for CLI predictions and web dashboard integration.")
    print("=" * 60)

if __name__ == "__main__":
    main()
