import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

def main():
    print("=" * 60)
    print("             SALES PREDICTION: MODEL TRAINING")
    print("=" * 60)

    # 1. Create directories
    os.makedirs("models", exist_ok=True)

    # 2. Load dataset
    print("[1/6] Loading advertising dataset...")
    csv_path = os.path.join("data", "advertising.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}. Please run download_data.py first.")
        return
        
    df = pd.read_csv(csv_path)
    print(f" -> Dataset loaded successfully! Shape: {df.shape}")
    print(f" -> Columns: {df.columns.tolist()}")

    # 3. Data Preprocessing & Cleaning
    print("\n[2/6] Performing Data Preprocessing...")
    # Drop index column 'Unnamed: 0' if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        print(" -> Dropped unnecessary 'Unnamed: 0' index column.")

    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    print(f" -> Features selected: {X.columns.tolist()}")
    print(" -> Target variable: 'Sales' (thousand units)")

    # 4. Split data (80% train, 20% test)
    print("\n[3/6] Splitting dataset into training and testing partitions...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f" -> Training set size: {X_train.shape[0]} samples")
    print(f" -> Testing set size: {X_test.shape[0]} samples")

    # 5. Define candidate models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # 6. Train and compare models
    print("\n[4/6] Training and comparing candidate regressors...")
    results = []
    best_r2 = -float('inf')
    best_model_name = ""
    best_pipeline = None

    # Preprocessor
    scaler = StandardScaler()

    # Train a baseline Linear Regression explicitly to extract exact coefficients
    lr_model = LinearRegression()
    lr_pipeline = Pipeline([
        ('scaler', scaler),
        ('regressor', lr_model)
    ])
    lr_pipeline.fit(X_train, y_train)
    
    # Store standard coefficients for spend optimization (rescaled back to original units)
    # y = intercept + w1*x1 + w2*x2 + w3*x3
    # If scaled: x_scaled = (x - mean)/std
    # y = intercept_scaled + sum(w_scaled * (x - mean)/std)
    #   = (intercept_scaled - sum(w_scaled * mean / std)) + sum((w_scaled / std) * x)
    mean_vals = lr_pipeline.named_steps['scaler'].mean_
    scale_vals = lr_pipeline.named_steps['scaler'].scale_
    coef_scaled = lr_pipeline.named_steps['regressor'].coef_
    intercept_scaled = lr_pipeline.named_steps['regressor'].intercept_
    
    # Rescaled coefficients (representing dollar-for-unit-sales elasticity)
    original_coefs = coef_scaled / scale_vals
    original_intercept = intercept_scaled - np.sum((coef_scaled * mean_vals) / scale_vals)
    
    linear_elasticities = {
        "TV": original_coefs[0],
        "Radio": original_coefs[1],
        "Newspaper": original_coefs[2],
        "Intercept": original_intercept
    }

    for name, model in models.items():
        # Create pipeline containing scaler and model
        pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        results.append({
            "Model": name,
            "R2_Score": r2,
            "MAE": mae,
            "RMSE": rmse
        })
        
        print(f" -> {name:<20} | R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    # Print comparison table
    print("\n" + "-" * 65)
    print(f"{'Regressor Model':<22} | {'R2 Score':<10} | {'MAE':<12} | {'RMSE':<12}")
    print("-" * 65)
    for res in results:
        print(f"{res['Model']:<22} | {res['R2_Score']:<10.2%} | {res['MAE']:<12.4f} | {res['RMSE']:<12.4f}")
    print("-" * 65)

    # 7. Detailed Evaluation on Best Model
    print(f"\n[5/6] Selected Best Model: {best_model_name}")
    print(f" -> Best R2 Score: {best_r2:.2%}")
    
    # Calculate feature importances or coefficients
    importance_dict = {}
    regressor = best_pipeline.named_steps['regressor']
    
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        importance_dict = dict(zip(X.columns, importances))
        print("\nFeature Importances:")
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_importance:
            print(f" -> {feat:<15}: {imp:.2%}")
    elif hasattr(regressor, 'coef_'):
        coefs = regressor.coef_
        importance_dict = dict(zip(X.columns, coefs))
        print("\nModel Coefficients:")
        for feat, coef in importance_dict.items():
            print(f" -> {feat:<15}: {coef:.4f}")

    # 8. Serialize the Best Model Payload
    print("\n[6/6] Serializing and saving best model pipeline...")
    model_path = os.path.join("models", "sales_prediction_model.pkl")
    
    model_payload = {
        "pipeline": best_pipeline,
        "model_name": best_model_name,
        "feature_names": X.columns.tolist(),
        "feature_importances": importance_dict,
        "linear_elasticities": linear_elasticities,
        "dataset_stats": {
            "TV": {"min": float(X['TV'].min()), "max": float(X['TV'].max()), "mean": float(X['TV'].mean())},
            "Radio": {"min": float(X['Radio'].min()), "max": float(X['Radio'].max()), "mean": float(X['Radio'].mean())},
            "Newspaper": {"min": float(X['Newspaper'].min()), "max": float(X['Newspaper'].max()), "mean": float(X['Newspaper'].mean())},
            "Sales": {"min": float(y.min()), "max": float(y.max()), "mean": float(y.mean())}
        },
        "metrics": {
            "r2": best_r2,
            "mae": mean_absolute_error(y_test, best_pipeline.predict(X_test)),
            "rmse": np.sqrt(mean_squared_error(y_test, best_pipeline.predict(X_test)))
        }
    }
    
    joblib.dump(model_payload, model_path)
    print(f" -> Model pipeline saved successfully to: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
