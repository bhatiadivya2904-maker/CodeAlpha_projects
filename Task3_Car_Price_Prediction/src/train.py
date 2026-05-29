import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

def main():
    print("=" * 60)
    print("         CAR RESALE VALUE PREDICTION: MODEL TRAINING")
    print("=" * 60)

    # 1. Create directories
    os.makedirs("models", exist_ok=True)

    # 2. Load dataset
    print("[1/6] Loading used car dataset...")
    csv_path = os.path.join("data", "car_data.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}. Please run download_data.py first.")
        return
        
    df = pd.read_csv(csv_path)
    print(f" -> Dataset loaded successfully! Shape: {df.shape}")
    print(f" -> Columns: {df.columns.tolist()}")

    # 3. Feature Engineering
    print("\n[2/6] Performing Feature Engineering...")
    # Calculate age of the car (relative to 2026)
    current_year = 2026
    df['Car_Age'] = current_year - df['Year']
    print(f" -> Created 'Car_Age' feature (Current year: {current_year})")
    
    # Drop calendar Year and Car_Name (Car_Name is high-cardinality and leads to overfitting)
    X = df.drop(columns=['Selling_Price', 'Year', 'Car_Name'])
    y = df['Selling_Price']
    
    print(f" -> Features selected: {X.columns.tolist()}")
    print(" -> Target variable: 'Selling_Price'")

    # 4. Preprocessing Pipeline setup
    # Numeric features
    numeric_features = ['Present_Price', 'Kms_Driven', 'Car_Age', 'Owner']
    # Categorical features
    categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    # 5. Split data (80% train, 20% test)
    print("\n[3/6] Splitting dataset into training and testing partitions...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f" -> Training set size: {X_train.shape[0]} samples")
    print(f" -> Testing set size: {X_test.shape[0]} samples")

    # 6. Define candidate models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # 7. Train and compare models
    print("\n[4/6] Training and comparing candidate regressors...")
    results = []
    best_r2 = -float('inf')
    best_model_name = ""
    best_pipeline = None

    for name, model in models.items():
        # Create pipeline containing preprocessor and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
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
        
        print(f" -> {name:<20} | R2: {r2:.4f} | MAE: {mae:.4f} Lakhs | RMSE: {rmse:.4f} Lakhs")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    # Print comparison table
    print("\n" + "-" * 65)
    print(f"{'Regressor Model':<22} | {'R2 Score':<10} | {'MAE (Lakhs)':<12} | {'RMSE (Lakhs)':<12}")
    print("-" * 65)
    for res in results:
        print(f"{res['Model']:<22} | {res['R2_Score']:<10.2%} | {res['MAE']:<12.4f} | {res['RMSE']:<12.4f}")
    print("-" * 65)

    # 8. Detailed Evaluation on Best Model
    print(f"\n[5/6] Selected Best Model: {best_model_name}")
    print(f" -> Best R2 Score: {best_r2:.2%}")
    
    # Calculate feature importances if applicable
    importance_dict = {}
    regressor = best_pipeline.named_steps['regressor']
    
    # Get feature names after one-hot encoding
    cat_encoder = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names
    
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        importance_dict = dict(zip(all_feature_names, importances))
        print("\nFeature Importances:")
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_importance:
            print(f" -> {feat:<35}: {imp:.2%}")

    # 9. Serialize the Best Model Payload
    print("\n[6/6] Serializing and saving best model pipeline...")
    model_path = os.path.join("models", "car_price_model.pkl")
    
    model_payload = {
        "pipeline": best_pipeline,
        "model_name": best_model_name,
        "feature_names": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_importances": importance_dict,
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
