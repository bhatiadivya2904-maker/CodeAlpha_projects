import os
import sys
import pandas as pd
import joblib

def load_inference_pipeline():
    model_path = os.path.join("models", "car_price_model.pkl")

    if not os.path.exists(model_path):
        print("Error: Serialized model file not found in 'models/' directory.")
        print("Please run 'python src/train.py' first to train and save the model.")
        sys.exit(1)

    model_payload = joblib.load(model_path)
    return model_payload

def predict_resale_value(present_price, kms_driven, fuel_type, seller_type, transmission, owner, car_age):
    # Load pipeline payload
    payload = load_inference_pipeline()
    pipeline = payload["pipeline"]
    
    # Input dataframe (features must match model training exactly)
    input_df = pd.DataFrame([{
        'Present_Price': float(present_price),
        'Kms_Driven': int(kms_driven),
        'Fuel_Type': str(fuel_type).strip(),
        'Seller_Type': str(seller_type).strip(),
        'Transmission': str(transmission).strip(),
        'Owner': int(owner),
        'Car_Age': int(car_age)
    }])
    
    # Predict
    predicted_val = pipeline.predict(input_df)[0]
    
    # Ensure predicted price doesn't go below zero
    predicted_val = max(0.0, predicted_val)
    
    return predicted_val, payload["metrics"]

def main():
    print("=" * 60)
    print("         USED CAR RESALE VALUE: PREDICTION TOOL")
    print("=" * 60)

    # Check CLI arguments (Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner, Car_Age)
    if len(sys.argv) == 8:
        try:
            present_price = float(sys.argv[1])
            kms_driven = int(sys.argv[2])
            fuel_type = sys.argv[3]
            seller_type = sys.argv[4]
            transmission = sys.argv[5]
            owner = int(sys.argv[6])
            car_age = int(sys.argv[7])
        except ValueError:
            print("Error: Numeric parameters (Present Price, Kms Driven, Owner, Age) must be valid numbers.")
            sys.exit(1)
    else:
        print("No command-line arguments detected. Entering interactive mode.\n")
        try:
            present_price = float(input("Enter current ex-showroom price (Lakhs) (e.g. 5.59): "))
            kms_driven = int(input("Enter kilometers driven (e.g. 27000): "))
            fuel_type = input("Enter fuel type (Petrol/Diesel/CNG): ").capitalize().strip()
            if fuel_type not in ["Petrol", "Diesel", "CNG"]:
                print("Warning: Standard fuel types are Petrol, Diesel, CNG.")
            seller_type = input("Enter seller type (Dealer/Individual): ").capitalize().strip()
            transmission = input("Enter transmission type (Manual/Automatic): ").capitalize().strip()
            owner = int(input("Enter number of previous owners (0/1/3): "))
            car_age = int(input("Enter age of the car in years (e.g. 5): "))
        except ValueError:
            print("Error: Input must be numeric or valid string.")
            sys.exit(1)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting prediction tool.")
            sys.exit(0)

    print("\nProcessing parameters and running regression...")
    pred_price, metrics = predict_resale_value(
        present_price, kms_driven, fuel_type, seller_type, transmission, owner, car_age
    )

    print("\n" + "=" * 45)
    print(f"  ESTIMATED RESALE VALUE: {pred_price:.2f} Lakhs")
    print("=" * 45)
    print("Model Evaluation Performance:")
    print(f" -> Algorithm        : Gradient Boosting Regressor")
    print(f" -> R2 Score         : {metrics['r2']:.2%}")
    print(f" -> Avg Error (MAE)  : {metrics['mae']:.2f} Lakhs")
    print("=" * 45)

if __name__ == "__main__":
    main()
