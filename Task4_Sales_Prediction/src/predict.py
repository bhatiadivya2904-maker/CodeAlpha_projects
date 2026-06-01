import os
import argparse
import joblib
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Predict Sales based on Advertising Budgets.")
    parser.add_argument("--tv", type=float, required=True, help="Advertising budget spent on TV (in thousands of dollars).")
    parser.add_argument("--radio", type=float, required=True, help="Advertising budget spent on Radio (in thousands of dollars).")
    parser.add_argument("--newspaper", type=float, required=True, help="Advertising budget spent on Newspaper (in thousands of dollars).")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("             SALES PREDICTION: INFERENCE ENGINE")
    print("=" * 60)
    
    # Load model
    model_path = os.path.join("models", "sales_prediction_model.pkl")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please run 'python src/train.py' first to train and serialize the model.")
        return
        
    try:
        model_payload = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    pipeline = model_payload["pipeline"]
    model_name = model_payload["model_name"]
    metrics = model_payload["metrics"]
    
    print(f" -> Loaded Model: {model_name} (Test R2: {metrics['r2']:.2%})")
    
    # Build input dataframe
    input_data = pd.DataFrame([{
        "TV": args.tv,
        "Radio": args.radio,
        "Newspaper": args.newspaper
    }])
    
    # Predict
    try:
        prediction = pipeline.predict(input_data)[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
        
    print("\n" + "-" * 45)
    print(f"{'Platform':<15} | {'Budget ($ thousands)':<25}")
    print("-" * 45)
    print(f"{'TV':<15} | ${args.tv:<24.2f}")
    print(f"{'Radio':<15} | ${args.radio:<24.2f}")
    print(f"{'Newspaper':<15} | ${args.newspaper:<24.2f}")
    print("-" * 45)
    print(f"{'PREDICTED SALES':<15} | {prediction:<24.4f} thousand units")
    print(f"{'Estimated Revenue':<15} | ${prediction * 1000:<24.2f} (assuming $1 per unit)")
    print("-" * 45)
    print("=" * 60)

if __name__ == "__main__":
    main()
