import os
import sys
import numpy as np
import joblib

def load_inference_pipeline():
    model_path = os.path.join("models", "iris_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: Serialized model or scaler files not found in 'models/' directory.")
        print("Please run 'python src/train.py' first to train and save the model.")
        sys.exit(1)

    model_payload = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model_payload, scaler

def predict_species(sepal_len, sepal_wid, petal_len, petal_wid):
    # Load model and scaler
    model_payload, scaler = load_inference_pipeline()
    
    model = model_payload["model"]
    target_names = model_payload["target_names"]
    
    # Input vector
    input_data = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    pred_class_idx = model.predict(input_scaled)[0]
    pred_species = target_names[pred_class_idx]
    
    # Probabilities
    probabilities = model.predict_proba(input_scaled)[0]
    
    return pred_species, dict(zip(target_names, probabilities))

def main():
    print("=" * 60)
    print("          IRIS FLOWER CLASSIFICATION: PREDICTION TOOL")
    print("=" * 60)

    # Check if arguments are provided via CLI
    if len(sys.argv) == 5:
        try:
            sepal_len = float(sys.argv[1])
            sepal_wid = float(sys.argv[2])
            petal_len = float(sys.argv[3])
            petal_wid = float(sys.argv[4])
        except ValueError:
            print("Error: All 4 measurements must be numeric values.")
            sys.exit(1)
    else:
        print("No command-line arguments detected. Entering interactive mode.\n")
        try:
            sepal_len = float(input("Enter Sepal Length (cm) (e.g. 5.1): "))
            sepal_wid = float(input("Enter Sepal Width (cm)  (e.g. 3.5): "))
            petal_len = float(input("Enter Petal Length (cm) (e.g. 1.4): "))
            petal_wid = float(input("Enter Petal Width (cm)  (e.g. 0.2): "))
        except ValueError:
            print("Error: Input must be a valid float value.")
            sys.exit(1)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting prediction tool.")
            sys.exit(0)

    print("\nProcessing measurements...")
    species, probs = predict_species(sepal_len, sepal_wid, petal_len, petal_wid)

    print("\n" + "=" * 40)
    print(f"  PREDICTED SPECIES: {species.upper()}")
    print("=" * 40)
    print("Confidence breakdown:")
    for spec, prob in probs.items():
        print(f" -> {spec.capitalize():<12}: {prob:.2%}")
    print("=" * 40)

if __name__ == "__main__":
    main()
