import pickle
from faultdetect_train import predict_state


# ── Load Model & Scaler ───────────────────────────────────────────────────────
def load_artifacts(model_path="motor_fault_rf.pkl", scaler_path="scaler_rf.pkl"):
    model  = pickle.load(open(model_path,  "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    return model, scaler


# ── Single Prediction ─────────────────────────────────────────────────────────
def predict_single(current, vibration, temperature, model, scaler):
    pred_label, proba_dict = predict_state(current, vibration, temperature, model, scaler)
    return pred_label, proba_dict


# ── Batch Prediction ──────────────────────────────────────────────────────────
def predict_batch(samples, model, scaler):
    results = []
    for s in samples:
        label, proba = predict_state(s["current"], s["vibration"], s["temperature"], model, scaler)
        results.append({"input": s, "prediction": label, "probabilities": proba})
    return results


# ── Interactive Mode ──────────────────────────────────────────────────────────
def interactive_mode(model, scaler):
    print("Type 'exit' to quit.\n")
    while True:
        try:
            val = input("Current (A)                   : ").strip()
            if val.lower() == "exit": break
            current = float(val)

            val = input("Vibration (0 = none, 1 = yes) : ").strip()
            if val.lower() == "exit": break
            vibration = int(val)

            val = input("Temperature (°C)              : ").strip()
            if val.lower() == "exit": break
            temperature = float(val)

            label, proba = predict_single(current, vibration, temperature, model, scaler)
            print(f"\n  Prediction    : {label}")
            print(f"  Top match     : {max(proba, key=proba.get)} ({max(proba.values()):.4f})\n")

        except ValueError as e:
            print(f"  Invalid input: {e}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, scaler = load_artifacts()

    label, proba = predict_single(0.314, 0, 27.4, model, scaler)
    print(label, proba)

    results = predict_batch([
        {"current": 0.314, "vibration": 0, "temperature": 27.4},
        {"current": 0.368, "vibration": 1, "temperature": 34.0},
        {"current": 0.382, "vibration": 0, "temperature": 28.3},
        {"current": 0.305, "vibration": 1, "temperature": 35.0},
    ], model, scaler)

    for r in results:
        print(r["prediction"], r["probabilities"])

    interactive_mode(model, scaler)