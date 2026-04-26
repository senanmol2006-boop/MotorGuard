import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# ── Label Mapping ─────────────────────────────────────────────────────────────
STATE_MAP = {
    'stable'              : 0,
    'over temp'           : 1,
    'vibrational problem' : 2,
    'vib-temp'            : 3,
    'current_fault_only'  : 4,
    'curr-temp'           : 5,
    'curr-vibration'      : 6,
    'all'                 : 7
}

STATE_LABELS = {v: k for k, v in STATE_MAP.items()}


# ── Load & Prepare Data ───────────────────────────────────────────────────────
def load_data(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['current', 'vibration', 'temperature', 'state']
    df['state'] = df['state'].str.strip()

    X = df.drop(columns=['state'])
    y = df['state'].map(STATE_MAP)

    return X, y


# ── Train ─────────────────────────────────────────────────────────────────────
def train(csv_path="data/main_data.csv"):
    X, y = load_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=6924
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    pickle.dump(model,  open("motor_fault_rf.pkl", "wb"))
    pickle.dump(scaler, open("scaler_rf.pkl", "wb"))
    print("Saved: motor_fault_rf.pkl | scaler_rf.pkl")


# ── Predict (imported by predict_rf.py) ──────────────────────────────────────
def predict_state(current, vibration, temperature, model, scaler):
    sample     = np.array([[current, vibration, temperature]])
    sample     = scaler.transform(sample)
    pred_class = model.predict(sample)[0]
    pred_label = STATE_LABELS[pred_class]
    proba      = model.predict_proba(sample)[0]
    proba_dict = {STATE_LABELS[i]: round(float(p), 4)
                  for i, p in zip(model.classes_, proba)}

    return pred_label, proba_dict


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train("data/main_data.csv")