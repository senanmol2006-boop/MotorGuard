import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ── Label Mapping ─────────────────────────────────────────
STATE_MAP = {
    'stable': 0,
    'over temp': 1,
    'vibrational problem': 2,
    'vib-temp': 3,
    'current_fault_only': 4,
    'curr-temp': 5,
    'curr-vibration': 6,
    'all': 7
}

# ── 1. Load Data ──────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path, header=None)
    df.columns = ['current', 'vibration', 'temp', 'state']
    df['state'] = df['state'].str.strip()

    X = df[['current', 'vibration', 'temp']]
    y = df['state'].map(STATE_MAP)

    return X, y


# ── 2. Scale ──────────────────────────────────────────────
def scale_data(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return X_scaled, scaler


# ── 3. Model ──────────────────────────────────────────────
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model


# ── 4. Train ──────────────────────────────────────────────
def train(model, X_train, y_train):
    early_stop = EarlyStopping(patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    return history


# ── 5. Save ───────────────────────────────────────────────
def save(model):
    model.save("healthscore_model.keras")


# ── 6. Predict (FIXED HEALTH LOGIC ONLY) ──────────────────
def predict_health(current, vibration, temp, model, scaler):
    sample = np.array([[current, vibration, temp]])
    sample_scaled = scaler.transform(sample)

    # model predicts fault class (0–7 approx)
    pred = model.predict(sample_scaled, verbose=0)[0][0]

    pred_class = int(round(pred))
    pred_class = max(0, min(7, pred_class))

    # ── HEALTH SCORE MAPPING (YOUR SYSTEM LOGIC) ──
    HEALTH_MAP = {
        0: 100,  # stable
        1: 85,   # over temp
        2: 65,   # vibration
        3: 50,   # vib-temp
        4: 65,   # current fault only
        5: 70,   # curr-temp
        6: 50,   # curr-vibration
        7: 0     # all faults
    }

    return float(HEALTH_MAP[pred_class])


# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    X, y = load_data("data/main_data.csv")

    X_scaled, scaler = scale_data(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = build_model(X_train.shape[1])

    train(model, X_train, y_train)

    save(model)

    print("✅ Training complete. Model + scaler saved.")