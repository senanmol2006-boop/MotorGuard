import serial
import time
import numpy as np
import joblib
import pickle
from tensorflow.keras.models import load_model

import firebase_admin
from firebase_admin import credentials, firestore

# =========================
# FIREBASE SETUP
# =========================
cred = credentials.Certificate(
    "/Users/shineamgain/Downloads/hacka/motorfaultdetection-d9270-firebase-adminsdk-fbsvc-2aa3e83d3f.json"
)
firebase_admin.initialize_app(cred)
db = firestore.client()

# =========================
# LOAD MODELS
# =========================
keras_model = load_model("healthscore_model.keras")

with open("scaler.pkl", "rb") as f:
    keras_scaler = pickle.load(f)

fault_model = joblib.load("motor_fault_rf.pkl")

# =========================
# STATE MAP (IMPORTANT FIX)
# =========================
STATE_MAP = {
    0: "stable",
    1: "over temp",
    2: "vibrational problem",
    3: "vib-temp",
    4: "current_fault_only",
    5: "curr-temp",
    6: "curr-vibration",
    7: "all"
}

# =========================
# SERIAL SETUP
# =========================
PORT = "/dev/cu.usbmodem141011"
BAUD = 9600


def connect_serial():
    while True:
        try:
            ser = serial.Serial(PORT, BAUD, timeout=1)
            time.sleep(2)
            ser.reset_input_buffer()
            print("✅ Arduino Connected")
            return ser
        except Exception as e:
            print("⏳ Waiting for Arduino...", e)
            time.sleep(2)


arduino = connect_serial()

# =========================
# MAIN LOOP
# =========================
while True:
    try:
        line = arduino.readline().decode(errors="ignore").strip()

        if not line:
            continue

        # Expected: current,vibration,temp
        parts = line.split(",")

        if len(parts) != 3:
            print("⚠️ Invalid format:", line)
            continue

        # =========================
        # SENSOR VALUES
        # =========================
        current = float(parts[0])
        vibration = float(parts[1])
        temp = float(parts[2])

        X = np.array([[current, vibration, temp]])

        # =========================
        # KERAS MODEL (FIXED)
        # =========================
        X_scaled = keras_scaler.transform(X)

        keras_raw = keras_model.predict(X_scaled, verbose=0)[0]

        keras_class = int(np.argmax(keras_raw))
        keras_score = float(np.max(keras_raw) * 100)
        state_label = STATE_MAP[keras_class]

        # =========================
        # RANDOM FOREST (FAULT MODEL)
        # =========================
        fault_score = int(fault_model.predict(X)[0])

        # =========================
        # OPTIONAL FUSION (IMPROVES REALISM)
        # =========================
        final_health = keras_score

        if fault_score != 0:
            final_health *= 0.7  # reduce confidence under fault condition

        # =========================
        # OUTPUT
        # =========================
        print(f"Current       : {current:.2f} A")
        print(f"Vibration     : {vibration}")
        print(f"Temp          : {temp:.2f} °C")
        print(f"Health Score  : {final_health:.2f}")
        print(f"State         : {state_label}")
        print(f"Fault Status  : {fault_score}")
        print("-" * 50)

        # =========================
        # FIREBASE WRITE
        # =========================
        data = {
            'current': float(current),
            'vibration': float(vibration),
            'temperature': float(temp),

            'health_score': float(final_health),
            'predicted_state': state_label,

            'fault_class': int(fault_score)
        }

        db.collection('PredictionData').document('latest').set(data)

        print("📄 Updated Firestore")

    # =========================
    # RECONNECT HANDLING
    # =========================
    except serial.SerialException:
        print("❌ Arduino disconnected. Reconnecting...")
        arduino = connect_serial()

    # =========================
    # ERROR HANDLING
    # =========================
    except Exception as e:
        print("⚠️ Error:", e, "| Raw line:", line)