import serial
import time
import numpy as np
import joblib
import pickle
from datetime import datetime, timezone

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
# LOAD MODELS (ONLY FAULT MODEL USED)
# =========================
fault_model = joblib.load("motor_fault_rf.pkl")

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
        print("RAW:", repr(line))

        if not line:
            continue

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
        # FAULT DETECTION (ML MODEL)
        # =========================
        fault_score = int(fault_model.predict(X)[0])

        # =========================
        # HEALTH SCORE (RULE-BASED SYSTEM)
        # =========================
        BASE_SCORE = 100

        current_fault = 35
        vibration_fault = 50
        temp_fault = 15

        # logic thresholds (you can tune later)
        current_unstable = 1 if current > 0.25 else 0
        vibration_unstable = 1 if vibration == 1 else 0
        temp_unstable = 1 if temp > 35 else 0

        deduction = 0

        if current_unstable:
            deduction += current_fault

        if vibration_unstable:
            deduction += vibration_fault

        if temp_unstable:
            deduction += temp_fault

        health_score = max(0, BASE_SCORE - deduction)

        # =========================
        # OUTPUT
        # =========================
        print(f"Current      : {current:.2f} A")
        print(f"Vibration    : {vibration}")
        print(f"Temp         : {temp:.2f} °C")
        print(f"Health Score : {health_score:.2f}")
        print(f"Fault Status : {fault_score}")
        print("-" * 50)

        # =========================
        # TIMESTAMP
        # =========================
        now_iso = datetime.now(timezone.utc).isoformat()

        data = {
            'current': float(current),
            'vibration': float(vibration),
            'temperature': float(temp),
            'score': float(health_score),
            'fault': int(fault_score),
            'timestamp': now_iso,
        }

        # =========================
        # FIREBASE WRITE
        # =========================
        db.collection('PredictionData').document('latest').set(data)

        db.collection('PredictionData').document('latest') \
          .collection('history').add(data)

        print(f"✅ Firestore updated @ {now_iso}")

    # =========================
    # HANDLE DISCONNECT
    # =========================
    except serial.SerialException:
        print("❌ Arduino disconnected. Reconnecting...")
        arduino = connect_serial()

    # =========================
    # HANDLE ERRORS
    # =========================
    except Exception as e:
        print("⚠️ Error:", e, "| Raw line:", line)