# utils.py
import pickle
import os

ENCODING_PATH = "faces/encodings.pkl"

def load_encodings():
    if os.path.exists(ENCODING_PATH):
        with open(ENCODING_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_encodings(data):
    os.makedirs("faces", exist_ok=True)
    with open(ENCODING_PATH, "wb") as f:
        pickle.dump(data, f)
