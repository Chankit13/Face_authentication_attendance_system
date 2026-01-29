import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from face_embedder import extract_face_embedding
from utils import load_encodings
import os

THRESHOLD = 0.95   # IMPORTANT

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def mark_attendance(name, action):
    file = "attendance.csv"

    # Case 1 & 2: file does not exist OR is empty
    if not os.path.exists(file) or os.stat(file).st_size == 0:
        df = pd.DataFrame(columns=["Name", "Action", "Time"])
    else:
        df = pd.read_csv(file)

    df.loc[len(df)] = [name, action, datetime.now()]
    df.to_csv(file, index=False)

def recognize(action):
    data = load_encodings()
    print("Loaded encodings:", data.keys())

    if len(data) == 0:
        print("No registered faces found!")
        return None

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emb = extract_face_embedding(frame)

        cv2.putText(
            frame,
            "Looking for face... Press ESC to cancel",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if emb is not None:
            for name, saved_emb in data.items():
                dist = cosine_distance(emb, saved_emb)
                print(f"Comparing with {name}, distance={dist:.3f}")

                if dist < THRESHOLD:
                    print("MATCH FOUND:", name, action)
                    mark_attendance(name, action)
                    cap.release()
                    cv2.destroyAllWindows()
                    return name

        cv2.imshow("Recognizing", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return None
