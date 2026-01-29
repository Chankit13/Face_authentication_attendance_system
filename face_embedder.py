import mediapipe as mp
import cv2
import numpy as np

mp_face = mp.solutions.face_detection

detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7)

def extract_face_embedding(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)

    if not result.detections:
        return None

    detection = result.detections[0]
    bbox = detection.location_data.relative_bounding_box

    h, w, _ = frame.shape
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = int((bbox.xmin + bbox.width) * w)
    y2 = int((bbox.ymin + bbox.height) * h)

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None

    face = cv2.resize(face, (112, 112))
    face = face / 255.0
    return face.flatten()
