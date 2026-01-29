import cv2
from face_embedder import extract_face_embedding
from utils import load_encodings, save_encodings

def register_user(name):
    cap = cv2.VideoCapture(0)
    data = load_encodings()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        embedding = extract_face_embedding(frame)

        # 🔹 Instruction text
        cv2.putText(
            frame,
            "Press 'S' to Save Face | Press 'Q' to Quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and embedding is not None:
            data[name] = embedding
            save_encodings(data)
            break

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
