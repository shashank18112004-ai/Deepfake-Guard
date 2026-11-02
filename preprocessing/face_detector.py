import cv2
import mediapipe as mp
import os
import time

# -----------------------------
# MediaPipe setup
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "preprocessing/face_landmarker.task"

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Dataset folders
# -----------------------------
REAL_DIR = "dataset/processed/real"
FAKE_DIR = "dataset/processed/fake"

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press R = Save as REAL")
print("Press F = Save as FAKE")
print("Press Q = Quit")

face_resized = None

with FaceLandmarker.create_from_options(options) as landmarker:

    frame_timestamp_ms = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

        frame_timestamp_ms += 33

        face_resized = None

        if result.face_landmarks:

            face_landmarks = result.face_landmarks[0]

            height, width, _ = frame.shape

            x_coordinates = []
            y_coordinates = []

            for landmark in face_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                x_coordinates.append(x)
                y_coordinates.append(y)

            # Face boundaries
            x_min = max(min(x_coordinates), 0)
            x_max = min(max(x_coordinates), width)

            y_min = max(min(y_coordinates), 0)
            y_max = min(max(y_coordinates), height)

            # Add margin
            margin = 30

            x_min = max(x_min - margin, 0)
            y_min = max(y_min - margin, 0)

            x_max = min(x_max + margin, width)
            y_max = min(y_max + margin, height)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            # Extract ROI
            face_roi = frame[
                y_min:y_max,
                x_min:x_max
            ]

            if face_roi.size > 0:

                face_resized = cv2.resize(
                    face_roi,
                    (224, 224)
                )

                cv2.imshow(
                    "Face ROI - 224x224",
                    face_resized
                )

            cv2.putText(
                frame,
                "FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            "R: REAL | F: FAKE | Q: QUIT",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Deepfake Detection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Save REAL image
        if key == ord("r"):

            if face_resized is not None:

                filename = os.path.join(
                    REAL_DIR,
                    f"real_{int(time.time() * 1000)}.jpg"
                )

                cv2.imwrite(
                    filename,
                    face_resized
                )

                print("Saved REAL:", filename)

            else:
                print("No face available to save.")

        # Save FAKE image
        elif key == ord("f"):

            if face_resized is not None:

                filename = os.path.join(
                    FAKE_DIR,
                    f"fake_{int(time.time() * 1000)}.jpg"
                )

                cv2.imwrite(
                    filename,
                    face_resized
                )

                print("Saved FAKE:", filename)

            else:
                print("No face available to save.")

        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("Webcam stopped.")