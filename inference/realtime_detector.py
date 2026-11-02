import cv2
import time
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp

from torchvision import models


# ==========================================================
# SETTINGS
# ==========================================================

MODEL_PATH = "model/efficientnet_b0_deepfake_best.pth"
FACE_MODEL_PATH = "preprocessing/face_landmarker.task"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WINDOW_SIZE = 30


# ==========================================================
# LOAD EFFICIENTNET MODEL
# ==========================================================

print("Device:", DEVICE)

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("✅ Deepfake model loaded successfully!")

CLASS_TO_IDX = checkpoint.get(
    "class_to_idx",
    {
        "fake": 0,
        "real": 1
    }
)

VIDEO_THRESHOLD = checkpoint.get(
    "video_threshold",
    0.45
)

print("Classes:", CLASS_TO_IDX)
print("Video threshold:", VIDEO_THRESHOLD)


# ==========================================================
# MEDIAPIPE FACE LANDMARKER
# ==========================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================================
# FACE PREPROCESSING
# ==========================================================

def preprocess_face(face_bgr):

    face_rgb = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB
    )

    face_rgb = cv2.resize(
        face_rgb,
        (224, 224)
    )

    face_tensor = torch.from_numpy(
        face_rgb
    ).float() / 255.0

    face_tensor = face_tensor.permute(
        2,
        0,
        1
    )

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    face_tensor = (
        face_tensor - mean
    ) / std

    return face_tensor.unsqueeze(0)


# ==========================================================
# VIDEO-LEVEL RISK SCORE
# ==========================================================

def calculate_video_score(probability_history):

    if len(probability_history) == 0:
        return 0.0

    probs = np.array(
        probability_history
    )

    mean_fake = np.mean(
        probs
    )

    top_count = max(
        1,
        int(len(probs) * 0.30)
    )

    top_fake_mean = np.mean(
        np.sort(probs)[-top_count:]
    )

    fake_vote_ratio = np.mean(
        probs >= 0.50
    )

    video_fake_score = (
        0.40 * mean_fake
        +
        0.40 * top_fake_mean
        +
        0.20 * fake_vote_ratio
    )

    return float(video_fake_score)


# ==========================================================
# START WEBCAM
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

fake_probability_history = []

start_time = time.perf_counter()

print("✅ Webcam started.")
print("Press Q to quit.")


# ==========================================================
# WEBCAM LOOP
# ==========================================================

with FaceLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        height, width = frame.shape[:2]


        # ------------------------------------------
        # Convert webcam frame for MediaPipe
        # ------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ------------------------------------------
        # Monotonic timestamp
        # ------------------------------------------

        timestamp_ms = int(
            (
                time.perf_counter()
                - start_time
            )
            * 1000
        )


        # ------------------------------------------
        # Detect face
        # ------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        if result.face_landmarks:

            landmarks = (
                result.face_landmarks[0]
            )


            xs = [
                int(lm.x * width)
                for lm in landmarks
            ]

            ys = [
                int(lm.y * height)
                for lm in landmarks
            ]


            x1 = max(
                0,
                min(xs)
            )

            y1 = max(
                0,
                min(ys)
            )

            x2 = min(
                width,
                max(xs)
            )

            y2 = min(
                height,
                max(ys)
            )


            # --------------------------------------
            # Add 20% face margin
            # --------------------------------------

            face_width = x2 - x1
            face_height = y2 - y1

            margin_x = int(
                face_width * 0.20
            )

            margin_y = int(
                face_height * 0.20
            )

            x1 = max(
                0,
                x1 - margin_x
            )

            y1 = max(
                0,
                y1 - margin_y
            )

            x2 = min(
                width,
                x2 + margin_x
            )

            y2 = min(
                height,
                y2 + margin_y
            )


            face = frame[
                y1:y2,
                x1:x2
            ]


            if face.size > 0:

                # ----------------------------------
                # Model preprocessing
                # ----------------------------------

                face_tensor = preprocess_face(
                    face
                ).to(DEVICE)


                # ----------------------------------
                # EfficientNet prediction
                # ----------------------------------

                with torch.no_grad():

                    output = model(
                        face_tensor
                    )

                    probabilities = torch.softmax(
                        output,
                        dim=1
                    )[0]

                    fake_probability = (
                        probabilities[
                            CLASS_TO_IDX["fake"]
                        ].item()
                    )

                    real_probability = (
                        probabilities[
                            CLASS_TO_IDX["real"]
                        ].item()
                    )


                # ----------------------------------
                # Rolling history
                # ----------------------------------

                fake_probability_history.append(
                    fake_probability
                )

                if (
                    len(fake_probability_history)
                    > WINDOW_SIZE
                ):
                    fake_probability_history.pop(
                        0
                    )


                # ----------------------------------
                # Video-level risk
                # ----------------------------------

                video_fake_score = (
                    calculate_video_score(
                        fake_probability_history
                    )
                )


                if (
                    video_fake_score
                    >= VIDEO_THRESHOLD
                ):
                    prediction = "FAKE"

                else:
                    prediction = "REAL"


                # ----------------------------------
                # Draw face rectangle
                # ----------------------------------

                if prediction == "FAKE":
                    box_color = (
                        0,
                        0,
                        255
                    )

                else:
                    box_color = (
                        0,
                        255,
                        0
                    )


                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )


                # ----------------------------------
                # Display values
                # ----------------------------------

                cv2.putText(
                    frame,
                    f"Frame FAKE: {fake_probability * 100:.1f}%",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Frame REAL: {real_probability * 100:.1f}%",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Video Risk: {video_fake_score * 100:.1f}%",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Prediction: {prediction}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Frames in window: {len(fake_probability_history)}/{WINDOW_SIZE}",
                    (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )


        else:

            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


        # ------------------------------------------
        # Show webcam
        # ------------------------------------------

        cv2.imshow(
            "Real-Time Deepfake Detector",
            frame
        )


        # ------------------------------------------
        # Q = Quit
        # ------------------------------------------

        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):
            break


# ==========================================================
# CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()

print("Webcam stopped.")