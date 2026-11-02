import cv2
import mediapipe as mp
import os
import sys

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
# Face extraction function
# -----------------------------
def extract_faces(video_path, label, frame_interval=10):

    if label.lower() == "real":
        output_dir = "dataset/processed/real"

    elif label.lower() == "fake":
        output_dir = "dataset/processed/fake"

    else:
        print("Label must be either real or fake")
        return

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Video:", video_path)
    print("Total frames:", total_frames)
    print("FPS:", fps)
    print("Saving every", frame_interval, "frames")

    frame_count = 0
    saved_count = 0

    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Skip frames
            if frame_count % frame_interval != 0:
                continue

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamp required for VIDEO mode
            timestamp_ms = int(
                (frame_count / max(fps, 1)) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if not result.face_landmarks:
                continue

            face_landmarks = result.face_landmarks[0]

            height, width, _ = frame.shape

            x_values = []
            y_values = []

            for landmark in face_landmarks:

                x_values.append(
                    int(landmark.x * width)
                )

                y_values.append(
                    int(landmark.y * height)
                )

            x_min = max(min(x_values), 0)
            x_max = min(max(x_values), width)

            y_min = max(min(y_values), 0)
            y_max = min(max(y_values), height)

            # Add some margin
            margin = 30

            x_min = max(x_min - margin, 0)
            y_min = max(y_min - margin, 0)

            x_max = min(x_max + margin, width)
            y_max = min(y_max + margin, height)

            face_roi = frame[
                y_min:y_max,
                x_min:x_max
            ]

            if face_roi.size == 0:
                continue

            face_resized = cv2.resize(
                face_roi,
                (224, 224)
            )

            filename = os.path.join(
                output_dir,
                f"{label}_{saved_count:06d}.jpg"
            )

            cv2.imwrite(
                filename,
                face_resized
            )

            saved_count += 1

            print(
                f"\rProcessed frame: {frame_count} | "
                f"Saved faces: {saved_count}",
                end=""
            )

    cap.release()

    print("\nProcessing completed.")
    print("Total face images saved:", saved_count)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage:"
        )

        print(
            "python preprocessing/video_preprocessor.py "
            "<video_path> <real/fake>"
        )

        sys.exit()

    video_path = sys.argv[1]
    label = sys.argv[2]

    extract_faces(
        video_path,
        label
    )