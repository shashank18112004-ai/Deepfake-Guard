import cv2
import mediapipe as mp
import time

FACE_MODEL_PATH = "preprocessing/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5
)

landmarker = FaceLandmarker.create_from_options(options)


def detect_and_crop_face(image_bgr):

    height, width = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    result = landmarker.detect(
        mp_image
    )

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]

    xs = [
        int(lm.x * width)
        for lm in landmarks
    ]

    ys = [
        int(lm.y * height)
        for lm in landmarks
    ]

    x1 = max(0, min(xs))
    y1 = max(0, min(ys))
    x2 = min(width, max(xs))
    y2 = min(height, max(ys))

    face_width = x2 - x1
    face_height = y2 - y1

    margin_x = int(face_width * 0.20)
    margin_y = int(face_height * 0.20)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(width, x2 + margin_x)
    y2 = min(height, y2 + margin_y)

    face = image_bgr[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:
        return None

    return face