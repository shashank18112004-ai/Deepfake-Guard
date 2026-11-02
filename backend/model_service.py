import torch
import torch.nn as nn
import numpy as np
import cv2

from torchvision import models


MODEL_PATH = "model/efficientnet_b0_deepfake_best.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_deepfake_model():

    print("Loading deepfake model...")

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

    class_to_idx = checkpoint.get(
        "class_to_idx",
        {
            "fake": 0,
            "real": 1
        }
    )

    video_threshold = checkpoint.get(
        "video_threshold",
        0.45
    )

    print("Deepfake model loaded successfully!")
    print("Device:", DEVICE)
    print("Classes:", class_to_idx)
    print("Video threshold:", video_threshold)

    return model, class_to_idx, video_threshold


model, CLASS_TO_IDX, VIDEO_THRESHOLD = load_deepfake_model()


def preprocess_image(image_bgr):

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    image_rgb = cv2.resize(
        image_rgb,
        (224, 224)
    )

    tensor = torch.from_numpy(
        image_rgb
    ).float() / 255.0

    tensor = tensor.permute(
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

    tensor = (
        tensor - mean
    ) / std

    return tensor.unsqueeze(0)


def predict_image(image_bgr):

    tensor = preprocess_image(
        image_bgr
    ).to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    fake_probability = probabilities[
        CLASS_TO_IDX["fake"]
    ].item()

    real_probability = probabilities[
        CLASS_TO_IDX["real"]
    ].item()

    if fake_probability >= 0.50:
        prediction = "FAKE"
    else:
        prediction = "REAL"

    return {
        "prediction": prediction,
        "fake_probability": fake_probability,
        "real_probability": real_probability
    }