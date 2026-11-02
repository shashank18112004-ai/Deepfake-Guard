import cv2
import numpy as np
import torch

from backend.model_service import (
    model,
    DEVICE,
    CLASS_TO_IDX,
    preprocess_image
)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._save_activations
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._save_gradients
            )
        )

    def _save_activations(
        self,
        module,
        input,
        output
    ):
        self.activations = output.detach()

    def _save_gradients(
        self,
        module,
        grad_input,
        grad_output
    ):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image_bgr,
        target_class=None
    ):
        # Preprocess face
        input_tensor = preprocess_image(
            image_bgr
        ).to(DEVICE)

        # Grad-CAM needs gradients
        self.model.zero_grad()

        output = self.model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        fake_index = CLASS_TO_IDX["fake"]

        fake_probability = (
            probabilities[fake_index].item()
        )

        # For deepfake explanation we normally
        # explain the FAKE class.
        if target_class is None:
            target_class = fake_index

        score = output[0, target_class]

        score.backward()

        if (
            self.activations is None
            or self.gradients is None
        ):
            raise RuntimeError(
                "Grad-CAM hooks did not capture "
                "activations/gradients."
            )

        # Global-average-pool gradients
        weights = torch.mean(
            self.gradients,
            dim=(2, 3),
            keepdim=True
        )

        # Weighted feature maps
        cam = torch.sum(
            weights * self.activations,
            dim=1
        )

        cam = torch.relu(cam)

        cam = cam[0].cpu().numpy()

        # Normalize 0 -> 1
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (
                (cam - cam_min)
                / (cam_max - cam_min)
            )
        else:
            cam = np.zeros_like(cam)

        # Resize heatmap to face size
        heatmap = cv2.resize(
            cam,
            (
                image_bgr.shape[1],
                image_bgr.shape[0]
            )
        )

        # Convert to 0-255
        heatmap_uint8 = np.uint8(
            255 * heatmap
        )

        # Apply heatmap
        colored_heatmap = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        # Overlay heatmap on original face
        overlay = cv2.addWeighted(
            image_bgr,
            0.60,
            colored_heatmap,
            0.40,
            0
        )

        return {
            "overlay": overlay,
            "heatmap": colored_heatmap,
            "fake_probability": fake_probability
        }


# EfficientNet-B0 final convolution block
target_layer = model.features[-1]

gradcam = GradCAM(
    model=model,
    target_layer=target_layer
)


def generate_gradcam(face_bgr):
    """
    Generate Grad-CAM explanation for
    the FAKE class.
    """

    return gradcam.generate(
        face_bgr,
        target_class=CLASS_TO_IDX["fake"]
    )