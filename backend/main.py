import base64

import cv2
import numpy as np

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)

from fastapi.middleware.cors import CORSMiddleware

from backend.face_service import detect_and_crop_face
from backend.gradcam_service import generate_gradcam

from backend.model_service import (
    DEVICE,
    CLASS_TO_IDX,
    VIDEO_THRESHOLD,
    predict_image,
)


# ==========================================================
# FASTAPI APPLICATION
# ==========================================================

app = FastAPI(
    title="Real-Time Deepfake Detection API",
    version="1.0",
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# ROOT ENDPOINT
# ==========================================================

@app.get("/")
def root():
    return {
        "message": "Deepfake Detection API is running"
    }


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "EfficientNet-B0",
        "model_loaded": True,
        "device": str(DEVICE),
        "classes": CLASS_TO_IDX,
        "video_threshold": VIDEO_THRESHOLD,
    }


# ==========================================================
# SINGLE IMAGE PREDICTION
# ==========================================================

@app.post("/predict-image")
async def predict_image_endpoint(
    file: UploadFile = File(...)
):
    if file.content_type not in [
        "image/jpeg",
        "image/png",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported.",
        )

    contents = await file.read()

    np_array = np.frombuffer(
        contents,
        np.uint8,
    )

    image = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read image.",
        )

    # Detect and crop face
    face = detect_and_crop_face(image)

    if face is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in image.",
        )

    # Predict face
    result = predict_image(face)

    return {
        "filename": file.filename,
        "face_detected": True,
        "prediction": result["prediction"],
        "fake_probability": round(
            result["fake_probability"] * 100,
            2,
        ),
        "real_probability": round(
            result["real_probability"] * 100,
            2,
        ),
    }


# ==========================================================
# GRAD-CAM EXPLAINABILITY
# ==========================================================

@app.post("/gradcam")
async def gradcam_endpoint(
    file: UploadFile = File(...)
):
    if file.content_type not in [
        "image/jpeg",
        "image/png",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported.",
        )

    contents = await file.read()

    np_array = np.frombuffer(
        contents,
        np.uint8,
    )

    image = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read image.",
        )

    # Detect and crop face
    face = detect_and_crop_face(image)

    if face is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in image.",
        )

    try:
        # Generate Grad-CAM
        result = generate_gradcam(face)

        overlay = result["overlay"]

        # Convert Grad-CAM image to JPEG
        success, buffer = cv2.imencode(
            ".jpg",
            overlay,
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Could not encode Grad-CAM image.",
            )

        # Convert JPEG to Base64
        image_base64 = base64.b64encode(
            buffer.tobytes()
        ).decode("utf-8")

        return {
            "status": "success",
            "face_detected": True,
            "fake_probability": round(
                result["fake_probability"] * 100,
                2,
            ),
            "gradcam_image": (
                "data:image/jpeg;base64,"
                + image_base64
            ),
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Grad-CAM generation failed: "
                + str(e)
            ),
        )


# ==========================================================
# REAL-TIME DEEPFAKE DETECTION WEBSOCKET
# ==========================================================

@app.websocket("/ws/detect")
async def websocket_detect(
    websocket: WebSocket
):
    await websocket.accept()

    print("Deepfake detection WebSocket connected.")

    # Each connection has its own rolling history
    fake_probability_history = []

    WINDOW_SIZE = 30

    try:
        while True:

            # --------------------------------------------------
            # RECEIVE JPEG FRAME
            # --------------------------------------------------

            frame_bytes = await websocket.receive_bytes()

            np_array = np.frombuffer(
                frame_bytes,
                np.uint8,
            )

            frame = cv2.imdecode(
                np_array,
                cv2.IMREAD_COLOR,
            )

            if frame is None:
                await websocket.send_json({
                    "status": "error",
                    "message": "Invalid image frame",
                })

                continue

            # --------------------------------------------------
            # FACE DETECTION
            # --------------------------------------------------

            face = detect_and_crop_face(frame)

            if face is None:
                await websocket.send_json({
                    "status": "no_face",
                    "face_detected": False,
                    "frames_in_window":
                        len(fake_probability_history),
                    "window_size": WINDOW_SIZE,
                })

                continue

            # --------------------------------------------------
            # EFFICIENTNET FRAME PREDICTION
            # --------------------------------------------------

            result = predict_image(face)

            fake_probability = result[
                "fake_probability"
            ]

            real_probability = result[
                "real_probability"
            ]

            # --------------------------------------------------
            # UPDATE 30-FRAME WINDOW
            # --------------------------------------------------

            fake_probability_history.append(
                fake_probability
            )

            if (
                len(fake_probability_history)
                > WINDOW_SIZE
            ):
                fake_probability_history.pop(0)

            # --------------------------------------------------
            # VIDEO-LEVEL ANALYSIS
            # --------------------------------------------------

            probs = np.array(
                fake_probability_history,
                dtype=np.float32,
            )

            # Mean fake probability
            mean_fake = float(
                np.mean(probs)
            )

            # Top 30% suspicious frames
            top_count = max(
                1,
                int(len(probs) * 0.30),
            )

            sorted_probs = np.sort(probs)

            top_fake_mean = float(
                np.mean(
                    sorted_probs[-top_count:]
                )
            )

            # Percentage of frames classified as fake
            fake_vote_ratio = float(
                np.mean(
                    probs >= 0.50
                )
            )

            # --------------------------------------------------
            # FINAL VIDEO FAKE SCORE
            #
            # 40% Mean Fake Probability
            # 40% Top Suspicious Frames
            # 20% Fake Frame Vote Ratio
            # --------------------------------------------------

            video_fake_score = (
                0.40 * mean_fake
                + 0.40 * top_fake_mean
                + 0.20 * fake_vote_ratio
            )

            # --------------------------------------------------
            # FINAL VIDEO PREDICTION
            # --------------------------------------------------

            if (
                video_fake_score
                >= VIDEO_THRESHOLD
            ):
                video_prediction = "FAKE"

            else:
                video_prediction = "REAL"

            # --------------------------------------------------
            # SEND RESULT TO CLIENT
            # --------------------------------------------------

            await websocket.send_json({
                "status": "success",

                "face_detected": True,

                "frame_prediction":
                    result["prediction"],

                "frame_fake_probability":
                    round(
                        fake_probability * 100,
                        2,
                    ),

                "frame_real_probability":
                    round(
                        real_probability * 100,
                        2,
                    ),

                "mean_fake_probability":
                    round(
                        mean_fake * 100,
                        2,
                    ),

                "top_suspicious_probability":
                    round(
                        top_fake_mean * 100,
                        2,
                    ),

                "fake_vote_ratio":
                    round(
                        fake_vote_ratio * 100,
                        2,
                    ),

                "video_fake_score":
                    round(
                        video_fake_score * 100,
                        2,
                    ),

                "video_real_score":
                    round(
                        (1 - video_fake_score) * 100,
                        2,
                    ),

                "video_prediction":
                    video_prediction,

                "frames_in_window":
                    len(
                        fake_probability_history
                    ),

                "window_size":
                    WINDOW_SIZE,

                "threshold":
                    VIDEO_THRESHOLD,
            })

    except WebSocketDisconnect:

        print(
            "Deepfake detection WebSocket disconnected."
        )

    except Exception as e:

        print(
            "Deepfake detection WebSocket error:",
            str(e),
        )

        try:
            await websocket.send_json({
                "status": "error",
                "message": str(e),
            })

        except Exception:
            pass


# ==========================================================
# WEBRTC SIGNALING
# CANDIDATE <-> HR
# ==========================================================

signaling_rooms = {}


@app.websocket(
    "/ws/signaling/{interview_id}/{role}"
)
async def signaling_websocket(
    websocket: WebSocket,
    interview_id: str,
    role: str,
):
    await websocket.accept()

    # Clean values
    role = role.lower().strip()
    interview_id = interview_id.strip()

    # Only candidate and HR are allowed
    if role not in (
        "candidate",
        "hr",
    ):
        await websocket.close(
            code=1008,
            reason="Invalid role",
        )
        return

    # Interview ID must exist
    if not interview_id:
        await websocket.close(
            code=1008,
            reason="Interview ID required",
        )
        return

    # Create room
    if interview_id not in signaling_rooms:
        signaling_rooms[interview_id] = {}

    room = signaling_rooms[interview_id]

    # If same role reconnects, replace old socket
    old_socket = room.get(role)

    if (
        old_socket is not None
        and old_socket is not websocket
    ):
        try:
            await old_socket.close(
                code=1000,
                reason="New connection established",
            )
        except Exception:
            pass

    # Register current connection
    room[role] = websocket

    print(
        f"{role.upper()} connected to interview "
        f"{interview_id}"
    )

    # Tell this participant who is already present
    other_role = (
        "hr"
        if role == "candidate"
        else "candidate"
    )

    if room.get(other_role) is not None:
        try:
            await websocket.send_json({
                "type": "peer-ready",
                "role": other_role,
                "interview_id": interview_id,
            })
        except Exception:
            pass

    # Tell the other participant that this user joined
    other_socket = room.get(other_role)

    if other_socket is not None:
        try:
            await other_socket.send_json({
                "type": "peer-ready",
                "role": role,
                "interview_id": interview_id,
            })
        except Exception:
            pass

    try:
        while True:

            # Receive signaling JSON/text from this participant
            message = await websocket.receive_text()

            # Candidate -> HR
            # HR -> Candidate
            target_role = (
                "hr"
                if role == "candidate"
                else "candidate"
            )

            target_socket = room.get(
                target_role
            )

            if target_socket is not None:

                try:
                    await target_socket.send_text(
                        message
                    )

                except Exception as e:

                    print(
                        f"Could not send signaling "
                        f"message to {target_role}: "
                        f"{e}"
                    )

                    # Remove stale target socket
                    if (
                        room.get(target_role)
                        is target_socket
                    ):
                        room.pop(
                            target_role,
                            None,
                        )

    except WebSocketDisconnect:

        print(
            f"{role.upper()} disconnected from "
            f"interview {interview_id}"
        )

    except Exception as e:

        print(
            f"Signaling error for {role} "
            f"in {interview_id}: {e}"
        )

    finally:

        # Remove only this exact connection
        if room.get(role) is websocket:
            room.pop(
                role,
                None,
            )

        # Notify remaining participant
        remaining_role = (
            "hr"
            if role == "candidate"
            else "candidate"
        )

        remaining_socket = room.get(
            remaining_role
        )

        if remaining_socket is not None:
            try:
                await remaining_socket.send_json({
                    "type": "peer-left",
                    "role": role,
                    "interview_id": interview_id,
                })
            except Exception:
                pass

        # Remove empty room
        if not room:
            signaling_rooms.pop(
                interview_id,
                None,
            )

        print(
            f"{role.upper()} cleaned up from "
            f"interview {interview_id}"
        )

# ==========================================================
# INTERVIEW HISTORY (SQLite)
# ==========================================================

from pydantic import BaseModel, Field
from typing import Literal
from backend.interview_database import (
    initialize_database, save_interview, list_interviews, get_interview, delete_interview
)

initialize_database()


class InterviewResult(BaseModel):
    interview_id: str = Field(min_length=1, max_length=100)
    candidate_name: str = Field(default="", max_length=150)
    candidate_usn: str = Field(default="", max_length=50)
    prediction: Literal["REAL", "FAKE"]
    risk_score: float = Field(ge=0, le=100, allow_inf_nan=False)
    frames_analyzed: int = Field(ge=0)
    record_type: Literal["LIVE_INTERVIEW", "VIDEO_TEST"] = "LIVE_INTERVIEW"


@app.post("/interviews", status_code=201)
def create_interview_record(data: InterviewResult):
    return save_interview(
        data.interview_id.strip(), data.candidate_name.strip(),
        data.candidate_usn.strip(), data.prediction,
        data.risk_score, data.frames_analyzed, data.record_type
    )


@app.get("/interviews")
def get_interview_history(limit: int = 100):
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    return list_interviews(limit)


@app.get("/interviews/{record_id}")
def get_interview_record(record_id: int):
    record = get_interview(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Interview record not found")
    return record


# ==========================================================
# DOWNLOAD SAVED INTERVIEW PDF REPORT
# ==========================================================

from fastapi.responses import Response
from backend.report_service import build_interview_pdf


@app.get("/interviews/{record_id}/report")
def download_interview_report(record_id: int):
    record = get_interview(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Interview record not found")

    pdf_bytes = build_interview_pdf(record)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="interview_report_{record_id}.pdf"',
            "Cache-Control": "no-store",
        },
    )


# Delete an individual saved interview or video test.
@app.delete("/interviews/{record_id}", status_code=204)
def delete_interview_record(record_id: int):
    if not delete_interview(record_id):
        raise HTTPException(status_code=404, detail="Detection record not found")
    return Response(status_code=204)
