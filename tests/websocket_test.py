import asyncio
import json

import cv2
import websockets


async def test_websocket():

    # WebSocket API address
    uri = "ws://127.0.0.1:8000/ws/detect"

    print("Connecting to Deepfake Detection API...")

    async with websockets.connect(uri) as websocket:

        print("Connected successfully!")

        # --------------------------------------------------
        # Load known FAKE Celeb-DF video
        # --------------------------------------------------

        video_path = "dataset/raw/fake/id33_id20_0005.mp4"

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("ERROR: Could not open fake video.")
            return

        print("Fake video started.")
        print("Press Q to stop.")

        # --------------------------------------------------
        # Read video frame by frame
        # --------------------------------------------------

        while True:

            success, frame = cap.read()

            if not success:
                print("\nVideo finished.")
                break

            # Resize frame before sending
            frame = cv2.resize(
                frame,
                (640, 480)
            )

            # --------------------------------------------------
            # Convert frame to JPEG
            # --------------------------------------------------

            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )

            if not success:
                continue

            # --------------------------------------------------
            # Send JPEG frame to FastAPI WebSocket
            # --------------------------------------------------

            await websocket.send(
                buffer.tobytes()
            )

            # --------------------------------------------------
            # Receive prediction from server
            # --------------------------------------------------

            response = await websocket.recv()

            data = json.loads(response)

            # --------------------------------------------------
            # Successful prediction
            # --------------------------------------------------

            if data.get("status") == "success":

                prediction = data["video_prediction"]
                risk = data["video_fake_score"]
                frame_fake = data["frame_fake_probability"]
                window = data["frames_in_window"]

                print(
                    f"\rFrame FAKE: {frame_fake:.2f}%"
                    f" | Video Risk: {risk:.2f}%"
                    f" | Prediction: {prediction}"
                    f" | Window: {window}/30",
                    end=""
                )

                # Red = FAKE
                # Green = REAL
                if prediction == "FAKE":
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                # --------------------------------------------------
                # Display information on video
                # --------------------------------------------------

                cv2.putText(
                    frame,
                    f"Frame Fake: {frame_fake:.1f}%",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Video Risk: {risk:.1f}%",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Prediction: {prediction}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Window: {window}/30",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            # --------------------------------------------------
            # No face detected
            # --------------------------------------------------

            elif data.get("status") == "no_face":

                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # --------------------------------------------------
            # Other server response
            # --------------------------------------------------

            else:

                print(
                    "\nServer response:",
                    data
                )

            # --------------------------------------------------
            # Show video
            # --------------------------------------------------

            cv2.imshow(
                "FAKE Video - WebSocket Deepfake Detection",
                frame
            )

            # Press Q to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # --------------------------------------------------
        # Cleanup
        # --------------------------------------------------

        cap.release()
        cv2.destroyAllWindows()


# --------------------------------------------------
# Start program
# --------------------------------------------------

asyncio.run(test_websocket())