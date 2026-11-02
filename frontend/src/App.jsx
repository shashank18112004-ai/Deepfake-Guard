import { useRef, useState } from "react";

import "./App.css";

function App() {

  const [status, setStatus] = useState("READY");

  const [cameraStarted, setCameraStarted] = useState(false);

  const [apiConnected, setApiConnected] = useState(false);

  const [frameFake, setFrameFake] = useState(0);

  const [frameReal, setFrameReal] = useState(0);

  const [videoRisk, setVideoRisk] = useState(0);

  const [frames, setFrames] = useState(0);

  const [testVideoMode, setTestVideoMode] = useState(false);

  const [interviewId, setInterviewId] = useState("");
  const [candidateName, setCandidateName] = useState("");
  const [candidateUsn, setCandidateUsn] = useState("");
  const [saveMessage, setSaveMessage] = useState("");
  const [currentInterviewRecord, setCurrentInterviewRecord] = useState(null);
  const [savingInterview, setSavingInterview] = useState(false);
  const latestResultRef = useRef(null);
  const saveLockRef = useRef(false);
  const pendingSaveRef = useRef(null);
  const videoTestIdRef = useRef("");

  const [hrConnectionStatus, setHrConnectionStatus] = useState("Not Joined");

  // Grad-CAM states

  const [gradcamImage, setGradcamImage] = useState(null);

  const [gradcamLoading, setGradcamLoading] = useState(false);

  const [gradcamFakeProbability, setGradcamFakeProbability] =

    useState(null);

  const videoRef = useRef(null);

  const hrLocalVideoRef = useRef(null);

  const hrStreamRef = useRef(null);

  const streamRef = useRef(null);

  const socketRef = useRef(null);

  const canvasRef = useRef(null);

  const intervalRef = useRef(null);

  const signalingSocketRef = useRef(null);

  const peerRef = useRef(null);

  const pendingIceRef = useRef([]);

  // Prevent repeated Grad-CAM requests

  const gradcamRequestedRef = useRef(false);

  // =====================================================

  // RESET DETECTION VALUES

  // =====================================================

  const resetDetection = () => {
    latestResultRef.current = null;

    setFrameFake(0);

    setFrameReal(0);

    setVideoRisk(0);

    setFrames(0);

    setGradcamImage(null);

    setGradcamLoading(false);

    setGradcamFakeProbability(null);

    gradcamRequestedRef.current = false;

  };

  // =====================================================

  // GENERATE GRAD-CAM

  // =====================================================

  const generateGradcam = async () => {

  const video = videoRef.current;

  const canvas = canvasRef.current;

  if (!video || !canvas) {

    console.error("Video or canvas not available.");

    gradcamRequestedRef.current = false;

    return;

  }

  if (

    video.readyState < 2 ||

    video.videoWidth === 0 ||

    video.videoHeight === 0

  ) {

    console.error("Video frame is not ready.");

    gradcamRequestedRef.current = false;

    return;

  }

  const context = canvas.getContext("2d");

  if (!context) {

    console.error("Canvas context not available.");

    gradcamRequestedRef.current = false;

    return;

  }

  setGradcamLoading(true);

  try {

    canvas.width = 640;

    canvas.height = 480;

    context.drawImage(

      video,

      0,

      0,

      canvas.width,

      canvas.height

    );

    const blob = await new Promise((resolve) => {

      canvas.toBlob(

        resolve,

        "image/jpeg",

        0.9

      );

    });

    if (!blob) {

      throw new Error(

        "Could not capture video frame."

      );

    }

    console.log(

      "Grad-CAM frame captured:",

      blob.size,

      "bytes"

    );

    const formData = new FormData();

    formData.append(

      "file",

      blob,

      "gradcam_frame.jpg"

    );

    console.log(

      "Sending frame to Grad-CAM API..."

    );

    const controller =

      new AbortController();

    const timeout = setTimeout(() => {

      controller.abort();

    }, 30000);

    const response = await fetch(

      "http://127.0.0.1:8000/gradcam",

      {

        method: "POST",

        body: formData,

        signal: controller.signal,

      }

    );

    clearTimeout(timeout);

    console.log(

      "Grad-CAM HTTP status:",

      response.status

    );

    if (!response.ok) {

      const errorText =

        await response.text();

      throw new Error(

        `Grad-CAM API error ${response.status}: ${errorText}`

      );

    }

    const data = await response.json();

    console.log(

      "Grad-CAM response received:",

      data

    );

    if (

      data.status === "success" &&

      data.gradcam_image

    ) {

      setGradcamImage(

        data.gradcam_image

      );

      setGradcamFakeProbability(

        Number(

          data.fake_probability

        ) || 0

      );

      console.log(

        "Grad-CAM image displayed successfully."

      );

    } else {

      throw new Error(

        "Grad-CAM image missing from API response."

      );

    }

  } catch (error) {

    console.error(

      "Grad-CAM generation failed:",

      error

    );

    gradcamRequestedRef.current = false;

    if (

      error.name === "AbortError"

    ) {

      alert(

        "Grad-CAM request took more than 30 seconds."

      );

    }

  } finally {

    // Very important:

    // spinner must always stop.

    setGradcamLoading(false);

  }

};

  // =====================================================

  // HANDLE WEBSOCKET RESULT

  // =====================================================

  const handleDetectionResult = (event) => {

    try {

      const data = JSON.parse(event.data);

      console.log(

        "Detection result:",

        data

      );

      if (data.status === "success") {

        const fakeProbability =

          Number(

            data.frame_fake_probability

          ) || 0;

        const realProbability =

          Number(

            data.frame_real_probability

          ) || 0;

        const risk =

          Number(

            data.video_fake_score

          ) || 0;

        const frameCount =

          Number(

            data.frames_in_window

          ) || 0;

        const prediction =

          data.video_prediction ||

          "ANALYZING";

        setFrameFake(fakeProbability);

        setFrameReal(realProbability);

        setVideoRisk(risk);

        setFrames(frameCount);

        setStatus(prediction);
        // Store the latest complete analysis independently of React's render cycle.
        if (frameCount >= 30 && (prediction === "REAL" || prediction === "FAKE")) {
          latestResultRef.current = {
            prediction,
            risk_score: risk,
            frames_analyzed: frameCount,
          };
        }

        // =============================================

        // GENERATE GRAD-CAM ONLY AFTER FAKE DETECTION

        // =============================================

        if (

          prediction === "FAKE" &&

          frameCount >= 30 &&

          !gradcamRequestedRef.current

        ) {

          gradcamRequestedRef.current = true;

          generateGradcam();

        }

      } else if (

        data.status === "no_face"

      ) {

        setStatus("NO FACE");

      } else if (

        data.status === "error"

      ) {

        console.error(

          "Backend error:",

          data.message

        );

      }

    } catch (error) {

      console.error(

        "Could not read WebSocket response:",

        error

      );

    }

  };

  // =====================================================

  // CONNECT WEBSOCKET

  // =====================================================

  const connectWebSocket = () => {

    return new Promise(

      (resolve, reject) => {

        if (socketRef.current) {

          try {

            socketRef.current.close();

          } catch {

            // Ignore close error

          }

          socketRef.current = null;

        }

        const socket = new WebSocket(

          "ws://127.0.0.1:8000/ws/detect"

        );

        socketRef.current = socket;

        socket.onopen = () => {

          console.log(

            "WebSocket connected successfully."

          );

          setApiConnected(true);

          setStatus("ANALYZING");

          resolve(socket);

        };

        socket.onmessage =

          handleDetectionResult;

        socket.onerror = (error) => {

          console.error(

            "WebSocket error:",

            error

          );

          setApiConnected(false);

          setStatus("API ERROR");

          reject(error);

        };

        socket.onclose = () => {

          console.log(

            "WebSocket disconnected."

          );

          setApiConnected(false);

        };

      }

    );

  };

  // =====================================================

  // SEND VIDEO FRAMES TO FASTAPI

  // =====================================================

  const startSendingFrames = () => {

    if (intervalRef.current) {

      clearInterval(

        intervalRef.current

      );

    }

    intervalRef.current =

      setInterval(() => {

        const video =

          videoRef.current;

        const socket =

          socketRef.current;

        const canvas =

          canvasRef.current;

        if (

          !video ||

          !canvas ||

          !socket

        ) {

          return;

        }

        if (

          socket.readyState !==

          WebSocket.OPEN

        ) {

          return;

        }

        if (

          video.readyState < 2 ||

          video.videoWidth === 0 ||

          video.videoHeight === 0

        ) {

          return;

        }

        const context =

          canvas.getContext("2d");

        if (!context) {

          return;

        }

        canvas.width = 640;

        canvas.height = 480;

        try {

          context.drawImage(

            video,

            0,

            0,

            canvas.width,

            canvas.height

          );

        } catch (error) {

          console.error(

            "Could not capture video frame:",

            error

          );

          return;

        }

        canvas.toBlob(

          (blob) => {

            const currentSocket =

              socketRef.current;

            if (

              blob &&

              currentSocket &&

              currentSocket.readyState ===

                WebSocket.OPEN

            ) {

              currentSocket.send(blob);

            }

          },

          "image/jpeg",

          0.8

        );

      }, 500);

  };

  // =====================================================

  // JOIN CANDIDATE WEBRTC INTERVIEW AS HR

  // =====================================================

  const createHrPeerConnection = (hrStream) => {
  if (peerRef.current) return peerRef.current;

  const peer = new RTCPeerConnection({
    iceServers: [
      { urls: "stun:stun.l.google.com:19302" },
    ],
  });

  peerRef.current = peer;

  // Send HR camera and microphone to the candidate.
  if (hrStream) {
    hrStream.getTracks().forEach((track) => {
      peer.addTrack(track, hrStream);
    });
  }

  peer.onicecandidate = (event) => {
    const signaling = signalingSocketRef.current;

    if (
      event.candidate &&
      signaling?.readyState === WebSocket.OPEN
    ) {
      signaling.send(
        JSON.stringify({
          type: "candidate",
          candidate: event.candidate,
        })
      );
    }
  };

  // Receive the candidate's camera stream.
  peer.ontrack = async (event) => {
    const remoteStream =
      event.streams?.[0] ||
      new MediaStream([event.track]);

    const video = videoRef.current;
    if (!video) return;

    streamRef.current = remoteStream;
    video.srcObject = remoteStream;
    // Unmute the candidate's remote stream so HR can hear the candidate.
    // The separate local HR preview remains muted to prevent echo.
    video.muted = false;
    video.playsInline = true;

    try {
      await video.play();
    } catch (error) {
      console.warn("Candidate video playback:", error);
    }

    setCameraStarted(true);
    setHrConnectionStatus("Candidate Connected");
    setStatus("CONNECTING");

    try {
      if (!socketRef.current ||
          socketRef.current.readyState === WebSocket.CLOSED) {
        await connectWebSocket();
        startSendingFrames();
      }
    } catch (error) {
      console.error("Detection connection failed:", error);
      setStatus("API ERROR");
    }
  };

  peer.onconnectionstatechange = () => {
    console.log("HR WebRTC state:", peer.connectionState);

    if (peer.connectionState === "connected") {
      setHrConnectionStatus("Candidate Connected");
    } else if (peer.connectionState === "connecting") {
      setHrConnectionStatus("Connecting");
    } else if (
      peer.connectionState === "disconnected" ||
      peer.connectionState === "failed"
    ) {
      setHrConnectionStatus("Candidate Disconnected");
    }
  };

  return peer;
};

  const handleHrSignalingMessage = async (event) => {

    try {

      const data = JSON.parse(event.data);

      const peer = peerRef.current;

      console.log("HR signaling message:", data);

      if (!peer) return;

      if (data.type === "peer-ready" && data.role === "candidate") {

        setHrConnectionStatus("Candidate Found");

        return;

      }

      if (data.type === "offer" && data.offer) {

        await peer.setRemoteDescription(new RTCSessionDescription(data.offer));

        for (const candidate of pendingIceRef.current) {

          await peer.addIceCandidate(new RTCIceCandidate(candidate));

        }

        pendingIceRef.current = [];

        const answer = await peer.createAnswer();

        await peer.setLocalDescription(answer);

        const signaling = signalingSocketRef.current;

        if (signaling?.readyState === WebSocket.OPEN) {

          signaling.send(JSON.stringify({ type: "answer", answer: peer.localDescription }));

        }

        setHrConnectionStatus("Connecting");

        return;

      }

      if (data.type === "candidate" && data.candidate) {

        if (peer.remoteDescription) {

          await peer.addIceCandidate(new RTCIceCandidate(data.candidate));

        } else {

          pendingIceRef.current.push(data.candidate);

        }

        return;

      }

      if (data.type === "peer-left" && data.role === "candidate") {

        setHrConnectionStatus("Candidate Left");

        setStatus("READY");

      }

    } catch (error) {

      console.error("HR signaling error:", error);

      setHrConnectionStatus("Signaling Error");

    }

  };

  const startInterview = async () => {

    const cleanInterviewId = interviewId.trim();

    if (!cleanInterviewId) {

      alert("Please enter the Interview ID.");

      return;

    }

    try {

      setCurrentInterviewRecord(null);
      resetDetection();
      pendingSaveRef.current = null;
      setSaveMessage("");

      setTestVideoMode(false);

      setStatus("READY");

      setHrConnectionStatus("Joining Room");

      const video = videoRef.current;

      if (video) {

        video.pause();

        video.removeAttribute("src");

        video.srcObject = null;

      }

      // Request the HR camera/microphone before creating the peer connection.
      // Creating the peer first would leave it without HR tracks.
      const hrStream = await navigator.mediaDevices.getUserMedia({
  video: { width: { ideal: 640 }, height: { ideal: 480 } },
  audio: true,
});

hrStreamRef.current = hrStream;

if (hrLocalVideoRef.current) {
  hrLocalVideoRef.current.srcObject = hrStream;
  hrLocalVideoRef.current.muted = true;
  await hrLocalVideoRef.current.play();
}

createHrPeerConnection(hrStream);

      const encodedId = encodeURIComponent(cleanInterviewId);

      const signaling = new WebSocket(`ws://127.0.0.1:8000/ws/signaling/${encodedId}/hr`);

      signalingSocketRef.current = signaling;

      signaling.onopen = () => {

        console.log("HR joined signaling room:", cleanInterviewId);

        setHrConnectionStatus("Waiting for Candidate");

      };

      signaling.onmessage = handleHrSignalingMessage;

      signaling.onerror = (error) => {

        console.error("HR signaling WebSocket error:", error);

        setHrConnectionStatus("Server Error");

      };

      signaling.onclose = () => console.log("HR signaling WebSocket closed.");

    } catch (error) {

      console.error("Could not join interview:", error);

      setHrConnectionStatus("Error");

      setStatus("ERROR");

    }

  };

  // =====================================================

  // START KNOWN FAKE VIDEO TEST

  // =====================================================

  const startFakeVideoTest =

    async () => {

      try {

        resetDetection();
        pendingSaveRef.current = null;
        setSaveMessage("");
        setCurrentInterviewRecord(null);
        videoTestIdRef.current = `TEST-${Date.now()}`;

        setTestVideoMode(true);

        setCameraStarted(true);

        setStatus("LOADING VIDEO");

        if (streamRef.current) {
      // The candidate owns remote tracks; close the peer connection instead.
      streamRef.current = null;

        }

        const video =

          videoRef.current;

        if (!video) {

          throw new Error(

            "Video element not available."

          );

        }

        video.pause();

        video.srcObject = null;

        video.src =

          "/test_videos/id33_id20_0005_browser.mp4";

        video.loop = true;

        video.muted = true;

        video.playsInline = true;

        await new Promise(

          (resolve, reject) => {

            const handleLoaded =

              () => {

                cleanup();

                resolve();

              };

            const handleError =

              () => {

                cleanup();

                reject(

                  new Error(

                    "Fake test video could not be loaded."

                  )

                );

              };

            const cleanup = () => {

              video.removeEventListener(

                "loadeddata",

                handleLoaded

              );

              video.removeEventListener(

                "error",

                handleError

              );

            };

            video.addEventListener(

              "loadeddata",

              handleLoaded

            );

            video.addEventListener(

              "error",

              handleError

            );

            video.load();

          }

        );

        await video.play();

        console.log(

          "Fake test video started successfully."

        );

        await connectWebSocket();

        startSendingFrames();

        console.log(

          "Fake-video detection started."

        );

      } catch (error) {

        console.error(

          "Fake video test error:",

          error

        );

        setStatus("VIDEO ERROR");

        alert(

          "Could not start the fake test video. Check frontend/public/test_videos/id33_id20_0005_browser.mp4."

        );

      }

    };

  // =====================================================

  // STOP INTERVIEW / TEST

  // =====================================================

  // Save a completed live interview or a completed video test.
  const saveCompletedInterview = async (payload) => {
    if (saveLockRef.current) return false;
    saveLockRef.current = true;
    setSavingInterview(true);
    setSaveMessage("Saving detection result...");

    try {
      const response = await fetch("http://127.0.0.1:8000/interviews", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(`Database API returned HTTP ${response.status}`);
      }
      const saved = await response.json();
      pendingSaveRef.current = null;
      setCurrentInterviewRecord(saved);
      setSaveMessage(`${saved.record_type === "VIDEO_TEST" ? "Video test" : "Interview"} saved successfully (record #${saved.id}).`);
      return true;
    } catch (error) {
      console.error("Could not save interview:", error);
      pendingSaveRef.current = payload;
      setSaveMessage("Save failed. Click Retry Save after checking the backend.");
      return false;
    } finally {
      saveLockRef.current = false;
      setSavingInterview(false);
    }
  };

  const retrySave = async () => {
    if (pendingSaveRef.current) {
      await saveCompletedInterview(pendingSaveRef.current);
    }
  };

  const stopInterview = async () => {
    if (saveLockRef.current) return;

    // Snapshot BEFORE resetDetection() clears the last result.
    const completedResult = latestResultRef.current;
    const shouldSave = completedResult && (testVideoMode || interviewId.trim());
    if (shouldSave) {
      const payload = {
        interview_id: testVideoMode ? videoTestIdRef.current : interviewId.trim(),
        candidate_name: testVideoMode ? "Known Fake Video Test" : candidateName.trim(),
        candidate_usn: testVideoMode ? "" : candidateUsn.trim(),
        record_type: testVideoMode ? "VIDEO_TEST" : "LIVE_INTERVIEW",
        ...completedResult,
      };
      await saveCompletedInterview(payload);
    } else {
      setSaveMessage(`${testVideoMode ? "Video test" : "Interview"} ended without a completed 30-frame result; no record saved.`);
    }

    if (intervalRef.current) {

      clearInterval(

        intervalRef.current

      );

      intervalRef.current = null;

    }

    if (socketRef.current) {

      try {

        socketRef.current.close();

      } catch {

        // Ignore close error

      }

      socketRef.current = null;

    }

    if (signalingSocketRef.current) {

      try { signalingSocketRef.current.close(); } catch {}

      signalingSocketRef.current = null;

    }

    if (peerRef.current) {

      try { peerRef.current.close(); } catch {}

      peerRef.current = null;

    }

    pendingIceRef.current = [];

    if (streamRef.current) {
      // The candidate owns remote tracks; close the peer connection instead.
      streamRef.current = null;

    }
    // Stop HR camera and microphone
if (hrStreamRef.current) {
  hrStreamRef.current.getTracks().forEach((track) => {
    track.stop();
  });
  hrStreamRef.current = null;
}

// Clear HR local video preview
if (hrLocalVideoRef.current) {
  hrLocalVideoRef.current.srcObject = null;
}

    if (videoRef.current) {

      videoRef.current.pause();

      videoRef.current.srcObject =

        null;

      videoRef.current.removeAttribute(

        "src"

      );

      videoRef.current.load();

    }

    setTestVideoMode(false);

    setCameraStarted(false);

    setApiConnected(false);

    setStatus("READY");

    setHrConnectionStatus("Not Joined");

    resetDetection();

    console.log(

      "Interview/test stopped."

    );

  };

  // =====================================================

  // PREDICTION CSS CLASS

  // =====================================================

  const getPredictionClass = () => {

    if (status === "REAL") {

      return "prediction real";

    }

    if (status === "FAKE") {

      return "prediction fake";

    }

    return "prediction";

  };

  // =====================================================

  // STATUS DESCRIPTION

  // =====================================================

  const getStatusMessage = () => {

    if (!cameraStarted) {

      return "Waiting for interview";

    }

    if (status === "REAL") {

      return "No manipulation detected";

    }

    if (status === "FAKE") {

      return "Potential manipulation detected";

    }

    if (status === "NO FACE") {

      return "Face not detected";

    }

    if (

      status === "LOADING VIDEO"

    ) {

      return "Loading test video...";

    }

    if (

      status === "CONNECTING"

    ) {

      return "Connecting to detection API...";

    }

    if (

      status === "API ERROR"

    ) {

      return "Detection API connection error";

    }

    if (

      status === "VIDEO ERROR"

    ) {

      return "Could not load test video";

    }

    return "Analyzing interview\\...";

  };

  return (

    <div className="dashboard">

      {/* ================= HEADER ================= */}

      <header className="header">

        <div>

          <h1>Deepfake Guard</h1>

          <p>

            Corporate HR Video Interview

            Security

          </p>

        </div>

        <div className="connection">

          <span

            className={

              apiConnected

                ? "connection-dot connected"

                : "connection-dot disconnected"

            }

          ></span>

          {apiConnected

            ? "API Connected"

            : "API Ready"}

        </div>

        <a href="/history" style={{color:"#fff",fontWeight:600,textDecoration:"none",padding:"9px 15px",border:"1px solid #75518e",borderRadius:10}}>All Interview History →</a>
      </header>

      {/* ================= MAIN ================= */}

      <main className="main-content">

        {/* ============= VIDEO CARD ============= */}

        <section className="video-card">

          <div className="card-title">

            <div>

              <h2>

                {testVideoMode

                  ? "Fake Video Test"

                  : "Live Interview"}

              </h2>

              <p>

                {testVideoMode

                  ? "Known Celeb-DF manipulation test"

                  : "Candidate video monitoring"}

              </p>

            </div>

            <span className="live-badge">

              {testVideoMode

                ? "TEST"

                : "LIVE"}

            </span>

          </div>

          
          {!testVideoMode && (
            <div className="hr-interview-controls interview-form">
              <div className="form-group">
                <label htmlFor="hr-interview-id">
                  Interview ID
                </label>

                <div className="hr-interview-row">
                  <input
                    id="hr-interview-id"
                    type="text"
                    placeholder="Example: INT-2026-001"
                    value={interviewId}
                    onChange={(e) => setInterviewId(e.target.value)}
                    disabled={
                      cameraStarted ||
                      hrConnectionStatus === "Joining Room" ||
                      hrConnectionStatus === "Waiting for Candidate"
                    }
                  />

                  <span>{hrConnectionStatus}</span>
                </div>
              </div>

              <div className="candidate-fields">
                <div className="form-group">
                  <label htmlFor="candidate-name">
                    Candidate Name
                    <span className="optional-label"> (Optional)</span>
                  </label>

                  <input
                    id="candidate-name"
                    type="text"
                    placeholder="Enter candidate name"
                    value={candidateName}
                    onChange={(e) => setCandidateName(e.target.value)}
                    disabled={cameraStarted || savingInterview}
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="candidate-usn">
                    Candidate USN
                    <span className="optional-label"> (Optional)</span>
                  </label>

                  <input
                    id="candidate-usn"
                    type="text"
                    placeholder="Enter candidate USN"
                    value={candidateUsn}
                    onChange={(e) => setCandidateUsn(e.target.value)}
                    disabled={cameraStarted || savingInterview}
                  />
                </div>
              </div>
            </div>
          )}


          <div className="video-container">

            <video

              ref={videoRef}

              autoPlay

              playsInline

              className="live-video"

              style={{

                display: cameraStarted

                  ? "block"

                  : "none",

              }}

            />

            {!cameraStarted && (

              <div className="camera-placeholder">

                <div className="camera-icon">

                  ◉

                </div>

                <h3>

                  Camera not started

                </h3>

                <p>

                  Start the interview to

                  begin real-time analysis

                </p>

              </div>

            )}

           {!testVideoMode && (
             <div className="hr-local-preview">
             <video
               ref={hrLocalVideoRef}
               autoPlay
               playsInline
               muted
              />
              <span>HR — You</span>
             </div>
             )} 

          </div>

          <canvas

            ref={canvasRef}

            style={{

              display: "none",

            }}

          />

          <div className="buttons">

            <button

              className="start-btn"

              onClick={startInterview}

              disabled={cameraStarted || hrConnectionStatus === "Joining Room" || hrConnectionStatus === "Waiting for Candidate" || hrConnectionStatus === "Connecting" || hrConnectionStatus === "Candidate Found"}

            >

              Start Interview

            </button>

            <button

              className="test-fake-btn"

              onClick={

                startFakeVideoTest

              }

              disabled={cameraStarted}

            >

              Test Fake Video

            </button>

            <button

              className="stop-btn"

              onClick={stopInterview}

              disabled={savingInterview || (!cameraStarted && !signalingSocketRef.current && !testVideoMode)}

            >

              Stop Interview

            </button>

          </div>

          {saveMessage && (
            <div role="status" style={{ marginTop: 12, fontSize: 14 }}>
              {saveMessage}
              {pendingSaveRef.current && (
                <button
                  type="button"
                  onClick={retrySave}
                  disabled={savingInterview}
                  style={{ marginLeft: 10 }}
                >
                  Retry Save
                </button>
              )}
            </div>
          )}
        </section>

        {/* =========== DETECTION CARD =========== */}

        <section className="detection-card">

          <div className="card-title">

            <div>

              <h2>

                Detection Status

              </h2>

              <p>

                Real-time manipulation

                analysis

              </p>

            </div>

          </div>

          {/* Prediction */}

          <div className="prediction-box">

            <p>

              Current Prediction

            </p>

            <h2

              className={

                getPredictionClass()

              }

            >

              {status}

            </h2>

            <span>

              {getStatusMessage()}

            </span>

          </div>

          {/* Risk Score */}

          <div className="metric">

            <div className="metric-header">

              <span>

                Video Risk Score

              </span>

              <strong>

                {videoRisk.toFixed(1)}%

              </strong>

            </div>

            <div className="progress">

              <div

                className="progress-value"

                style={{

                  width: `${Math.min(

                    Math.max(

                      videoRisk,

                      0

                    ),

                    100

                  )}%`,

                }}

              ></div>

            </div>

          </div>

          {/* Statistics */}

          <div className="stats">

            <div className="stat-card">

              <span>

                Frame Fake

              </span>

              <strong>

                {frameFake.toFixed(1)}%

              </strong>

            </div>

            <div className="stat-card">

              <span>

                Frame Real

              </span>

              <strong>

                {frameReal.toFixed(1)}%

              </strong>

            </div>

            <div className="stat-card">

              <span>

                Frames

              </span>

              <strong>

                {frames} / 30

              </strong>

            </div>

            <div className="stat-card">

              <span>

                Threshold

              </span>

              <strong>

                45%

              </strong>

            </div>

          </div>

          {/* Detection Pipeline */}

          <div className="system-info">

            <h3>

              Detection Pipeline

            </h3>

            <div className="pipeline-item">

              <span>

                Face Detection

              </span>

              <strong>

                MediaPipe

              </strong>

            </div>

            <div className="pipeline-item">

              <span>

                Deep Learning Model

              </span>

              <strong>

                EfficientNet-B0

              </strong>

            </div>

            <div className="pipeline-item">

              <span>

                Analysis Window

              </span>

              <strong>

                30 Frames

              </strong>

            </div>

            <div className="pipeline-item">

              <span>

                Explainability

              </span>

              <strong>

                Grad-CAM

              </strong>

            </div>

          </div>

          {/* =====================================

              GRAD-CAM EXPLAINABILITY

          ====================================== */}

          {(status === "FAKE" ||

            gradcamLoading ||

            gradcamImage) && (

            <div className="gradcam-section">

              <div className="gradcam-header">

                <div>

                  <h3>

                    Grad-CAM Explanation

                  </h3>

                  <p>

                    Regions influencing the

                    deepfake prediction

                  </p>

                </div>

                <span className="gradcam-badge">

                  AI Explanation

                </span>

              </div>

              {gradcamLoading && (

                <div className="gradcam-loading">

                  <div className="gradcam-spinner"></div>

                  <p>

                    Generating Grad-CAM

                    explanation...

                  </p>

                </div>

              )}

              {!gradcamLoading &&

                gradcamImage && (

                  <>

                    <div className="gradcam-image-container">

                      <img

                        src={

                          gradcamImage.startsWith(

                            "data:"

                          )

                            ? gradcamImage

                            : `data:image/jpeg;base64,${gradcamImage}`

                        }

                        alt="Grad-CAM explanation"

                        className="gradcam-image"

                      />

                    </div>

                    <div className="gradcam-info">

                      <div>

                        <span>

                          Explanation

                        </span>

                        <strong>

                          Grad-CAM

                        </strong>

                      </div>

                      <div>

                        <span>

                          Model

                        </span>

                        <strong>

                          EfficientNet-B0

                        </strong>

                      </div>

                      <div>

                        <span>

                          Grad-CAM Fake

                        </span>

                        <strong>

                          {gradcamFakeProbability !==

                          null

                            ? `${gradcamFakeProbability.toFixed(

                                1

                              )}%`

                            : "--"}

                        </strong>

                      </div>

                    </div>

                    <p className="gradcam-note">

                      Highlighted regions

                      indicate image areas

                      that influenced the

                      model's deepfake

                      classification. They

                      should be treated as

                      model explanations,

                      not definitive proof

                      of manipulation.

                    </p>

                  </>

                )}

              {!gradcamLoading &&

                !gradcamImage &&

                status === "FAKE" && (

                  <p className="gradcam-waiting">

                    Waiting for Grad-CAM

                    explanation...

                  </p>

                )}

            </div>

          )}

        </section>

      </main>

      {/* Most recently saved result from this dashboard session. */}
      <section className="current-history" style={{margin:"20px auto",padding:20,maxWidth:1100,border:"1px solid #5a406b",borderRadius:14,background:"#1b102b",color:"white"}}>
        <h2 style={{marginTop:0}}>Current Detection History</h2>
        {currentInterviewRecord ? (
          <div>
            <p><strong>Record:</strong> #{currentInterviewRecord.id} · <strong>Type:</strong> {currentInterviewRecord.record_type === "VIDEO_TEST" ? "Video Test" : "Live Interview"}</p>
            <p><strong>Detection ID:</strong> {currentInterviewRecord.interview_id}</p>
            {currentInterviewRecord.record_type !== "VIDEO_TEST" && (
              <p><strong>Candidate:</strong> {currentInterviewRecord.candidate_name || "Not provided"} ({currentInterviewRecord.candidate_usn || "No USN"})</p>
            )}
            <p><strong>Final prediction:</strong> {currentInterviewRecord.prediction} · <strong>Risk:</strong> {Number(currentInterviewRecord.risk_score).toFixed(2)}% · <strong>Frames:</strong> {currentInterviewRecord.frames_analyzed}</p>
            <p><strong>Saved at:</strong> {new Date(currentInterviewRecord.created_at).toLocaleString()}</p>
            <a href="/history" style={{color:"#d4a9ff"}}>View all saved detections →</a>
          </div>
         ) : cameraStarted ? (
          <p>Current {testVideoMode ? "video test" : "interview"}: {testVideoMode ? videoTestIdRef.current : (interviewId || "Not specified")}. Live result: {status}; risk: {videoRisk.toFixed(2)}%; frames: {frames}/30. This will be saved when you stop after 30 analyzed frames.</p>
        ) : (
          <p>No saved result for the current detection yet. Start a live interview or video test, then click Stop Interview after 30 analyzed frames.</p>
        )}
      </section>

      {/* ================= FOOTER ================= */}

      <footer>

        Real-Time Deepfake Detection API

        for Corporate HR Video Interviews

      </footer>

    </div>

  );

}

export default App;