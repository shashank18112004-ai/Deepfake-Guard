
import { useRef, useState } from "react";
import "./CandidateInterview.css";

function CandidateInterview() {
  const videoRef = useRef(null);
  const remoteVideoRef = useRef(null);
  const streamRef = useRef(null);
  const socketRef = useRef(null);
  const peerRef = useRef(null);
  const pendingCandidatesRef = useRef([]);

  const [interviewId, setInterviewId] = useState("");
  const [cameraStarted, setCameraStarted] = useState(false);
  const [hrVideoReceived, setHrVideoReceived] = useState(false);
  const [connectionStatus, setConnectionStatus] =
    useState("Not Connected");
  const [message, setMessage] =
    useState("Enter Interview ID to continue");

  const createPeerConnection = (stream) => {
    if (peerRef.current) return peerRef.current;

    const peer = new RTCPeerConnection({
      iceServers: [
        { urls: "stun:stun.l.google.com:19302" },
      ],
    });

    peerRef.current = peer;

    stream.getTracks().forEach((track) => {
      peer.addTrack(track, stream);
    });

    peer.onicecandidate = (event) => {
      const socket = socketRef.current;

      if (
        event.candidate &&
        socket?.readyState === WebSocket.OPEN
      ) {
        socket.send(JSON.stringify({
          type: "candidate",
          candidate: event.candidate,
        }));
      }
    };

    // Receive the HR camera stream.
    peer.ontrack = async (event) => {
      const remoteStream =
        event.streams?.[0] ||
        new MediaStream([event.track]);

      if (remoteVideoRef.current) {
        remoteVideoRef.current.srcObject = remoteStream;
        remoteVideoRef.current.muted = false;
        remoteVideoRef.current.playsInline = true;

        try {
          await remoteVideoRef.current.play();
        } catch (error) {
          console.warn("HR video playback:", error);
        }

        setHrVideoReceived(true);
      }
    };

    peer.onconnectionstatechange = () => {
      if (peer.connectionState === "connected") {
        setConnectionStatus("HR Connected");
        setMessage("Live interview connected successfully");
      } else if (
        peer.connectionState === "disconnected" ||
        peer.connectionState === "failed"
      ) {
        setConnectionStatus("HR Disconnected");
        setMessage("Connection with HR was interrupted");
      } else if (peer.connectionState === "closed") {
        setConnectionStatus("Not Connected");
      }
    };

    return peer;
  };

  const createOffer = async () => {
    const peer = peerRef.current;
    const socket = socketRef.current;

    if (
      !peer ||
      !socket ||
      socket.readyState !== WebSocket.OPEN ||
      peer.signalingState !== "stable"
    ) {
      return;
    }

    try {
      const offer = await peer.createOffer();
      await peer.setLocalDescription(offer);

      socket.send(JSON.stringify({
        type: "offer",
        offer: peer.localDescription,
      }));

      setMessage("HR found. Connecting video and audio...");
    } catch (error) {
      console.error("Could not create offer:", error);
      setMessage("Could not establish connection with HR.");
    }
  };

  const handleSignalingMessage = async (event) => {
    try {
      const data = JSON.parse(event.data);
      const peer = peerRef.current;

      if (!peer) return;

      if (
        data.type === "peer-ready" &&
        data.role === "hr"
      ) {
        setConnectionStatus("Connecting to HR");
        await createOffer();
        return;
      }

      if (data.type === "answer" && data.answer) {
        await peer.setRemoteDescription(
          new RTCSessionDescription(data.answer)
        );

        for (const candidate of pendingCandidatesRef.current) {
          try {
            await peer.addIceCandidate(
              new RTCIceCandidate(candidate)
            );
          } catch (error) {
            console.error("Pending ICE error:", error);
          }
        }

        pendingCandidatesRef.current = [];
        return;
      }

      if (data.type === "candidate" && data.candidate) {
        if (peer.remoteDescription) {
          await peer.addIceCandidate(
            new RTCIceCandidate(data.candidate)
          );
        } else {
          pendingCandidatesRef.current.push(data.candidate);
        }
        return;
      }

      if (data.type === "peer-left" && data.role === "hr") {
        setConnectionStatus("Waiting for HR");
        setHrVideoReceived(false);
        if (remoteVideoRef.current) {
          remoteVideoRef.current.srcObject = null;
        }
        setMessage("HR disconnected. Waiting for HR to rejoin...");
      }
    } catch (error) {
      console.error("Signaling message error:", error);
    }
  };

  const connectSignaling = (cleanInterviewId, stream) =>
    new Promise((resolve, reject) => {
      const encodedId = encodeURIComponent(cleanInterviewId);
      const socket = new WebSocket(
        `ws://127.0.0.1:8000/ws/signaling/${encodedId}/candidate`
      );

      socketRef.current = socket;

      socket.onopen = () => {
        createPeerConnection(stream);
        setConnectionStatus("Waiting for HR");
        setMessage("Camera connected. Waiting for HR to join...");
        resolve();
      };

      socket.onmessage = handleSignalingMessage;

      socket.onerror = (error) => {
        setConnectionStatus("Connection Error");
        setMessage("Could not connect to interview server.");
        reject(error);
      };

      socket.onclose = () => {
        if (streamRef.current) {
          setConnectionStatus("Server Disconnected");
        }
      };
    });

  const startInterview = async () => {
    const cleanInterviewId = interviewId.trim();

    if (!cleanInterviewId) {
      setMessage("Please enter the Interview ID.");
      return;
    }

    try {
      setMessage("Requesting camera and microphone...");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: true,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraStarted(true);
      setConnectionStatus("Connecting to Server");

      await connectSignaling(cleanInterviewId, stream);
    } catch (error) {
      console.error("Interview start error:", error);

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      if (videoRef.current) videoRef.current.srcObject = null;

      setCameraStarted(false);
      setConnectionStatus("Not Connected");
      setMessage(
        "Unable to start. Check camera, microphone, and backend server."
      );
    }
  };

  const leaveInterview = () => {
    if (peerRef.current) {
      peerRef.current.close();
      peerRef.current = null;
    }

    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) videoRef.current.srcObject = null;
    if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null;

    pendingCandidatesRef.current = [];

    setCameraStarted(false);
    setHrVideoReceived(false);
    setConnectionStatus("Not Connected");
    setMessage("Interview ended");
  };

  return (
    <div className="candidate-page">
      <div className="candidate-header">
        <div>
          <h1>Deepfake Guard</h1>
          <p>Secure Candidate Video Interview</p>
        </div>

        <div className={`camera-status ${cameraStarted ? "connected" : ""}`}>
          <span></span>
          {connectionStatus}
        </div>
      </div>

      <div className="candidate-card">
        <div className="candidate-title">
          <h2>Candidate Interview</h2>
          <p>Join your secure HR video interview</p>
        </div>

        <div className="interview-input">
          <label htmlFor="candidate-interview-id">Interview ID</label>
          <input
            id="candidate-interview-id"
            type="text"
            placeholder="Example: INT-2026-001"
            value={interviewId}
            onChange={(e) => setInterviewId(e.target.value)}
            disabled={cameraStarted}
          />
        </div>

        <div className="candidate-video">
          <div className="video-tile candidate-self-tile">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={cameraStarted ? "show-video" : "hide-video"}
            />
            {!cameraStarted && (
              <div className="remote-video-placeholder">
                <div className="candidate-camera-icon">◉</div>
                <h3>Camera not started</h3>
                <p>Enter your Interview ID and join the interview.</p>
              </div>
            )}
            <span className="video-label">You</span>
          </div>

          <div className="video-tile hr-remote-tile">
            <video
              ref={remoteVideoRef}
              autoPlay
              playsInline
              className={hrVideoReceived ? "show-video" : "hide-video"}
            />

            {!hrVideoReceived && (
              <div className="remote-video-placeholder">
                <div className="candidate-camera-icon">◉</div>
                <h3>Waiting for HR</h3>
                <p>HR video will appear here when connected.</p>
              </div>
            )}

            <span className="video-label">HR</span>
          </div>

        </div>

        <div className="candidate-message">{message}</div>

        {cameraStarted && (
          <div style={{
            marginTop: "12px",
            textAlign: "center",
            fontSize: "13px",
            color: "#829bb8",
          }}>
            Interview Room: <strong style={{ color: "#fff" }}>
              {interviewId}
            </strong>
          </div>
        )}

        <div className="candidate-buttons">
          <button
            className="join-button"
            onClick={startInterview}
            disabled={cameraStarted}
          >
            Join Interview
          </button>

          <button
            className="leave-button"
            onClick={leaveInterview}
            disabled={!cameraStarted}
          >
            Leave Interview
          </button>
        </div>

        <div className="privacy-info">
          <strong>Secure Interview</strong>
          <p>
            Your video will be analyzed during the interview for
            potential video manipulation.
          </p>
        </div>
      </div>

      <footer className="candidate-footer">
        Real-Time Deepfake Detection for Corporate HR Video Interviews
      </footer>
    </div>
  );
}

export default CandidateInterview;
