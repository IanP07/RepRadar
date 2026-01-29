import { useEffect, useRef, useState } from "react";
import type { Exercise, WorkoutResults, RepResult } from "../types";
import "./CameraView.css";
import { Pose, POSE_CONNECTIONS } from "@mediapipe/pose";
// @ts-ignore - drawing_utils has no TypeScript types
import * as drawingUtils from "@mediapipe/drawing_utils";
import { Cookies } from 'react-cookie'

interface CameraViewProps {
  exercise: Exercise;
  onStop: (results: WorkoutResults) => void;
}

export default function CameraView({ exercise, onStop }: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const poseRef = useRef<Pose | null>(null);
  const poseResultsRef = useRef<any | null>(null);
  const isFlippingRef = useRef(false);
  const poseErroredRef = useRef(false);
  const lastRepCountRef = useRef(0);

  const [repCount, setRepCount] = useState(0);
  const [currentScore, setCurrentScore] = useState(100);
  const [isProcessing, setIsProcessing] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [useDemoMode, setUseDemoMode] = useState(true);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [mirrorVideo, setMirrorVideo] = useState(false);
  const [feedback, updateFeedback] = useState<string | null>(null);
  // Connection debug state
  const [connectionStatus, setConnectionStatus] = useState<string>("Disconnected");
  const [debugWsUrl, setDebugWsUrl] = useState<string | null>(null);
  const [debugClientId, setDebugClientId] = useState<string | null>(null);
  const [lastWsError, setLastWsError] = useState<string | null>(null);
  const cookies = new Cookies();
  const [cookieCamEnabled, setCookieCamEnabled] = useState<boolean>(()=>{
    return cookies.get("cameraEnabled") === true;
  });
  // ---------------------- MEDIAPIPE POSE SETUP ----------------------
  useEffect(() => {
    console.log("POSE EFFECT RUNNING, cameraEnabled =", cameraEnabled);

    if (!cameraEnabled || poseRef.current) return;

    const pose = new Pose({
      locateFile: (file: any) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.onResults((results: any) => {
      poseResultsRef.current = results;
      console.log("Pose results:", results.poseLandmarks?.length, "landmarks");
    });

    poseRef.current = pose;

    return () => {
      if (poseRef.current) {
        poseRef.current.close();
        poseRef.current = null;
      }
      poseResultsRef.current = null;
      isFlippingRef.current = false;
      poseErroredRef.current = false;
    };
  }, [cameraEnabled]);

  // NEW: which camera to use ("user" = front, "environment" = back)
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");

  const repDataRef = useRef<RepResult[]>([]);

  useEffect(() => {
    // If cookie exists and is truthy, auto-enable camera and exit demo mode
    if (cookieCamEnabled) {
      setCameraEnabled(true);
      setUseDemoMode(false);
    }
  }, [cookieCamEnabled]);

  // ---------------------- DEMO MODE ----------------------
  useEffect(() => {
    if (useDemoMode) startDemoMode();
  }, [useDemoMode]);

  // ---------------------- CAMERA INITIALIZATION ----------------------
  useEffect(() => {
    if (!cameraEnabled) return;

    console.log(`yee haw ${exercise}`);

    let mounted = true;

    const initializeCamera = async () => {
      if (!videoRef.current) return;

      try {
        setIsProcessing(true);

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode,
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

        if (!mounted) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        videoRef.current.onloadedmetadata = () => {
          isFlippingRef.current = false;
        };

        setIsProcessing(false);
        setCameraError(null);
        cookies.set("cameraEnabled", true, {path: "/"})
        setCookieCamEnabled(true);
        startCanvasTracking();
      } catch (error: any) {
        console.error("Camera error:", error);
        setCameraError(error.message || "Camera access error");
        setUseDemoMode(true);
        setIsProcessing(false);
      }
    };

    initializeCamera();

    return () => {
      mounted = false;
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream)
          .getTracks()
          .forEach((t) => t.stop());
      }
      isFlippingRef.current = false;
      poseErroredRef.current = false;
    };
  }, [cameraEnabled, facingMode]); // facingMode triggers re-initialization

  // ---------------------- WEBSOCKET SETUP ----------------------
  useEffect(() => {
    if (!cameraEnabled) return;

    // Generate a unique client ID for this session
    const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Resolve WebSocket URL. Support full URL via VITE_SOCKET_URL (ws://... or wss://...),
    // or host via VITE_SOCKET_HOST. Default to the static ngrok URL provided by the user.
    const env = (import.meta as any)?.env || {};
    const DEFAULT_HOST = "https://shameka-unbridgeable-noncausally.ngrok-free.dev";
    const configuredUrl = env.VITE_SOCKET_URL || env.VITE_SOCKET_HOST || DEFAULT_HOST;
    let wsUrl: string;

    if (configuredUrl) {
      const trimmed = configuredUrl.replace(/\/$/, "");
      // If it already contains ws:// or wss://, use as-is
      if (/^wss?:\/\//i.test(trimmed)) {
        wsUrl = `${trimmed}/ws/${clientId}`;
      }
      // If it contains http:// or https://, convert to ws/wss
      else if (/^https?:\/\//i.test(trimmed)) {
        const scheme = trimmed.startsWith("https:") ? "wss:" : "ws:";
        // remove protocol
        const hostOnly = trimmed.replace(/^https?:\/\//i, "");
        wsUrl = `${scheme}//${hostOnly}/ws/${clientId}`;
      }
      // Otherwise treat as host (host[:port])
      else {
        const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
        wsUrl = `${scheme}//${trimmed}/ws/${clientId}`;
      }
    } else {
      const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
      wsUrl = `${scheme}//${window.location.host}/ws/${clientId}`;
    }

    console.log("Using WebSocket URL:", wsUrl);

    // expose debug info to the UI
    try { setDebugWsUrl(wsUrl); setDebugClientId(clientId); } catch(e){}

    // reflect that we're attempting to connect
    setConnectionStatus("Connecting");

    const ws = new WebSocket(wsUrl);
    // expose websocket reference immediately
    websocketRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket connected");
      setConnectionStatus("Connected");
      setLastWsError(null);
      sendExercise();
    };

    ws.onclose = (ev) => {
      console.log("WebSocket disconnected", ev);
      setConnectionStatus("Disconnected");
      websocketRef.current = null;
      try { setLastWsError(`Close code=${(ev as any)?.code} reason=${(ev as any)?.reason}`); } catch(e){}
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setConnectionStatus("Error");
      try { setLastWsError(String(error)); } catch(e){}
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        const { event: eventType, data } = message;

        if (eventType === "update") {
          console.log("Rep count:", data.rep_count);
          console.log("Score:", data.score);
          console.log("Feedback:", data.feedback);
          console.log("Exercise:", data.exercise);

          // Update your UI
          setRepCount(data.rep_count);
          setCurrentScore(data.score);
          updateFeedback(data.feedback);

          if (
            typeof data.rep_count === "number" &&
            data.rep_count > lastRepCountRef.current
          ) {
            lastRepCountRef.current = data.rep_count;

            repDataRef.current.push({
              repNumber: data.rep_count,
              score: typeof data.score === "number" ? data.score : 0,
              notes: data.feedback
                ? Array.isArray(data.feedback)
                  ? data.feedback
                  : [data.feedback]
                : [],
            });
          }
        } else if (eventType === "set_exercise_response") {
          console.log("Exercise set response:", data);
          // server acknowledged our set_exercise — mark connection confirmed
          if (data && data.success) {
            setConnectionStatus("Connected");
            setLastWsError(null);
          } else {
            setConnectionStatus("Error");
            try { setLastWsError(data && data.error ? String(data.error) : "set_exercise failed"); } catch(e){}
          }
        } else if (eventType === "error") {
          console.error("Server error:", data.error);
          setConnectionStatus("Error");
          try { setLastWsError(data && data.error ? String(data.error) : null); } catch(e){}
        }
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
      // Reset connection status on cleanup
      setConnectionStatus("Disconnected");
      websocketRef.current = null;
    };
  }, [cameraEnabled]);

  // ---------------------- SEND FRAMES ----------------------
  const startCanvasTracking = () => {
    if (!canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const sendFPS = 20;
    const interval = 1000 / sendFPS;
    let lastSent = 0;

    const loop = async (timestamp: number) => {
      const video = videoRef.current;

      // If we don't have a video element, just schedule the next frame
      if (!video) {
        requestAnimationFrame(loop);
        return;
      }

      // If we're in the middle of flipping cameras, skip sending frames to MediaPipe
      if (isFlippingRef.current || poseErroredRef.current) {
        requestAnimationFrame(loop);
        return;
      }

      const { videoWidth, videoHeight } = video;

      // Avoid sending frames while the video element has no valid dimensions yet
      if (!videoWidth || !videoHeight) {
        requestAnimationFrame(loop);
        return;
      }

      canvas.width = videoWidth;
      canvas.height = videoHeight;

      ctx.save();

      if (mirrorVideo) {
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
      }

      // Draw raw camera image
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Draw pose wireframe on top if we have landmarks
      const results = poseResultsRef.current;
      if (results && results.poseLandmarks) {
        drawingUtils.drawConnectors(
          ctx,
          results.poseLandmarks,
          POSE_CONNECTIONS,
          { color: "#ffffffff", lineWidth: 3 }
        );
      }

      ctx.restore();

      if (timestamp - lastSent > interval) {
        lastSent = timestamp;

        const base64 = canvas.toDataURL("image/jpeg", 0.7);

        if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
          const message = {
            event: "frame",
            data: { image: base64 }
          };
          websocketRef.current.send(JSON.stringify(message));
        }
      }

      try {
        // Send the frame to your pose solution / landmarker
        await poseRef.current?.send({ image: video });
      } catch (err) {
        console.error("Pose send error, disabling further processing", err);
        // Mark that pose has errored so we stop hammering the WASM graph
        poseErroredRef.current = true;
      }

      requestAnimationFrame(loop);
    };

    requestAnimationFrame(loop);
  };

  // ---------------------- Sends Exercise ----------------------
  const sendExercise = () => {
    if (!websocketRef.current || websocketRef.current.readyState !== WebSocket.OPEN) return;

    const message = {
      event: "set_exercise",
      data: { exercise: exercise }
    };

    websocketRef.current.send(JSON.stringify(message));
    console.log("Sent exercise:", exercise);
  };

  // ---------------------- DEMO MODE ----------------------
  const startDemoMode = () => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (canvas.width === 0) {
      canvas.width = 640;
      canvas.height = 480;
    }

    const draw = () => {
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      requestAnimationFrame(draw);
    };

    draw();
  };

  // ---------------------- STOP ----------------------
  const handleStop = () => {
    const reps = repDataRef.current;
    const avgScore =
      reps.length > 0
        ? Math.round(reps.reduce((sum, r) => sum + r.score, 0) / reps.length)
        : 0;

    // this function is actually handleStopAnalysis in App.tsx
    // it's passed in as a param to this component
    onStop({
      exercise,
      reps,
      overallScore: avgScore,
      overallNotes: [],
    });
  };

  // ---------------------- FLIP CAMERA BUTTON ----------------------
  const flipCamera = () => {
    const video = videoRef.current;

    // Mark that we're in the middle of a flip; the loop will skip frames until the new stream is ready
    isFlippingRef.current = true;
    poseErroredRef.current = false;

    // Stop existing stream tracks so the browser can attach a new one cleanly
    if (video && video.srcObject instanceof MediaStream) {
      video.srcObject.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }

    setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
  };

  // mirrors back camera
  useEffect(() => {
    setMirrorVideo(facingMode === "user");
  }, [facingMode]);

  return (
    <div className="camera-view">
      <div className="camera-container">
        <video
          ref={videoRef}
          className="video-element"
          autoPlay
          playsInline
          muted
          style={{
            transform: mirrorVideo ? "scaleX(-1)" : "scaleX(1)",
          }}
        />
        <canvas ref={canvasRef} className="pose-canvas" />
        
        {/* Enable camera button */}
        {useDemoMode && !cameraError && !isProcessing && !cookieCamEnabled &&(
          <div className="camera-toggle">
            <button
              className="enable-camera-button"
              onClick={() => {
                setCameraEnabled(true);
                setUseDemoMode(false);
              }}
            >
              Enable Camera
            </button>
          </div>
        )}

        {/* Flips Camera */}
        {cameraEnabled && (
          <button
            className="flip-button"
            onClick={flipCamera}
            style={{
              position: "absolute",
              top: 10,
              right: 10,
              padding: "10px 14px",
              borderRadius: 8,
              background: "rgba(0,0,0,0.6)",
              color: "white",
              fontSize: 14,
            }}
          >
            Flip Camera
          </button>
        )}
      </div>

      {/* Stats */}
      {/* Connection Debug Panel */}
      <div className="connection-status" style={{ textAlign: "center", margin: "10px 0" }}>
        <div style={{fontWeight:600}}>Socket: {connectionStatus}</div>
        {debugWsUrl && <div style={{fontSize:12, color:'#666'}}>WS URL: {debugWsUrl}</div>}
        {debugClientId && <div style={{fontSize:12, color:'#666'}}>Client ID: {debugClientId}</div>}
        {lastWsError && <div style={{fontSize:12, color:'crimson'}}>Last Error: {lastWsError}</div>}
      </div>
      <div className="stat-item">
        <span
          className="stat-value"
          style={{ display: "block", textAlign: "center" }}
        >
          {feedback}
        </span>
      </div>

      <div className="stats-panel">
        <div className="stat-item">
          <span className="stat-label">Reps Completed</span>
          <span className="stat-value">{repCount}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Current Score</span>
          <span className="stat-value">{currentScore}</span>
        </div>
      </div>

      <button className="stop-button" onClick={handleStop}>
        Stop Analysis
      </button>
    </div>
  );
}
