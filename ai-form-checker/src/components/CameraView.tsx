import { useEffect, useRef, useState } from "react";
import type { Exercise, WorkoutResults, RepResult } from "../types";
import "./CameraView.css";
import { io, Socket } from "socket.io-client";
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
  const socketRef = useRef<Socket | null>(null);
  const socketConnectedRef = useRef(false);
  const exerciseRef = useRef(exercise);
  const poseRef = useRef<Pose | null>(null);
  const poseResultsRef = useRef<any | null>(null);
  const isFlippingRef = useRef(false);
  const poseErroredRef = useRef(false);
  const lastRepCountRef = useRef(0);

  // Keep exerciseRef in sync with exercise prop
  useEffect(() => {
    exerciseRef.current = exercise;
  }, [exercise]);

  const [repCount, setRepCount] = useState(0);
  const [currentScore, setCurrentScore] = useState(100);
  const [isProcessing, setIsProcessing] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [useDemoMode, setUseDemoMode] = useState(true);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [mirrorVideo, setMirrorVideo] = useState(false);
  const [feedback, updateFeedback] = useState<string | null>(null);
  const [socketConnected, setSocketConnected] = useState(false);
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

  // ---------------------- SOCKET.IO SETUP ----------------------
  useEffect(() => {
    if (!cameraEnabled) return;

    // Static ngrok domain provided by user
    const baseUrl = "https://shameka-unbridgeable-noncausally.ngrok-free.dev";

    console.log("[Socket] Attempting connection to:", baseUrl);

    const socket = io(baseUrl, {
  transports: ["websocket"],
  upgrade: false,
  reconnection: true,
  reconnectionAttempts: Infinity,
  reconnectionDelay: 1000,
});


    socketRef.current = socket;

    socket.on("connect", () => {
      console.log("[Socket] Connected");
      socketConnectedRef.current = true;
      setSocketConnected(true);
      sendExercise();
    });

    socket.on("connected", (_data) => {
      console.log("[Socket] Server confirmed connection");
    });

    socket.io.engine.on("upgrade", (_transport) => {
      // Upgraded to websocket
    });

    socket.io.engine.on("upgradeError", (_error) => {
      // Upgrade error - will continue with polling
    });

    socket.io.engine.on("open", () => {
      // Engine IO connection opened
    });

    socket.io.engine.on("close", (_reason) => {
      // Engine IO connection closed
    });

    socket.on("disconnect", (reason) => {
      console.log("[Socket] Disconnected:", reason);
      socketConnectedRef.current = false;
      setSocketConnected(false);
    });

    socket.on("connect_error", (error) => {
      console.error("[Socket] Connection error:", error.message);
      socketConnectedRef.current = false;
      setSocketConnected(false);
    });

    socket.on("error", (error) => {
      console.error("[Socket] Error:", error);
      socketConnectedRef.current = false;
      setSocketConnected(false);
    });

    socket.on("reconnect_attempt", (_attempt) => {
      // Reconnection attempt
    });

    socket.on("reconnect_failed", () => {
      console.error("[Socket] Reconnection failed");
      socketConnectedRef.current = false;
      setSocketConnected(false);
    });

    socket.on("analysis", (data: any) => {
      if (data.repCount !== undefined) setRepCount(data.repCount);
      if (data.score !== undefined) setCurrentScore(data.score);
    });

    // Getting data from backend
    socket.on("update", (data) => {
      console.log("[Socket] Received update:", data);
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
          // assume backend sends 0–1 or 0–100; normalize if needed
          score: typeof data.score === "number" ? data.score : 0,
          notes: data.feedback
            ? Array.isArray(data.feedback)
              ? data.feedback
              : [data.feedback]
            : [],
        });
      }
    });

    return () => {
      socketConnectedRef.current = false;
      setSocketConnected(false);
      socket.disconnect();
      socketRef.current = null;
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

        if (socketRef.current && socketConnectedRef.current) {
          socketRef.current.emit("frame", { image: base64 });
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
    if (!socketRef.current) return;

    // Use ref to get current exercise value
    const data = { exercise: exerciseRef.current };

    // Emit to server
    socketRef.current.emit("set_exercise", data);
    console.log("Sent exercise:", exerciseRef.current);
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
        <div className="stat-item">
          <span className="stat-label">Socket Status</span>
          <span
            className="stat-value"
            style={{
              color: socketConnected ? "#22c55e" : "#ef4444",
              fontSize: "12px"
            }}
          >
            {socketConnected ? "✓ Connected" : "✗ Disconnected"}
          </span>
        </div>
      </div>

      <button className="stop-button" onClick={handleStop}>
        Stop Analysis
      </button>
    </div>
  );
}
