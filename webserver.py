import base64
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from state_manager import StateManager
from datetime import datetime, timezone, timedelta
import asyncio
import os
from typing import Dict

# ==========================================
# FastAPI Application Setup
# ==========================================
app = FastAPI(title="AI Exercise Form Checker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state_manager = StateManager()

STALE_TIMEOUT_SECONDS = 10
STALE_CHECK_INTERVAL_SECONDS = 5

# ==========================================
# WebSocket Connection Manager
# ==========================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, sid: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[sid] = websocket
        # Debug log with timestamp, user id, remote client (if available), and active count
        try:
            client_info = getattr(websocket, "client", None)
        except Exception:
            client_info = None
        print(f"[CONNECT] {utc_now().isoformat()} - User {sid} connected from {client_info}. Active: {len(self.active_connections)}")

    def disconnect(self, sid: str):
        # Capture client info if possible
        ws = self.active_connections.pop(sid, None)
        client_info = None
        try:
            if ws is not None:
                client_info = getattr(ws, "client", None)
        except Exception:
            client_info = None

        print(f"[DISCONNECT] {utc_now().isoformat()} - User {sid} disconnected from {client_info}. Active: {len(self.active_connections)}")

    async def emit(self, sid: str, event: str, data: dict):
        if sid in self.active_connections:
            try:
                await self.active_connections[sid].send_json({
                    "event": event,
                    "data": data
                })
            except Exception as e:
                print(f"Error emitting to {sid}: {e}")

    async def broadcast(self, event: str, data: dict):
        for sid in self.active_connections:
            await self.emit(sid, event, data)

connection_manager = ConnectionManager()

# ==========================================
# Utility Functions
# ==========================================
def utc_now():
    return datetime.now(timezone.utc)

def record_rep_event(state, exercise, rep_count, score, feedback=None):
    timestamp = utc_now().isoformat()
    state.setdefault("rep_events", []).append({
        "rep": rep_count,
        "score": int(score),
        "exercise": exercise,
        "timestamp": timestamp,
        "feedback": feedback,
    })
    return timestamp

# ==========================================
# Stale Client Monitor
# ==========================================
async def _start_stale_client_monitor():
    while True:
        await asyncio.sleep(STALE_CHECK_INTERVAL_SECONDS)
        now = utc_now()
        stale_sids = []
        for sid, state in list(state_manager.states.items()):
            last_seen = state.get("last_seen")
            if last_seen and (now - last_seen) > timedelta(seconds=STALE_TIMEOUT_SECONDS):
                stale_sids.append((sid, last_seen))
        for sid, last_seen in stale_sids:
            state_manager.update_state(sid, {"disconnect_time": now})
            state_manager.remove_state(sid)
            connection_manager.disconnect(sid)
            print(f"[STALE DISCONNECT] User {sid} cleaned up at {now.isoformat()} (last seen {last_seen.isoformat()}).")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_start_stale_client_monitor())

# ==========================================
# MediaPipe Pose Landmarker Setup
# ==========================================
model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)

LANDMARK_INDICES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28
}
KEY_LANDMARKS_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# ==========================================
# Helper Functions
# ==========================================
def angle_between_points(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_joint_angles(landmarks):
    def side_angles(side):
        shoulder_idx = LANDMARK_INDICES[f"{side}_SHOULDER"]
        elbow_idx = LANDMARK_INDICES[f"{side}_ELBOW"]
        wrist_idx = LANDMARK_INDICES[f"{side}_WRIST"]
        hip_idx = LANDMARK_INDICES[f"{side}_HIP"]
        knee_idx = LANDMARK_INDICES[f"{side}_KNEE"]
        ankle_idx = LANDMARK_INDICES[f"{side}_ANKLE"]

        shoulder = angle_between_points(
            [landmarks[elbow_idx].x, landmarks[elbow_idx].y],
            [landmarks[shoulder_idx].x, landmarks[shoulder_idx].y],
            [landmarks[hip_idx].x, landmarks[hip_idx].y]
        )
        elbow = angle_between_points(
            [landmarks[shoulder_idx].x, landmarks[shoulder_idx].y],
            [landmarks[elbow_idx].x, landmarks[elbow_idx].y],
            [landmarks[wrist_idx].x, landmarks[wrist_idx].y]
        )
        hip = angle_between_points(
            [landmarks[shoulder_idx].x, landmarks[shoulder_idx].y],
            [landmarks[hip_idx].x, landmarks[hip_idx].y],
            [landmarks[knee_idx].x, landmarks[knee_idx].y]
        )
        knee = angle_between_points(
            [landmarks[hip_idx].x, landmarks[hip_idx].y],
            [landmarks[knee_idx].x, landmarks[knee_idx].y],
            [landmarks[ankle_idx].x, landmarks[ankle_idx].y]
        )
        return shoulder, elbow, hip, knee

    shoulder_l, elbow_l, hip_l, knee_l = side_angles("LEFT")
    shoulder_r, elbow_r, hip_r, knee_r = side_angles("RIGHT")

    return {
        "shoulder_l": shoulder_l, "elbow_l": elbow_l, "hip_l": hip_l, "knee_l": knee_l,
        "shoulder_r": shoulder_r, "elbow_r": elbow_r, "hip_r": hip_r, "knee_r": knee_r
    }

def is_pose_visible(landmarks, visibility_threshold=0.5, required_ratio=0.6):
    visible_count = sum([1 for idx in KEY_LANDMARKS_INDICES if landmarks[idx].visibility >= visibility_threshold])
    return (visible_count / len(KEY_LANDMARKS_INDICES)) >= required_ratio

def calculateScore(low: float, high: float, actual: float) -> float:
    if high == low:
        return 10
    fitted_actual = max(low, min(actual, high))
    return 10 + (fitted_actual - low) * (90 / (high - low))

def reverseCalculateScore(low: float, high: float, actual: float) -> float:
    if high == low:
        return 10
    fitted_actual = max(low, min(actual, high))
    return max(100 - (fitted_actual - low) * (90 / (high - low)), 10)

# ==========================================
# WebSocket Handlers
# ==========================================
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await connection_manager.connect(client_id, websocket)

    now = utc_now()
    state_manager.init_state(client_id)
    state_manager.update_state(client_id, {
        "connect_time": now,
        "last_seen": now,
        "disconnect_time": None,
        "exercise": None,
        "rep_count": 0,
        "rep_stage": None,
        "frame_scores": [],
        "rep_scores": [],
        "rep_events": [],
        "min_knee_angle": 180,
        "min_elbow_angle": 180,
        "max_knee_dist": 0,
        "min_knee_dist": 1,
    })

    # ConnectionManager already logs detailed connect info; avoid duplicate prints here

    try:
        while True:
            data = await websocket.receive_json()
            event = data.get("event")
            state_manager.update_state(client_id, {"last_seen": utc_now()})

            if event == "frame":
                await handle_frame(client_id, data.get("data", {}))
            elif event == "set_exercise":
                await handle_set_exercise(client_id, data.get("data", {}))
            else:
                print(f"[WS] Unknown event from {client_id}: {event}")

    except WebSocketDisconnect:
        now = utc_now()
        state_manager.update_state(client_id, {"disconnect_time": now})
        state_manager.remove_state(client_id)
        connection_manager.disconnect(client_id)
        # ConnectionManager.disconnect will log the disconnect details

    except Exception as e:
        print(f"[WS ERROR] {client_id}: {e}")
        await connection_manager.emit(client_id, "error", {"error": str(e)})
        state_manager.remove_state(client_id)
        connection_manager.disconnect(client_id)

async def handle_frame(client_id: str, data: dict):
    """Process incoming frame data from client"""
    state = state_manager.get_state(client_id)
    if not state:
        print(f"[FRAME DEBUG] No state found for client_id={client_id}")
        return

    img_data = data.get("image")
    if not img_data:
        print(f"[FRAME DEBUG] No image data received for client_id={client_id}")
        await connection_manager.emit(client_id, "error", {"error": "No image data"})
        return

    if img_data.startswith("data:image"):
        img_data = img_data.split(",")[1]

    try:
        np_arr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[FRAME DEBUG] Could not decode frame for client_id={client_id}")
            await connection_manager.emit(client_id, "error", {"error": "Could not decode frame"})
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        pose_landmarker_result = pose_landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if not pose_landmarker_result.pose_landmarks or len(pose_landmarker_result.pose_landmarks) == 0:
            print(f"[FRAME DEBUG] No pose landmarks detected for client_id={client_id}")
            return

        landmarks = pose_landmarker_result.pose_landmarks[0]

        if not is_pose_visible(landmarks):
            print(f"[FRAME DEBUG] Pose not visible enough for client_id={client_id}")
            return

        angles = extract_joint_angles(landmarks)

        # Process exercise-specific logic
        exercise = state["exercise"]

        if exercise == "Squats":
            await handle_squats(client_id, state, angles)
        elif exercise == "Push-ups":
            await handle_pushups(client_id, state, angles)
        elif exercise == "Jumping Jacks":
            await handle_jumping_jacks(client_id, state, landmarks)
        elif exercise == "Lunges":
            await handle_lunges(client_id, state, landmarks)

    except Exception as e:
        print(f"[FRAME ERROR] {client_id}: {e}")
        await connection_manager.emit(client_id, "error", {"error": str(e)})


async def handle_squats(client_id: str, state: dict, angles: dict):
    """Process squats exercise"""
    knee = min(angles["knee_l"], angles["knee_r"])

    if state["rep_stage"] is None:
        state["rep_stage"] = "up"
        state["min_knee_angle"] = knee
        return
    if state["rep_stage"] != "down" and knee <= 110:
        state["rep_stage"] = "down"
        state["min_knee_angle"] = knee
    elif state["rep_stage"] == "down" and knee <= 110:
        state["min_knee_angle"] = min(state["min_knee_angle"], knee)
    elif state["rep_stage"] == "down" and knee > 110:
        state["rep_stage"] = "up"
        state["rep_count"] += 1

        score = reverseCalculateScore(low=30, high=50, actual=state["min_knee_angle"])

        if state["min_knee_angle"] <= 30:
            feedback = "Great squat"
        elif state["min_knee_angle"] <= 40:
            feedback = "Good squat, bend slightly more"
        elif state["min_knee_angle"] <= 50:
            feedback = "Ok squat, focus on form"
        else:
            feedback = "Bad squat, go lower"

        state["rep_scores"].append(score)
        timestamp = record_rep_event(state, "Squats", state["rep_count"], score, feedback)

        print(f"[REP COUNTED] Squats - client_id={client_id}, rep={state['rep_count']}, score={int(score)}, feedback={feedback}, timestamp={timestamp}")

        await connection_manager.emit(client_id, "update", {
            "exercise": "Squats",
            "rep_count": state["rep_count"],
            "score": int(score),
            "feedback": feedback,
            "timestamp": timestamp,
        })

    state_manager.update_state(client_id, state)


async def handle_pushups(client_id: str, state: dict, angles: dict):
    """Process push-ups exercise"""
    elbow_angle = min(angles["elbow_l"], angles["elbow_r"])
    if state["rep_stage"] is None:
        state["rep_stage"] = "up"
        state["min_elbow_angle"] = elbow_angle
        return

    if state["rep_stage"] != "down" and elbow_angle <= 90:
        state["rep_stage"] = "down"
        state["min_elbow_angle"] = elbow_angle
    elif state["rep_stage"] == "down" and elbow_angle <= 100:
        state["min_elbow_angle"] = min(state["min_elbow_angle"], elbow_angle)
    elif state["rep_stage"] == "down" and elbow_angle >= 150:
        state["rep_stage"] = "up"
        state["rep_count"] += 1

        if state["min_elbow_angle"] <= 70:
            feedback = "Good push-up"
            score = 100
        elif state["min_elbow_angle"] <= 80:
            feedback = "Decent push-up"
            score = 80
        elif state["min_elbow_angle"] <= 100:
            feedback = "Mediocre push-up"
            score = 60
        else:
            feedback = "Poor push-up"
            score = 50

        state["rep_scores"].append(score)
        timestamp = record_rep_event(state, "Push-ups", state["rep_count"], score, feedback)

        print(f"[REP COUNTED] Push-ups - client_id={client_id}, rep={state['rep_count']}, score={int(score)}, feedback={feedback}, timestamp={timestamp}")

        await connection_manager.emit(client_id, "update", {
            "exercise": "Push-ups",
            "rep_count": state["rep_count"],
            "score": int(score),
            "feedback": feedback,
            "timestamp": timestamp,
        })

    state_manager.update_state(client_id, state)


async def handle_jumping_jacks(client_id: str, state: dict, landmarks):
    """Process jumping jacks exercise"""
    hand_height = min(
        landmarks[LANDMARK_INDICES['LEFT_WRIST']].y,
        landmarks[LANDMARK_INDICES['RIGHT_WRIST']].y
    )
    knee_dist = abs(
        landmarks[LANDMARK_INDICES['LEFT_KNEE']].x -
        landmarks[LANDMARK_INDICES['RIGHT_KNEE']].x
    )

    UP_THRESHOLD = 0.38
    DOWN_THRESHOLD = 0.45

    state["max_knee_dist"] = max(state["max_knee_dist"], knee_dist)
    state["min_knee_dist"] = min(state["min_knee_dist"], knee_dist)

    if state["rep_stage"] is None:
        state["rep_stage"] = "down"
        state["max_knee_dist"] = knee_dist
        state["min_knee_dist"] = knee_dist
        return

    if hand_height < UP_THRESHOLD and state["rep_stage"] != "up":
        state["rep_stage"] = "up"
    elif hand_height > DOWN_THRESHOLD and state["rep_stage"] == "up":
        state["rep_stage"] = "down"
        state["rep_count"] += 1

        score = calculateScore(0.12, 0.3, state["max_knee_dist"])

        if state["max_knee_dist"] >= 0.30:
            feedback = "Good form"
        elif state["max_knee_dist"] >= 0.18:
            feedback = "Decent form"
        elif state["max_knee_dist"] >= 0.12:
            feedback = "Shallow form"
        else:
            feedback = "Very shallow form"

        state["rep_scores"].append(score)
        timestamp = record_rep_event(state, "Jumping Jacks", state["rep_count"], score, feedback)

        print(f"[REP COUNTED] Jumping Jacks - client_id={client_id}, rep={state['rep_count']}, score={int(score)}, feedback={feedback}, timestamp={timestamp}")

        state["max_knee_dist"] = 0
        state["min_knee_dist"] = 1

        await connection_manager.emit(client_id, "update", {
            "exercise": "Jumping Jacks",
            "rep_count": state["rep_count"],
            "score": int(score),
            "feedback": feedback,
            "timestamp": timestamp,
        })

    state_manager.update_state(client_id, state)


async def handle_lunges(client_id: str, state: dict, landmarks):
    """Process lunges exercise"""
    def knee_angle(hip, knee, ankle):
        a = np.array([hip.x, hip.y])
        b = np.array([knee.x, knee.y])
        c = np.array([ankle.x, ankle.y])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    left_knee_angle = knee_angle(landmarks[LANDMARK_INDICES['LEFT_HIP']],
                                 landmarks[LANDMARK_INDICES['LEFT_KNEE']],
                                 landmarks[LANDMARK_INDICES['LEFT_ANKLE']])
    right_knee_angle = knee_angle(landmarks[LANDMARK_INDICES['RIGHT_HIP']],
                                  landmarks[LANDMARK_INDICES['RIGHT_KNEE']],
                                  landmarks[LANDMARK_INDICES['RIGHT_ANKLE']])
    knee_used = min(left_knee_angle, right_knee_angle)

    if state["rep_stage"] is None:
        state["rep_stage"] = "up"
        state["min_knee_angle"] = knee_used
        return
    if state["rep_stage"] != "down" and knee_used < 120:
        state["rep_stage"] = "down"
        state["min_knee_angle"] = knee_used
    elif state["rep_stage"] == "down" and knee_used < 120:
        state["min_knee_angle"] = min(state["min_knee_angle"], knee_used)
    elif state["rep_stage"] == "down" and knee_used >= 120:
        state["rep_stage"] = "up"
        state["rep_count"] += 1

        score = reverseCalculateScore(low=60, high=80, actual=state["min_knee_angle"])
        if state["min_knee_angle"] <= 60:
            feedback = "Good lunge"
        elif state["min_knee_angle"] <= 70:
            feedback = "Decent lunge"
        elif state["min_knee_angle"] <= 80:
            feedback = "Shallow lunge"
        else:
            feedback = "Very shallow lunge"

        state["rep_scores"].append(score)
        timestamp = record_rep_event(state, "Lunges", state["rep_count"], score, feedback)

        print(f"[REP COUNTED] Lunges - client_id={client_id}, rep={state['rep_count']}, score={int(score)}, feedback={feedback}, timestamp={timestamp}")

        await connection_manager.emit(client_id, "update", {
            "exercise": "Lunges",
            "rep_count": state["rep_count"],
            "score": int(score),
            "feedback": feedback,
            "timestamp": timestamp,
        })

    state_manager.update_state(client_id, state)


async def handle_set_exercise(client_id: str, data: dict):
    """Handle exercise selection from client"""
    state = state_manager.get_state(client_id)
    if not state:
        print(f"[SET_EXERCISE DEBUG] No state found for client_id={client_id}")
        return

    exercise = data.get("exercise")
    print(f"[SET_EXERCISE DEBUG] User {client_id} setting exercise to '{exercise}'")

    if exercise not in ["Push-ups", "Jumping Jacks", "Squats", "Lunges"]:
        print(f"[SET_EXERCISE DEBUG] Invalid exercise '{exercise}' for client_id={client_id}")
        await connection_manager.emit(client_id, "set_exercise_response", {
            "success": False,
            "error": "Invalid exercise"
        })
        return

    state.update({
        "exercise": exercise,
        "rep_count": 0,
        "rep_stage": None,
        "frame_scores": [],
        "rep_scores": [],
        "rep_events": [],
        "min_knee_angle": 180,
        "min_elbow_angle": 180,
        "max_knee_dist": 0,
        "min_knee_dist": 1,
    })

    state_manager.update_state(client_id, state)
    print(f"[SET_EXERCISE DEBUG] Exercise set successfully for client_id={client_id}: {exercise}")

    await connection_manager.emit(client_id, "set_exercise_response", {
        "success": True,
        "exercise": exercise
    })

# ==========================================
# HTTP Endpoints
# ==========================================
@app.get("/")
async def root():
    return {"message": "AI Exercise Form Checker API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "clients_connected": len(connection_manager.active_connections)}

# ==========================================
# Run server
# ==========================================
if __name__ == "__main__":
    print("Starting AI Exercise Form Checker on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

