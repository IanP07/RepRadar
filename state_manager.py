import datetime


class StateManager:
    def __init__(self):
        self.states = {}

    def init_state(self, sid):
        self.states[sid] = {
            "exercise": "Jumping Jacks",
            "rep_count": 0,
            "rep_stage": None,
            "frame_scores": [],
            "rep_scores": [],
            "rep_events": [],
            "min_knee_angle": 180,
            "min_elbow_angle": 180,
            "max_knee_dist": 0,
            "min_knee_dist": 1,
            # Connection metadata
            "connect_time": datetime.datetime.now(datetime.timezone.utc),
            "last_seen": datetime.datetime.now(datetime.timezone.utc),
            "disconnect_time": None,
        }

    def get_state(self, sid):
        return self.states.get(sid)

    def update_state(self, sid, new_state):
        if sid in self.states:
            self.states[sid].update(new_state)

    def remove_state(self, sid):
        self.states.pop(sid, None)
