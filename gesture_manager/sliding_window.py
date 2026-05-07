from gesture_manager.features_manager import FeaturesManager
from gesture_manager.score_computer import compute_score
from config import SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STEP, SCORE_THRESHOLD

class SlidingWindow():

    WINDOW_SIZE = SLIDING_WINDOW_SIZE
    STEP_SIZE = SLIDING_WINDOW_STEP

    def __init__(self, sliding_window_id, start_frame, end_frame, person, fps):

        self.id = sliding_window_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.person = person
        self.duration_frames = self.end_frame - self.start_frame + 1
        self.duration_seconds = self.duration_frames / fps
        self.features_manager = FeaturesManager(self)
        self.score = compute_score(self.features_manager)
        self.threshold = 0
        self.distance_threhsold = 0
        self.is_gesture = 0

    def recompute_score(self):
        self.score = compute_score(self.features_manager)
    def contains_gesture(self,
                         score_threshold=SCORE_THRESHOLD):
        # TODO: see if you need some low pass/high pass filters like below
        # if self.max_acceleration < 0.05 and self.max_angular_velocity < 0.028: 
        #     return False
        if self.start_frame <= self.WINDOW_SIZE * 2:
            # don't accept gestures so early, because parameters are still settling in
            return False
        return self.score >= score_threshold
    
    def debug_print(self):
        return_str = "====================\n"
        return_str += f"Window starting at frame: {self.start_frame}\n"
        return_str += f"Score: {self.score}\n"
        return_str += "====================\n"
        return return_str