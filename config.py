# REMOVING CONFIG PARAMETERS MIGHT LEAD TO UNEXPECTED BEHAVIOR. DON'T!
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()


def _env_bool(name, default):
    return os.getenv(name, str(default)).strip().lower() == "true"


OPEN_POSE_DIR = os.getenv("OPEN_POSE_DIR")
INPUT_DIR = os.getenv("INPUT_DIR")

# 176x-1 → very fast
# 368x-1 → good balance
# 656x-1 → very accurate, slower
RESOLUTION = os.getenv("RESOLUTION", "176x-1")

# Booleans
GENERATE_JSONS = _env_bool("GENERATE_JSONS", False)
GESTURE_DETECTION = _env_bool("GESTURE_DETECTION", True)

# Example combinations:
# GENERATE_JSONS = False and GESTURE_DETECTION = True
# -> keypoint jsons already exist, only detect gestures from the data
#
# GENERATE_JSONS = True and GESTURE_DETECTION = True
# -> generate jsons and then detect gestures
#
# GENERATE_JSONS = True and GESTURE_DETECTION = False
# -> only generate keypoint jsons

# ---------------- Keypoint extraction ----------------
# "rtmpose": RTMPose via rtmlib (no OpenPose install needed, faster, more accurate)
# "openpose": OpenPoseDemo.exe from OPEN_POSE_DIR
# Each backend writes to its own folder (json_rtmpose / json) so results never get mixed.
POSE_BACKEND = os.getenv("POSE_BACKEND", "rtmpose").strip().lower()
# RTMPose model size: "lightweight" (~40 fps on CPU), "balanced" (~17 fps), "performance" (slowest)
RTMPOSE_MODE = os.getenv("RTMPOSE_MODE", "balanced")
# "cpu", or "cuda" with onnxruntime-gpu installed instead of onnxruntime
RTMPOSE_DEVICE = os.getenv("RTMPOSE_DEVICE", "cpu")
# Run the person detector every N frames (the subject is tracked in between)
RTMPOSE_DET_EVERY = int(os.getenv("RTMPOSE_DET_EVERY", "15"))
# Hand keypoints are NOT used by the gesture score (only the body wrists are), and hand models are
# the most expensive part. Leave this off unless you need hand data.
POSE_HANDS = _env_bool("POSE_HANDS", False)
# Rendering + encoding a skeleton video costs a lot of time. Only enable for visual checks.
WRITE_POSE_VIDEO = _env_bool("WRITE_POSE_VIDEO", False)
# Don't re-extract keypoints for a video whose json folder is already filled.
SKIP_EXISTING_JSONS = _env_bool("SKIP_EXISTING_JSONS", True)

# ---------------- Frame-rate handling ----------------
# All values below that are expressed in FRAMES are defined at this reference frame rate and are
# rescaled automatically to the real FPS of each video, so a 25 fps and a 30 fps video behave the same.
REFERENCE_FPS = 30

CORRECTION_FACTOR_VAR = 100000
CORRECTION_FACTOR_HANDS = 100
WEIGHTS ={
"motion_energy_weight": 0.3,
"mean_velocity_weight": 0.25,
"mean_velocity_variance_weight": 0,
"distal_proximal_weight": 0.0,
"persistence_weight": 0.0,
"directional_weight": 0.00,
"max_angular_velocity_weight": 0.28,
"hands_energy_weight": 0.0,
"mean_baseline_distance": 0,
"acc_weight": 0.08,
"saliency_weight": 0.05,
"burst_weight": 0.05,
"changes_weight": 0.0,
"path_efficiency_weight": 0.0,
}

# Number of frames for a sliding window object. Gesture detection happens per sliding window.
# If too small -> not enough information to determine if gesture or not
# If too large -> multiple gestures might get merged, or tiny motions might get lost in averages
SLIDING_WINDOW_SIZE = 18
# Step to advance to the next sliding window. Note that there should always be an overlap in sliding windows.
SLIDING_WINDOW_STEP = 9
# Number of frames for smoothing out coordinates, velocities and accelerations, reducing noise in the data.
SMOOTHING_WINDOW = 4
# Only consider gestures that consists of more windows than this threshold
MIN_WINDOW_THRESHOLD = 3
# Any adjacent sliding windows that are less than TEMPORAL_GAP number of frames apart will be merged into one single gesture.
TEMPORAL_GAP =  SLIDING_WINDOW_STEP + 6
# The maximum number of sliding windows for which to consider a "HOLD". If the position is held for more than this number, the windows will not be marked as containing gestures
MAX_NUMBER_OF_HOLD_WINDOWS = 12

# ---------------- Keypoint cleaning ----------------
# Keypoints below this confidence are treated as missing (OpenPose writes 0,0,0 for undetected points;
# those zeros used to be averaged into the smoothing and created huge fake velocity spikes).
KEYPOINT_CONFIDENCE_THRESHOLD = 0.4
# Gaps of missing keypoints up to this many frames are linearly interpolated (motion blur during fast
# gestures often drops wrist confidence for a few frames).
MAX_INTERPOLATION_GAP = 6
# Body part used as the origin for normalization. The neck is visible for a seated person even when
# a table hides the hips.
ORIGIN_PART = "Neck"

# ---------------- Thresholding ----------------
# "otsu": per-video automatic split between "rest" and "moving" windows (does not force a fixed
#         fraction of the video to be marked as gesture).
# "percentile": old behaviour, always marks roughly the top (100 - SCORE_PERCENTILE)% of windows.
THRESHOLD_METHOD = os.getenv("THRESHOLD_METHOD", "otsu")
SCORE_PERCENTILE = 60
DISTANCE_PERCENTILE = 70
# Factor applied on top of the computed score threshold (<1 = more sensitive)
SCORE_THRESHOLD_FACTOR = 1.0

# BASELINE
ALPHA_CONFIG = 0.995
MAX_UPDATE_DIST_CONFIG = 0.5


# Below variables are no longer used in usual flows
# ============== OBSOLETHE ===================
# Number of frames for which we calculate a "baseline" position for a body part -> average coordinates throughout the duration
# If too small -> won't really be a baseline, but rather a snapshot of the coordinates
# If too large -> false baseline, because of natural shift in posture
BASELINE_WINDOW = 50
# A sliding window with a score above this threshold will be considered a gesture
SCORE_THRESHOLD = 1
# =============================================

# Body parts used by the analysis. Their position in the OpenPose output is looked up by NAME in
# BODY_25_INDEX / COCO_18_INDEX (do not rely on list order!).
COCO_PARTS = [
    "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip","LHip"
]
# OpenPose BODY_25 keypoint indices (default OpenPose model)
BODY_25_INDEX = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
}
# OpenPose COCO (18 keypoints) indices; MidHip does not exist there and is derived from the hips.
COCO_18_INDEX = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "LHip": 11,
}
HAND_PARTS = [
    "Wrist",
    "Thumb_1", "Thumb_2", "Thumb_3", "Thumb_4",
    "Index_1", "Index_2", "Index_3", "Index_4",
    "Middle_1", "Middle_2", "Middle_3", "Middle_4",
    "Ring_1", "Ring_2", "Ring_3", "Ring_4",
    "Pinky_1", "Pinky_2", "Pinky_3", "Pinky_4"
]


def frames_at_fps(n_reference_frames, fps):
    """Convert a frame count defined at REFERENCE_FPS to the equivalent count at `fps`."""
    return max(1, round(n_reference_frames * fps / REFERENCE_FPS))
