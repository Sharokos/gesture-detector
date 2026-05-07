

# REMOVING CONFIG PARAMETERS MIGHT LEAD TO UNEXPECTED BEHAVIOR. DON'T!

OPEN_POSE_DIR = r"G:\OpenPose\openpose"
INPUT_DIR = r"D:\DATA\U_SAOM\Work\Misc\GD\gd2\gesture-detector\INPUT_VIDEOS"

# 176x-1 → very fast
# 368x-1 → good balance
# 656x-1 → very accurate, slower
RESOLUTION = "176x-1"

# Whether the script should generate keypoint data using OpenPose
GENERATE_JSONS = False
# Whether the script should analyze keypoint data in order to identify gestures
GESTURE_DETECTION = True
# Example combination of the above and their meaning:
# GENERATE_JSONS = False and GESTURE_DETECTION = True -> kepoint jsons already exist, only detect gestures from the data
# GENERATE_JSONS = True and GESTURE_DETECTION = True -> start from scratch, use OpenPose to generate keypoint jsons and then detect gestuers
# GENERATE_JSONS = True and GESTURE_DETECTION = False -> only generate keypoint jsons using OpenPose for later use

CORRECTION_FACTOR_VAR = 100000
CORRECTION_FACTOR_HANDS = 100
WEIGHTS ={
"motion_energy_weight": 0.15,
"mean_velocity_weight": 0.35,
"mean_velocity_variance_weight": 0,
"distal_proximal_weight": 0.05,
"persistence_weight": 0.0,
"directional_weight": 0.05,
"max_angular_velocity_weight": 0.04,
"hands_energy_weight": 0,
"mean_baseline_distance": 0,
"acc_weight": 0.08,
"saliency_weight": 0.035,
"burst_weight": 0.0,
"changes_weight": 0.025,
"path_efficiency_weight": 0.05,
}

# Number of frames for a sliding window object. Gesture detection happens per sliding window.
# If too small -> not enough information to determine if gesture or not
# If too large -> multiple gestures might get merged, or tiny motions might get lost in averages
SLIDING_WINDOW_SIZE = 18
# Step to advance to the next sliding window. Note that there should always be an overlap in sliding windows.
SLIDING_WINDOW_STEP = 9
# Number of frames for smoothing out coordinates, velocities and accelerations, reducing noise in the data.
SMOOTHING_WINDOW = 9
# Only consider gestures that consists of more windows than this threshold
MIN_WINDOW_THRESHOLD = 3
# Any adjacent sliding windows that are less than TEMPORAL_GAP number of frames apart will be merged into one single gesture.
TEMPORAL_GAP =  SLIDING_WINDOW_STEP +3

# The maximum number of sliding windows for which to consider a "HOLD". If the position is held for more than this number, the windows will not be marked as containing gestures
MAX_NUMBER_OF_HOLD_WINDOWS = 15

# Below variables are no longer used in usual flows
# ============== OBSOLETHE ===================
# Number of frames for which we calculate a "baseline" position for a body part -> average coordinates throughout the duration
# If too small -> won't really be a baseline, but rather a snapshot of the coordinates
# If too large -> false baseline, because of natural shift in posture 
BASELINE_WINDOW = 50
# A sliding window with a score above this threshold will be considered a gesture
SCORE_THRESHOLD = 1
# =============================================

# COCO body parts minimal
COCO_PARTS = [
    "Neck", "RShoulder", "RElbow", "RWrist", 
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip","LHip"
]
HAND_PARTS = [
    "Wrist",
    "Thumb_1", "Thumb_2", "Thumb_3", "Thumb_4",
    "Index_1", "Index_2", "Index_3", "Index_4",
    "Middle_1", "Middle_2", "Middle_3", "Middle_4",
    "Ring_1", "Ring_2", "Ring_3", "Ring_4",
    "Pinky_1", "Pinky_2", "Pinky_3", "Pinky_4"
]

