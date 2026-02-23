

# REMOVING CONFIG PARAMETERS MIGHT LEAD TO UNEXPECTED BEHAVIOR. DON'T!

JSON_DIR = r"G:\OpenPose\detect_gestures\output_jsons_highres"
DEBUG_DIR = "OutputDebugData"
OPEN_POSE_DIR = r"G:\OpenPose\openpose"
VIDEO_LOCATION = r"G:\OpenPose\detect_gestures\video.mp4"

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

WEIGHTS ={
    "motion_energy_weight": 0.15,
    "mean_velocity_weight":0.15,
    "mean_velocity_variance_weight":0,
    "distal_proximal_weight":0.2,
    "persistence_weight":0.15,
    "directional_weight":0,
    "max_angular_velocity_weight":0.1,
    "hands_energy_weight":0,
    "mean_baseline_distance":0.45,
    "acc_weight":0.15
}

# Number of frames for a sliding window object. Gesture detection happens per sliding window.
# If too small -> not enough information to determine if gesture or not
# If too large -> multiple gestures might get merged, or tiny motions might get lost in averages
SLIDING_WINDOW_SIZE = 18
# Step to advance to the next sliding window. Note that there should always be an overlap in sliding windows.
SLIDING_WINDOW_STEP = 9
# Number of frames for smoothing out coordinates, velocities and accelerations, reducing noise in the data.
SMOOTHING_WINDOW = 9
# Number of frames for which we calculate a "baseline" position for a body part -> average coordinates throughout the duration
# If too small -> won't really be a baseline, but rather a snapshot of the coordinates
# If too large -> false baseline, because of natural shift in posture 
BASELINE_WINDOW = 500
# A sliding window with a score above this threshold will be considered a gesture
SCORE_THRESHOLD = 0.07
# Any adjacent sliding windows that are less than TEMPORAL_GAP number of frames apart will be merged into one single gesture.
TEMPORAL_GAP = 9


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

