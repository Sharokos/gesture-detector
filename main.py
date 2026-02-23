
from open_pose_handler import run_openpose
from gesture_manager.gesture_analysis import GestureAnalysis
from data_manager.debugger import export_person_bodyparts_data, export_person_features_data
from data_manager.plotter import plot_person_sliding_windows, plot_body_part_features, plot_normalization_data, plot_person_bodypart_features
from config import DEBUG_DIR, JSON_DIR, OPEN_POSE_DIR, GESTURE_DETECTION, GENERATE_JSONS, RESOLUTION, VIDEO_LOCATION

DEBUG_FLAG = False
DEEP_DEBUG = True
HAND_DEBUG = False

if GENERATE_JSONS:
    run_openpose(
        video_path=VIDEO_LOCATION,
        output_json_dir=JSON_DIR,
        openpose_dir=OPEN_POSE_DIR,
        openpose_args=f"--hand --net_resolution {RESOLUTION}"
    )
if GESTURE_DETECTION:
    # get frame count
    gesture_analysis = GestureAnalysis(
        input_folder=JSON_DIR) 

    gesture_analysis.execute()
    # Process gestures for person 0
    person_id = 0
    gesture_analysis.determine_gestures_for_person(person_id)
    gesture_analysis.export_windows_to_csv(DEBUG_DIR +"/" + "windows.csv")
    if DEBUG_FLAG:
        export_person_bodyparts_data(gesture_analysis, person_id, HAND_DEBUG)
        export_person_features_data(gesture_analysis, person_id, HAND_DEBUG)
        plot_person_sliding_windows(gesture_analysis, person_id)
        plot_normalization_data(gesture_analysis, person_id)
        plot_person_bodypart_features(gesture_analysis, person_id)
        if DEEP_DEBUG:
            plot_body_part_features(gesture_analysis, person_id)

        
