import os
from open_pose_handler import run_openpose
from gesture_manager.gesture_analysis import GestureAnalysis
from data_manager.debugger import export_person_bodyparts_data, export_person_features_data
from data_manager.plotter import (
    plot_person_sliding_windows,
    plot_body_part_features,
    plot_normalization_data,
    plot_person_bodypart_features
)
from config import OPEN_POSE_DIR, GESTURE_DETECTION, GENERATE_JSONS, RESOLUTION, INPUT_DIR

DEBUG_FLAG = True
DEEP_DEBUG = True
HAND_DEBUG = True

INPUT_ROOT = INPUT_DIR

OUTPUT_ROOT = os.path.join(
    os.path.dirname(INPUT_ROOT),
    "OUTPUT_" + os.path.basename(INPUT_ROOT)
)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

for subdir in os.listdir(INPUT_ROOT):
    input_dir = os.path.join(INPUT_ROOT, subdir)

    if not os.path.isdir(input_dir):
        continue

    

    json_dir = os.path.join(OUTPUT_ROOT, subdir, "json")
    output_path = os.path.join(OUTPUT_ROOT, subdir, "results")

    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # --- OPENPOSE ---
    if GENERATE_JSONS:
        print(f"Processing: {subdir}")
        try:
            video_files = [f for f in os.listdir(input_dir) if f.endswith((".mp4", ".avi"))]
            video_path = os.path.join(input_dir, video_files[0])
        except IndexError:
            print(f"No video found in {input_dir}")
        run_openpose(
            video_path=video_path,
            output_json_dir=json_dir,
            openpose_dir=OPEN_POSE_DIR,
            openpose_args=f"--hand --net_resolution {RESOLUTION}"
        )

    # --- GESTURE DETECTION ---
    if GESTURE_DETECTION:
        gesture_analysis = GestureAnalysis(input_folder=json_dir)
        gesture_analysis.execute()

        person_id = 0
        gesture_analysis.determine_gestures_for_person(person_id, output_path)

        if DEBUG_FLAG:
            gesture_analysis.export_windows_to_csv(os.path.join(output_path, "windows.csv"))
            export_person_bodyparts_data(gesture_analysis, person_id, HAND_DEBUG, output_path)
            export_person_features_data(gesture_analysis, person_id, HAND_DEBUG, output_path)

            plot_person_sliding_windows(gesture_analysis, person_id, output_path)
            plot_normalization_data(gesture_analysis, person_id, output_path)
            plot_person_bodypart_features(gesture_analysis, person_id, output_path)

            if DEEP_DEBUG:
                plot_body_part_features(gesture_analysis, person_id, output_path)