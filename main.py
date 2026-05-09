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
from data_convertor import eaf_to_json, json_to_eaf
from utility import get_video_frame_count
eaf_path = r"D:\DATA\U_SAOM\Work\Misc\GD\Output_Json_Mai\EAF\Emotional_LA_Pianist_L2_EM_AV_FS_P31.eaf"
json_output = r"D:\DATA\U_SAOM\Work\Misc\GD\Output_Json_Mai\EAF\Emotional_LA_Pianist_L2_EM_AV_FS_P31.json"
json_path =r"D:\DATA\U_SAOM\Work\Misc\GD\Output_Json_Mai\EAF\Emotional_LA_Pianist_L2_EM_AV_FS_P31.json"
eaf_output = r"D:\DATA\U_SAOM\Work\Misc\GD\Output_Json_Mai\EAF\Emotional_LA_Pianist_L2_EM_AV_FS_P31_test.eaf"

# eaf_to_json(eaf_path, json_path)
# exit()

DEBUG_FLAG = True
DEEP_DEBUG = False
HAND_DEBUG = False

INPUT_ROOT = INPUT_DIR



OUTPUT_ROOT = os.path.join(
    os.path.dirname(INPUT_ROOT),
    "OUTPUT_" + os.path.basename(INPUT_ROOT)
)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

for subdir in os.listdir(INPUT_ROOT):
    input_dir = os.path.join(INPUT_ROOT, subdir)
    video_fps = 30

    if not os.path.isdir(input_dir):
        continue

    

    json_dir = os.path.join(OUTPUT_ROOT, subdir, "json")
    output_path = os.path.join(OUTPUT_ROOT, subdir, "results")

    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    try:
        video_files = [f for f in os.listdir(input_dir) if f.endswith((".mp4", ".avi"))]
        video_path = os.path.join(input_dir, video_files[0])
        video_fps = get_video_frame_count(video_path)
        print(f"Processing video: {video_path}.")
        print(f"Detected FPS: {video_fps}.")
    except IndexError:
        print(f"No video found in {input_dir}")
        continue
    # --- OPENPOSE ---
    if GENERATE_JSONS:
        print(f"Processing (OPENPOSE): {subdir}")
        
        run_openpose(
            video_path=video_path,
            output_json_dir=json_dir,
            openpose_dir=OPEN_POSE_DIR,
            openpose_args=f"--hand --net_resolution {RESOLUTION}"
        )

    # --- GESTURE DETECTION ---
    if GESTURE_DETECTION:
        print(f"Processing (Gesture detection): {subdir}")
        gesture_analysis = GestureAnalysis(input_folder = json_dir,frame_rate = video_fps)
        
        # gesture_analysis.execute_debug()
        gesture_analysis.execute()
        person_id = 0
        gesture_analysis.determine_gestures_for_person(person_id, output_path)

        if DEBUG_FLAG:
            gesture_analysis.export_windows_to_csv(os.path.join(output_path, "windows.csv"))
            export_person_bodyparts_data(gesture_analysis, person_id, HAND_DEBUG, output_path)
            export_person_features_data(gesture_analysis, person_id, DEEP_DEBUG, output_path)

            plot_person_sliding_windows(gesture_analysis, person_id, output_path)
            plot_normalization_data(gesture_analysis, person_id, output_path)
            plot_person_bodypart_features(gesture_analysis, person_id, output_path)

            if DEEP_DEBUG:
                plot_body_part_features(gesture_analysis, person_id, output_path)