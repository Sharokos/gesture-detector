from data_manager import input_parser
from open_pose_handler import run_openpose
from gesture_analysis import GestureAnalysis

# run_openpose(
#     video_path=r"G:\OpenPose\detect_gestures\video.mp4",
#     output_json_dir=r"G:\OpenPose\detect_gestures\output_jsons",
#     openpose_dir=r"G:/OpenPose/openpose/",
#     openpose_args="--hand --net_resolution 176x-1"
# )

gesture_analysis = GestureAnalysis(
    input_folder=r"G:\OpenPose\detect_gestures\output_jsons")

# Parse OpenPose data and build all features

input_parser.parse_openpose_and_populate_persons(gesture_analysis=gesture_analysis)
# we have all data
gesture_analysis.build_all_data()

# Create sliding windows
gesture_analysis.create_sliding_windows()

# Build features for all sliding windows
for window in gesture_analysis.sliding_windows:
    window.build_features()

print(f"Created {len(gesture_analysis.sliding_windows)} sliding windows")
print(f"Detected {len(gesture_analysis.persons)} persons")

# Process gestures for person 0
person_id = 0
person = gesture_analysis.get_person_by_id(person_id)
gesture_analysis.export_person_windows_to_csv(person_id,"testington.csv")
# gesture_analysis.plot_person_sliding_windows(person_id)
# gesture_analysis.export_debug_data_for_person(person_id)
if person:
    gesture_analysis.export_person_bodyparts_to_csv(person)
    # Get all windows for this person
    person_windows = [w for w in gesture_analysis.sliding_windows if w.person.person_id == person_id]
    
    # Detect gestures (windows that contain significant motion)
    gesture_windows = [w for w in person_windows if w.contains_gesture()]
    
    print(f"\nPerson {person_id}: Detected {len(gesture_windows)} gesture windows")
    
    # Merge adjacent gesture windows that are similar
    gesture_groups = gesture_analysis.merge_gesture_windows(
        gesture_windows,
        max_temporal_gap=25,
    )
    
    # Set frame rate if different from default 30 FPS
    gesture_analysis.frame_rate = 24  # Adjust to your video's FPS
    
    # Print gesture summary with timestamps
    gesture_analysis.print_gesture_summary(gesture_groups)
    
    # Export to JSON for video helper
    gesture_analysis.export_gesture_groups_to_json(gesture_groups, "gestures.json")
    
