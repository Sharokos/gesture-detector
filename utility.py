from pathlib import Path
import json
from person import PersonGesture
import matplotlib.pyplot as plt
import plotly.graph_objs as go

COCO_PARTS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", 
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", 
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", 
    "REye", "LEye", "REar", "LEar", 
    "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]

HAND_PARTS = [
    "Wrist",
    "Thumb_1", "Thumb_2", "Thumb_3", "Thumb_4",
    "Index_1", "Index_2", "Index_3", "Index_4",
    "Middle_1", "Middle_2", "Middle_3", "Middle_4",
    "Ring_1", "Ring_2", "Ring_3", "Ring_4",
    "Pinky_1", "Pinky_2", "Pinky_3", "Pinky_4"
]

def parse_openpose_and_populate_persons(folder_path) -> dict:
    """
    Parses OpenPose JSONs and builds a dictionary of PersonGesture instances.

    Args:
        folder_path (str or Path): Path to folder with OpenPose JSON files.

    Returns:
        Dict[int, PersonGesture]: Mapping from person_id to their tracked gesture data.
    """
    folder_path = Path(folder_path)
    json_files = sorted(folder_path.glob("*.json"))

    persons = {}
    # Each file represents a frame, and each person in the frame has keypoints
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        frame_index = int(json_file.stem.split("_")[1])  # e.g., '000000000001'
        # print(f"Processing frame {frame_index} from {json_file.name}")
        for person_id, person in enumerate(data.get("people", [])):
            # Ensure person object exists
            if person_id not in persons:
                # print(f"Creating new PersonGesture for person_id {person_id}")
                persons[person_id] = PersonGesture(person_id)

            # Extract and organize keypoints
            body_data = person.get("pose_keypoints_2d", [])
            left_hand_data = person.get("hand_left_keypoints_2d", [])
            right_hand_data = person.get("hand_right_keypoints_2d", [])

            keypoint_data = {
                "body": {
                    COCO_PARTS[i]: (
                        body_data[i * 3],
                        body_data[i * 3 + 1],
                        body_data[i * 3 + 2]
                    )
                    for i in range(len(COCO_PARTS))
                },
                "left_hand": {
                    HAND_PARTS[i]: (
                        left_hand_data[i * 3],
                        left_hand_data[i * 3 + 1],
                        left_hand_data[i * 3 + 2]
                    )
                    for i in range(len(HAND_PARTS)) if left_hand_data
                },
                "right_hand": {
                    HAND_PARTS[i]: (
                        right_hand_data[i * 3],
                        right_hand_data[i * 3 + 1],
                        right_hand_data[i * 3 + 2]
                    )
                    for i in range(len(HAND_PARTS)) if right_hand_data
                }
            }

            persons[person_id].add_frame_data(frame_index, keypoint_data)
            # print(persons[person_id])

    return persons


persons = parse_openpose_and_populate_persons("output_jsons")

# print(persons[1].body["RWrist"])
# persons[1].body["RWrist"].display_frames()


# def plot_gesture_part_xy_vs_frame(gesture_part):
#     """
#     Plots x and y coordinates of a GesturePart vs frame number using matplotlib.

#     Args:
#         gesture_part (GesturePart): The gesture part to plot.
#     """
#     filtered_frames = [frame for frame in gesture_part.frames if frame.confidence > 0.1]
#     frame_numbers = [frame.frame_no for frame in filtered_frames]
#     x_coords = [frame.x for frame in filtered_frames]
#     y_coords = [frame.y for frame in filtered_frames]

#     plt.figure(figsize=(10, 5))
#     plt.plot(frame_numbers, x_coords, marker='o', label='X')
#     plt.plot(frame_numbers, y_coords, marker='o', label='Y')
#     plt.title(f"{gesture_part.part_name} coordinates vs Frame")
#     plt.xlabel("Frame Number")
#     plt.ylabel("Coordinate Value")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

def plot_gesture_part_xy_vs_frame(gesture_part):
    """
    Plots x and y coordinates of a GesturePart vs frame number using Plotly.
    Shows timestamp as hover text.
    """
    filtered_frames = [frame for frame in gesture_part.frames if frame.confidence > 0.1]
    frame_numbers = [frame.get_timestamp() for frame in filtered_frames]
    x_coords = [frame.x for frame in filtered_frames]
    y_coords = [frame.y for frame in filtered_frames]
    timestamps = [frame.get_timestamp() for frame in filtered_frames]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame_numbers, y=x_coords, mode='lines+markers', name='X',
        text=[f"Time: {ts}" for ts in timestamps], hoverinfo='text+y+x'
    ))
    fig.add_trace(go.Scatter(
        x=frame_numbers, y=y_coords, mode='lines+markers', name='Y',
        text=[f"Time: {ts}" for ts in timestamps], hoverinfo='text+y+x'
    ))
    fig.update_layout(
        title=f"{gesture_part.part_name} coordinates vs Frame (confidence > 0.7)",
        xaxis_title="Frame Number",
        yaxis_title="Coordinate Value",
        legend_title="Coordinate"
    )
    fig.show()
# After you have a GesturePart, e.g.:
# print(persons[0].right_hand)
rwrist = persons[0].left_hand["L_Thumb_2"]
plot_gesture_part_xy_vs_frame(rwrist)