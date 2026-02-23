from pathlib import Path
import json

from config import COCO_PARTS, HAND_PARTS
from data_model.person import PersonGesture


def parse_openpose_and_populate_persons(gesture_analysis=None):
        """
        Parses OpenPose JSONs and builds a dictionary of PersonGesture instances.

        Returns:
            Dict[int, PersonGesture]: Mapping from person_id to their tracked gesture data.
        """
        persons = {}
        folder_path = Path(gesture_analysis.input_folder)
        json_files = sorted(folder_path.glob("*.json"))
        number_of_frames = 0
        # Each file represents a frame, and each person in the frame has keypoints
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            frame_index = int(json_file.stem.split("_")[1])  # e.g., '000000000001'
            number_of_frames += 1
            for person_id, person_data in enumerate(data.get("people", [])):
                # Ensure person object exists
                if person_id not in persons:
                    persons[person_id] = PersonGesture(person_id, gesture_analysis=gesture_analysis)

                # Extract and organize keypoints
                body_data = person_data.get("pose_keypoints_2d", [])
                left_hand_data = person_data.get("hand_left_keypoints_2d", [])
                right_hand_data = person_data.get("hand_right_keypoints_2d", [])

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

        gesture_analysis.number_of_frames = number_of_frames
        gesture_analysis.persons = persons