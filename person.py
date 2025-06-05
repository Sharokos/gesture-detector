from gesture_part import GesturePart
from frame import Frame
class PersonGesture:
    # COCO body parts
    BODY_PARTS = [
        "Nose", "Neck", "RShoulder", "RElbow", "RWrist", 
        "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", 
        "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", 
        "REye", "LEye", "REar", "LEar", 
        "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
    ]

    # Hand parts (21 points)
    HAND_PARTS = [
        "Wrist",
        "Thumb_1", "Thumb_2", "Thumb_3", "Thumb_4",
        "Index_1", "Index_2", "Index_3", "Index_4",
        "Middle_1", "Middle_2", "Middle_3", "Middle_4",
        "Ring_1", "Ring_2", "Ring_3", "Ring_4",
        "Pinky_1", "Pinky_2", "Pinky_3", "Pinky_4"
    ]

    def __init__(self, person_id):
        self.person_id = person_id
        self.body = {part: GesturePart(part) for part in self.BODY_PARTS}
        self.left_hand = {f"L_{part}": GesturePart(f"L_{part}") for part in self.HAND_PARTS}
        self.right_hand = {f"R_{part}": GesturePart(f"R_{part}") for part in self.HAND_PARTS}

    def add_frame_data(self, frame_idx, keypoint_data):
        """
        Add keypoints for a frame from parsed OpenPose JSON data.

        Args:
            frame_idx (int): Frame number.
            keypoint_data (dict): Should contain 'body', 'left_hand', 'right_hand' dicts.
        """
        for part, (x, y, c) in keypoint_data.get("body", {}).items():
            if part in self.body:
                temp_frame = Frame(frame_idx, x, y, c)
                self.body[part].add_keyframe(temp_frame)

        for part, (x, y, c) in keypoint_data.get("left_hand", {}).items():
            pname = f"L_{part}"
            if pname in self.left_hand:
                temp_frame = Frame(frame_idx, x, y, c)
                self.left_hand[pname].add_keyframe(temp_frame)

        for part, (x, y, c) in keypoint_data.get("right_hand", {}).items():
            pname = f"R_{part}"
            if pname in self.right_hand:
                temp_frame = Frame(frame_idx, x, y, c)
                self.right_hand[pname].add_keyframe(temp_frame)

    def __repr__(self):
        return (f"PersonGesture(person_id={self.person_id}, "
                f"body_parts={len(self.body)}, "
                f"left_hand_parts={len(self.left_hand)}, "
                f"right_hand_parts={len(self.right_hand)})")
