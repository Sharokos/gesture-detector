from data_model.body_part import BodyPart
from data_model.frame import Frame
import math
import numpy as np

class PersonGesture:
    def __init__(self, person_id,origin="Neck",gesture_analysis=None):
        self.person_id = person_id
        self.gesture_analysis = gesture_analysis
        self.body = {part: BodyPart(part, person_id,self.gesture_analysis) for part in self.gesture_analysis.COCO_PARTS}
        self.left_hand = {f"L_{part}": BodyPart(f"L_{part}",person_id,self.gesture_analysis) for part in self.gesture_analysis.HAND_PARTS}
        self.right_hand = {f"R_{part}": BodyPart(f"R_{part}", person_id,self.gesture_analysis) for part in self.gesture_analysis.HAND_PARTS}
        self.origin_part = origin

    def build_reference_data(self):
        origins = []
        lengths = []
        for frame_idx in self.body["RShoulder"].frames:
            if self.shoulders_confident(frame_idx):
                lengths.append(self.get_shoulder_length(frame_idx))
            if self.body[self.origin_part].frames[frame_idx].is_valid():
                origins.append(self.get_origin(frame_idx))
                

        self.avg_origin_x, self.avg_origin_y = np.median(origins, axis=0)
        self.avg_shoulder_length = np.median(lengths)
    
    def shoulders_confident(self, frame_idx):
        return self.body["RShoulder"].frames[frame_idx].is_valid() and self.body["LShoulder"].frames[frame_idx].is_valid()
    def build_all_data(self):
        """
        Build all necessary data for this person gesture.
        """
        self.build_reference_data()
        self.normalize_all_parts()
        self.build_magnitudes_all_parts()

    def normalize_all_parts(self):
        """
        Normalize all body parts across all frames.
        """
        for part in self.body.values():
            part.update_normalized()
        for part in self.left_hand.values():
            part.update_normalized()
        for part in self.right_hand.values():
            part.update_normalized()

    def build_magnitudes_all_parts(self):
        """
        Build magnitudes for all body parts across all frames.
        """
        for part in self.body.values():
            part.build_velocities_and_accelerations()
        for part in self.left_hand.values():
            part.build_velocities_and_accelerations()
        for part in self.right_hand.values():
            part.build_velocities_and_accelerations()
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
        

    def get_body_part(self, part_name):
        """
        Retrieve a BodyPart object by name.

        Args:
            part_name (str): Name of the body part.
        Returns:
            BodyPart object or None if not found.
        """
        if part_name in self.body:
            return self.body[part_name]
        elif part_name in self.left_hand:
            return self.left_hand[part_name]
        elif part_name in self.right_hand:
            return self.right_hand[part_name]
        return None
    
    def get_origin(self, frame_idx):
        """
        Get the (x, y) coordinates of the origin body part at a given frame index.

        Args:
            frame_idx (int): Frame number.
        Returns:
            Tuple (x, y) if origin exists, else None.
        """        
        origin_part = self.get_body_part(self.origin_part)
        if origin_part is None:
            return None
        return origin_part.get_coordinates(frame_idx)

# TODO: should these values be normalized? Think about it
    def get_shoulder_length(self,frame_idx):
        """
        Calculate shoulder length at a specific frame index.

        Args:
            frame_idx (int): Frame number.
        Returns:
            float: Distance between left and right shoulder, or 0 if not available.
        """

        left_shoulder = self.body["LShoulder"].get_coordinates(frame_idx)
        right_shoulder = self.body["RShoulder"].get_coordinates(frame_idx)
        if left_shoulder is None or right_shoulder is None:
            return 0
        lx, ly = left_shoulder
        rx, ry = right_shoulder
        return math.hypot(rx - lx, ry - ly)
    
    
    def __repr__(self):
        return (f"PersonGesture(person_id={self.person_id}, "
                f"body_parts={len(self.body)}, "
                f"left_hand_parts={len(self.left_hand)}, "
                f"right_hand_parts={len(self.right_hand)})")
