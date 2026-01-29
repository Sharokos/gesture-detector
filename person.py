from body_part import BodyPart
from frame import Frame
from frame_normalization import FrameNormalization
import math
class PersonGesture:
    def __init__(self, person_id,origin="Neck",gesture_analysis=None):
        self.person_id = person_id
        self.gesture_analysis = gesture_analysis
        self.body = {part: BodyPart(part, person_id,self.gesture_analysis) for part in self.gesture_analysis.COCO_PARTS}
        # self.left_hand = {f"L_{part}": BodyPart(f"L_{part}") for part in self.HAND_PARTS}
        # self.right_hand = {f"R_{part}": BodyPart(f"R_{part}") for part in self.HAND_PARTS}
        self.origin_part = origin
        self.normalization_data = {}
    
    def build_all_data(self):
        """
        Build all necessary data for this person gesture.
        """
        self.build_normalization_data()
        self.normalize_all_parts()
        self.build_magnitudes_all_parts()

    def build_normalization_data(self):
        """
        Add additional data to all body parts for a specific frame.
        """
        for frame_idx in self.body[self.origin_part].frames:
            x_origin, y_origin = self.get_origin(frame_idx)
            shoulder_length = self.get_shoulder_length(frame_idx)
            temp_norm = FrameNormalization(frame_idx, x_origin, y_origin, shoulder_length)
            self.normalization_data[frame_idx] = temp_norm
    def normalize_all_parts(self):
        """
        Normalize all body parts across all frames.
        """
        for part in self.body.values():
            part.update_normalized()
        # for part in self.left_hand.values():
        #     part.update_normalized()
        # for part in self.right_hand.values():
        #     part.update_normalized()
    def get_normalization_data(self, frame_idx):
        """
        Retrieve normalization data for a specific frame.

        Args:
            frame_idx (int): Frame number.      
        Returns:

            FrameNormalization object if exists, else None.
        """        
        return self.normalization_data.get(frame_idx, None)
    def build_magnitudes_all_parts(self):
        """
        Build magnitudes for all body parts across all frames.
        """
        for part in self.body.values():
            part.build_velocities_and_accelerations()
        # for part in self.left_hand.values():
        #     part.build_velocities_and_accelerations()
        # for part in self.right_hand.values():
        #     part.build_velocities_and_accelerations()
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

        # for part, (x, y, c) in keypoint_data.get("left_hand", {}).items():
        #     pname = f"L_{part}"
        #     if pname in self.left_hand:
        #         temp_frame = Frame(frame_idx, x, y, c)
        #         self.left_hand[pname].add_keyframe(temp_frame)

        # for part, (x, y, c) in keypoint_data.get("right_hand", {}).items():
        #     pname = f"R_{part}"
        #     if pname in self.right_hand:
        #         temp_frame = Frame(frame_idx, x, y, c)
        #         self.right_hand[pname].add_keyframe(temp_frame)
        
# TODO: this might be obsolete
    def build_velocity_magnitudes(self, part_name=""):
        """
        Build velocity magnitudes for a specific body part across all frames.

        Args:
            part_name (str): Name of the body part. If empty, uses origin.
        Returns:
            List of float: velocity magnitudes between consecutive frames.
        """
        velocities = [0]
        print(len(self.body[part_name].frames))
        for frame_idx in range(1,len(self.body[part_name].frames)):
            velocities.append(self.get_velocity_magnitude(frame_idx,part_name))
        return velocities
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
