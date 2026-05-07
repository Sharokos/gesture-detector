from data_model.body_part import BodyPart
from data_model.frame import Frame
from data_model.frame_normalization import FrameNormalization
import math
import numpy as np
from config import SMOOTHING_WINDOW
class PersonGesture:
    def __init__(self, person_id,origin="LHip",gesture_analysis=None):
        self.person_id = person_id
        self.gesture_analysis = gesture_analysis
        self.body = {part: BodyPart(part, person_id,self.gesture_analysis) for part in self.gesture_analysis.COCO_PARTS}
        self.left_hand = {f"L_{part}": BodyPart(f"L_{part}",person_id,self.gesture_analysis) for part in self.gesture_analysis.HAND_PARTS}
        self.right_hand = {f"R_{part}": BodyPart(f"R_{part}", person_id,self.gesture_analysis) for part in self.gesture_analysis.HAND_PARTS}
        self.origin_part = origin

    # def build_reference_data(self):
    #     # Compute average origin and shoulder length across frames
    #     origins = []
    #     lengths = []
    #     for frame_idx in self.body["RShoulder"].frames:
    #         if self.shoulders_confident(frame_idx):
    #             lengths.append(self.get_shoulder_length(frame_idx))
    #         if self.body[self.origin_part].frames[frame_idx].is_valid():
    #             origins.append(self.get_origin(frame_idx))
                
    #     self.avg_origin_x, self.avg_origin_y = np.median(origins, axis=0)
    #     self.avg_shoulder_length = np.median(lengths)
    def build_average_shoulder_length(self):
        # Compute average origin and shoulder length across frames
        lengths = []
        for frame_idx in self.body["RShoulder"].frames:
            if self.shoulders_confident(frame_idx):
                lengths.append(self.get_shoulder_length(frame_idx))
        self.avg_shoulder_length = np.median(lengths)

    def build_reference_data(self):
        """
        Builds per-frame normalization reference data.
        For frames without confident detection, reuses last valid reference.
        """
        self.build_average_shoulder_length()
        self.normalization_data = {}

        last_x_origin = None
        last_y_origin = None
        last_shoulder_length = self.avg_shoulder_length

        for frame_idx in sorted(self.body[self.origin_part].frames.keys()):
            frame = self.body[self.origin_part].frames[frame_idx]

            # Update origin if confident
            if frame.is_valid():
                x_origin, y_origin = self.get_origin(frame_idx)
                last_x_origin = x_origin
                last_y_origin = y_origin

            # Update scale if confident
            if self.shoulders_confident(frame_idx):
                shoulder_length = self.avg_shoulder_length
                last_shoulder_length = shoulder_length

            # Only create normalization if we have valid data
            if (last_x_origin is None or
                last_y_origin is None or
                last_shoulder_length is None):
                continue  # skip until we have a baseline

            self.normalization_data[frame_idx] = FrameNormalization(
                frame_idx,
                last_x_origin,
                last_y_origin,
                last_shoulder_length
            )
    def shoulders_confident(self, frame_idx):
        return self.body["RShoulder"].frames[frame_idx].is_valid() and self.body["LShoulder"].frames[frame_idx].is_valid()
    def build_all_data(self):
        """
        Build all necessary data for this person gesture.
        """
        self.smooth_person_keypoints(window=SMOOTHING_WINDOW)
        self.build_reference_data()
        self.normalize_all_parts()
        self.compute_baseline_all_parts()
        self.build_magnitudes_all_parts()

    def normalize_all_parts(self):
        """
        Normalize all body parts across all frames.
        """
        for part in self.body.values():
            if part.part_name == self.origin_part:
                continue
            part.update_normalized()
        for part in self.left_hand.values():
            part.update_normalized()
        for part in self.right_hand.values():
            part.update_normalized()
    
    def compute_baseline_all_parts(self):
        for part in self.body.values():
            part.compute_baselines()
        # for part in self.left_hand.values():
        #     part.update_normalized()
        # for part in self.right_hand.values():
        #     part.update_normalized()
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
                temp_frame.add_body_part_reference(self.body[part])
                self.body[part].add_keyframe(temp_frame)

        for part, (x, y, c) in keypoint_data.get("left_hand", {}).items():
            pname = f"L_{part}"
            if pname in self.left_hand:
                temp_frame = Frame(frame_idx, x, y, c)
                temp_frame.add_body_part_reference(self.left_hand[pname])
                self.left_hand[pname].add_keyframe(temp_frame)

        for part, (x, y, c) in keypoint_data.get("right_hand", {}).items():
            pname = f"R_{part}"
            if pname in self.right_hand:
                temp_frame = Frame(frame_idx, x, y, c)
                temp_frame.add_body_part_reference(self.right_hand[pname])
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
    def get_origin_part(self):
        """
        Retrieve a BodyPart object by name.

        Args:
            part_name (str): Name of the body part.
        Returns:
            BodyPart object or None if not found.
        """

        return self.body[self.origin_part]

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
        ox, oy = self.get_origin(frame_idx)
        left_shoulder = self.body["LShoulder"].get_coordinates(frame_idx)
        right_shoulder = self.body["RShoulder"].get_coordinates(frame_idx)
        if left_shoulder is None or right_shoulder is None:
            return 0
        lx, ly = left_shoulder
        rx, ry = right_shoulder
        lx -= ox
        rx -= ox
        ly -=oy
        ry -= oy
        return math.hypot(rx - lx, ry - ly)

    def smooth_person_keypoints(self, window=3):
         # Smooth function
        def smooth_array(arr, w):
            smoothed = arr.copy()
            for i in range(n_frames):
                start = max(0, i - w)
                end = min(n_frames, i + w + 1)
                window_vals = arr[start:end]
                if np.any(~np.isnan(window_vals)):
                    smoothed[i] = np.nanmean(window_vals)
                else:
                    smoothed[i] = np.nan
            return smoothed
        for part_name, body_part in self.body.items():
            # Skip if no frames
            if not body_part.frames:
                continue

            # Get frame indices sorted
            frame_indices = sorted(body_part.frames.keys())
            n_frames = len(frame_indices)

            # Extract x and y arrays
            x = np.array([body_part.frames[i].x for i in frame_indices], dtype=float)
            y = np.array([body_part.frames[i].y for i in frame_indices], dtype=float)

            # Replace None with np.nan
            x = np.where(x is None, np.nan, x)
            y = np.where(y is None, np.nan, y)

           

            x_smooth = smooth_array(x, window)
            y_smooth = smooth_array(y, window)

            # Write back smoothed values
            for idx, frame_idx in enumerate(frame_indices):
                frame = body_part.frames[frame_idx]
                frame.x = x_smooth[idx]
                frame.y = y_smooth[idx]
        for part_name, body_part in self.right_hand.items():
            # Skip if no frames
            if not body_part.frames:
                continue

            # Get frame indices sorted
            frame_indices = sorted(body_part.frames.keys())
            n_frames = len(frame_indices)

            # Extract x and y arrays
            x = np.array([body_part.frames[i].x for i in frame_indices], dtype=float)
            y = np.array([body_part.frames[i].y for i in frame_indices], dtype=float)

            # Replace None with np.nan
            x = np.where(x is None, np.nan, x)
            y = np.where(y is None, np.nan, y)

            

            x_smooth = smooth_array(x, window)
            y_smooth = smooth_array(y, window)

            # Write back smoothed values
            for idx, frame_idx in enumerate(frame_indices):
                frame = body_part.frames[frame_idx]
                frame.x = x_smooth[idx]
                frame.y = y_smooth[idx]
        for part_name, body_part in self.left_hand.items():
            # Skip if no frames
            if not body_part.frames:
                continue

            # Get frame indices sorted
            frame_indices = sorted(body_part.frames.keys())
            n_frames = len(frame_indices)

            # Extract x and y arrays
            x = np.array([body_part.frames[i].x for i in frame_indices], dtype=float)
            y = np.array([body_part.frames[i].y for i in frame_indices], dtype=float)

            # Replace None with np.nan
            x = np.where(x is None, np.nan, x)
            y = np.where(y is None, np.nan, y)

            

            x_smooth = smooth_array(x, window)
            y_smooth = smooth_array(y, window)

            # Write back smoothed values
            for idx, frame_idx in enumerate(frame_indices):
                frame = body_part.frames[frame_idx]
                frame.x = x_smooth[idx]
                frame.y = y_smooth[idx]
    def get_normalization_data(self, frame_idx):   
        return self.normalization_data.get(frame_idx, None)
    def __repr__(self):
        return (f"PersonGesture(person_id={self.person_id}, "
                f"body_parts={len(self.body)}, "
                f"left_hand_parts={len(self.left_hand)}, "
                f"right_hand_parts={len(self.right_hand)})")
