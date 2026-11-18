import copy

import numpy as np
import pandas as pd

from config import (
    ALPHA_CONFIG,
    COCO_PARTS,
    HAND_PARTS,
    KEYPOINT_CONFIDENCE_THRESHOLD,
    MAX_INTERPOLATION_GAP,
    MAX_UPDATE_DIST_CONFIG,
    ORIGIN_PART,
    SMOOTHING_WINDOW,
    frames_at_fps,
)
from data_model.body_part import BodyPart
from data_model.frame_normalization import FrameNormalization


class PersonGesture:
    def __init__(self, person_id, origin=ORIGIN_PART, gesture_analysis=None):
        self.person_id = person_id
        self.gesture_analysis = gesture_analysis
        self.body = {part: BodyPart(part, person_id, gesture_analysis) for part in COCO_PARTS}
        self.left_hand = {
            f"L_{part}": BodyPart(f"L_{part}", person_id, gesture_analysis) for part in HAND_PARTS
        }
        self.right_hand = {
            f"R_{part}": BodyPart(f"R_{part}", person_id, gesture_analysis) for part in HAND_PARTS
        }
        self.origin_part = origin
        self.normalization_data = {}
        self.avg_shoulder_length = np.nan

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def set_keypoints(self, body, left_hand=None, right_hand=None):
        """Arrays of shape (n_frames, n_parts, 3), ordered like COCO_PARTS / HAND_PARTS."""
        for i, part in enumerate(COCO_PARTS):
            self.body[part].set_raw(body[:, i])
        if left_hand is not None:
            for i, part in enumerate(HAND_PARTS):
                self.left_hand[f"L_{part}"].set_raw(left_hand[:, i])
        if right_hand is not None:
            for i, part in enumerate(HAND_PARTS):
                self.right_hand[f"R_{part}"].set_raw(right_hand[:, i])

    def _all_parts(self):
        parts = list(self.body.values())
        parts += [p for p in self.left_hand.values() if p.has_data()]
        parts += [p for p in self.right_hand.values() if p.has_data()]
        return parts

    @property
    def fps(self):
        return self.gesture_analysis.frame_rate

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def build_all_data(self, smoothing_window=SMOOTHING_WINDOW):
        half_window = frames_at_fps(smoothing_window, self.fps)
        self.smooth_person_keypoints(half_window)
        self.build_reference_data()
        self.normalize_all_parts()
        self.compute_baseline_all_parts()
        self.build_magnitudes_all_parts(half_window)

    def build_all_data_debug_smooth(self, smoothing_windows):
        part_list = []
        for window in smoothing_windows:
            self.build_all_data(smoothing_window=window)
            part_list.append(copy.deepcopy(self.get_body_part("LWrist")))
        return part_list

    def smooth_person_keypoints(self, half_window):
        max_gap = frames_at_fps(MAX_INTERPOLATION_GAP, self.fps)
        for part in self._all_parts():
            part.clean_and_smooth(KEYPOINT_CONFIDENCE_THRESHOLD, max_gap, half_window)

    def build_average_shoulder_length(self):
        rs, ls = self.body["RShoulder"], self.body["LShoulder"]
        lengths = np.hypot(rs.x - ls.x, rs.y - ls.y)
        lengths = lengths[np.isfinite(lengths)]
        self.avg_shoulder_length = float(np.median(lengths)) if len(lengths) else np.nan

    def build_reference_data(self):
        """
        Per-frame origin (ORIGIN_PART, carried over frames where it is missing) and one scale for the
        whole video (median shoulder width), so coordinates are in "shoulder widths" and comparable
        between people / camera distances.
        """
        self.build_average_shoulder_length()
        origin = self.body[self.origin_part]
        # Carry the last valid origin forward (and the first one backward for the very first frames)
        self.origin_x = pd.Series(origin.x).ffill().bfill().to_numpy(copy=True)
        self.origin_y = pd.Series(origin.y).ffill().bfill().to_numpy(copy=True)

        self.normalization_data = {}
        if not np.isfinite(self.avg_shoulder_length):
            print("WARNING: shoulders never detected confidently, cannot normalize this video")
            return
        for i in np.nonzero(np.isfinite(self.origin_x))[0]:
            self.normalization_data[int(i)] = FrameNormalization(
                int(i), self.origin_x[i], self.origin_y[i], self.avg_shoulder_length
            )

    def normalize_all_parts(self):
        for part in self._all_parts():
            part.update_normalized(self.origin_x, self.origin_y, self.avg_shoulder_length)

    def compute_baseline_all_parts(self, alpha=ALPHA_CONFIG, max_update_dist=MAX_UPDATE_DIST_CONFIG):
        for part in self.body.values():
            part.compute_baselines(alpha=alpha, max_update_dist=max_update_dist, fps=self.fps)

    def build_magnitudes_all_parts(self, half_window):
        for part in self._all_parts():
            part.build_velocities_and_accelerations(fps=self.fps, half_window=half_window)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_body_part(self, part_name):
        if part_name in self.body:
            return self.body[part_name]
        if part_name in self.left_hand:
            return self.left_hand[part_name]
        if part_name in self.right_hand:
            return self.right_hand[part_name]
        return None

    def get_origin_part(self):
        return self.body[self.origin_part]

    def get_origin(self, frame_idx):
        return self.body[self.origin_part].get_coordinates(frame_idx)

    def get_normalization_data(self, frame_idx):
        return self.normalization_data.get(frame_idx, None)

    def __repr__(self):
        return (
            f"PersonGesture(person_id={self.person_id}, "
            f"body_parts={len(self.body)}, "
            f"left_hand_parts={len(self.left_hand)}, "
            f"right_hand_parts={len(self.right_hand)})"
        )
