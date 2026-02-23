import math
import numpy as np

from data_model.frame import Frame
from math_utility import smooth_keypoints
from config import BASELINE_WINDOW, SMOOTHING_WINDOW

class BodyPart:
    def __init__(self, part_name, person_id,gesture_analysis=None):
        """
        Initialize a BodyPart to track a single keypoint across frames.

        Args:
            part_name (str): Name of the keypoint (e.g., "RWrist", "Thumb_4").
        """
        self.part_name = part_name
        self.person_id = person_id
        self.gesture_analysis = gesture_analysis
        self.frames = {}  # dict[int, Frame]
        self.velocities = []
        self.accelerations = []

    def get_coordinates(self, frame_idx):
        """
        Get the (x, y) coordinates for a specific frame index.

        Args:
            frame_idx (int): Frame number.
        Returns:
            Tuple (x, y) if frame exists, else None.
        """
        frame = self.frames.get(frame_idx)
        if frame is None:
            return (None, None)
        # if not self.confident(frame_idx):
        #     return None
        return (frame.x, frame.y)
    def get_normalized_coordinates(self, frame_idx):
        """
        Get normalized (x, y) coordinates for a specific frame index.

        Args:
            frame_idx (int): Frame number.
        Returns:

            Tuple (x_normalized, y_normalized) if frame exists, else None.
        """
        frame = self.frames.get(frame_idx)
        if frame is None:
            return (None, None)
        if not self.confident(frame_idx):
            return (None, None)
        return (frame.x_normalized, frame.y_normalized)
        # return (frame.x, frame.y)
    def compute_velocity_magnitude(self, frame_idx):
        """
        Get the precomputed velocity magnitude for a frame. If internal arrays are dirty,
        rebuild velocities and accelerations first. Returns 0.0 when value is not available.
        Args:
            frame_idx (int): Frame number.
        Returns:
            Float: velocity magnitude, or 0.0 if not computable.
        """
        # Ensure arrays up to date
        if getattr(self, "_dirty", True):
            self.build_velocities_and_accelerations()

        if frame_idx < 0 or frame_idx >= len(self.velocities):
            return 0.0

        val = self.velocities[frame_idx]
        if not self.confident(frame_idx):
            return 0.0
        try:
            if np.isnan(val):
                return 0.0
        except Exception:
            pass
        return float(val)

    def get_velocity_vector(self, frame_idx):
        # TODO: not relevant for now. Only used in plotting and in a parameter not calculated yet.
        """
        Returns (vx, vy) for the given frame index (velocity resolved in normalized units per second).
        If not available, returns (0.0, 0.0).
        """
        if getattr(self, "_dirty", True):
            self.build_velocities_and_accelerations()

        if not hasattr(self, 'vx') or frame_idx < 0 or frame_idx >= len(self.vx):
            return (0.0, 0.0)

        vx = self.vx[frame_idx]
        vy = self.vy[frame_idx]
        
        if np.isnan(vx) or np.isnan(vy):
            return (0.0, 0.0)
        if not self.confident(frame_idx):
            return (0.0, 0.0)
        return (float(vx), float(vy))

    def get_velocity_magnitude(self, frame_idx):
        """
        Backwards-compatible wrapper to retrieve velocity magnitude as float.
        """
        return self.compute_velocity_magnitude(frame_idx)

    def get_acceleration_magnitude(self, frame_idx):
        """
        Backwards-compatible wrapper to retrieve acceleration magnitude as float.
        """
        if getattr(self, "_dirty", True):
            self.build_velocities_and_accelerations()
        if frame_idx < 0 or frame_idx >= len(self.accelerations):
            return 0.0
        val = self.accelerations[frame_idx]
        if not self.confident(frame_idx):
            return 0.0
        if np.isnan(val):
            return 0.0
        return float(val)

    def compute_acceleration_magnitude(self, frame_idx):
        """
        Compute acceleration magnitude using precomputed arrays (kept for compatibility).
        """
        return self.get_acceleration_magnitude(frame_idx)

    def build_velocities_and_accelerations(self, fps=None):
        """
        Build velocity (vx, vy) and acceleration arrays for the frames of this body part.

        Implementation notes:
        - Uses frame indices as timestamps (frame_no / fps)
        - Handles missing frames by using NaN placeholders
        - Velocity is stored at the later frame index of an interval (i.e., velocity between i-1 and i is stored at i)
        - Acceleration is stored at the later frame index of the velocity interval
        """


        fps = fps or getattr(self.gesture_analysis, 'frame_rate', getattr(Frame, 'FRAME_RATE', 24))

        if not self.frames:
            self.vx = np.array([], dtype=float)
            self.vy = np.array([], dtype=float)
            self.velocities = np.array([], dtype=float)
            self.accelerations = np.array([], dtype=float)
            self._dirty = False
            return

        max_frame = max(self.frames.keys())
        n = max_frame + 1

        x = np.full(n, np.nan, dtype=float)
        y = np.full(n, np.nan, dtype=float)

        for idx, f in self.frames.items():
            # x[idx] = f.x_normalized
            # y[idx] = f.y_normalized
            if self.get_normalized_coordinates(idx) is not None:
                x[idx], y[idx] = self.get_normalized_coordinates(idx)
            else:
                x[idx] = np.nan
                y[idx] = np.nan
        # x = smooth_signal(x, window=3)
        # y = smooth_signal(y, window=3)
        valid = ~np.isnan(x) & ~np.isnan(y)
        idxs = np.nonzero(valid)[0]

        vx = np.full(n, np.nan, dtype=float)
        vy = np.full(n, np.nan, dtype=float)
        vel = np.full(n, np.nan, dtype=float)
        acc = np.full(n, np.nan, dtype=float)

        if len(idxs) >= 2:
            # compute per-interval dt (in seconds)
            dt = np.diff(idxs) #/ float(fps)
            dx = np.diff(x[idxs])
            dy = np.diff(y[idxs])

            # velocity components per interval
            vx_intervals = dx / dt
            vy_intervals = dy / dt
            mag_intervals = np.hypot(vx_intervals, vy_intervals)

            # store velocity at the later frame index of each interval
            vx[idxs[1:]] = vx_intervals
            vy[idxs[1:]] = vy_intervals
            vel[idxs[1:]] = mag_intervals

            # acceleration between successive velocity intervals
            if len(mag_intervals) >= 2:
                dv = np.diff(mag_intervals)
                dt2 = np.diff(idxs[1:]) / float(fps)
                acc_vals = np.abs(dv / dt2)
                acc[idxs[1:][1:]] = acc_vals

        # Assign arrays
        self.vx = vx
        self.vy = vy
        self.velocities = smooth_keypoints(vel,SMOOTHING_WINDOW)
        self.accelerations = smooth_keypoints(acc,SMOOTHING_WINDOW)
        self._dirty = False

    # def update_normalized(self):
    #     person = self.gesture_analysis.get_person_by_id(self.person_id)
    #     for frame in self.frames.values():
    #         frame.update_normalized(
    #             person.avg_origin_x,
    #             person.avg_origin_y,
    #             person.avg_shoulder_length
    #         )
    #     # Mark arrays as stale
    #     self._dirty = True

    def confident(self, frame_idx, confidence_threshold=0.7):
        """
        Returns whether the frame at frame_idx has a confidence score above the threshold.
        """
        frame = self.frames.get(frame_idx)
        if frame is None:
            return False
        return frame.is_valid(confidence_threshold)

    def add_keyframe(self, frame):
        """
        Add or update a frame observation.

        Args:
            frame (Frame): Frame object.
        """
        # store by frame_no for O(1) lookup
        frame.add_body_part_reference(self)
        self.frames[frame.frame_no] = frame
        # Mark cached arrays as stale
        self._dirty = True
    def update_normalized(self):
        person = self.gesture_analysis.get_person_by_id(self.person_id)
        for frame in self.frames.values():
            normalization_data = person.get_normalization_data(frame.frame_no)
            frame.update_normalized(
                normalization_data.x_origin,
                normalization_data.y_origin,
                normalization_data.shoulder_length
            )
        self._dirty = True

    # def compute_baselines(self, baseline_window=BASELINE_WINDOW):
    #     self.baselines = {}
    #     if not self.frames:
    #         return
    #     # Group frames by block
    #     blocks = {}
    #     for frame_idx, frame in self.frames.items():
    #         if frame.x_normalized is None or frame.y_normalized is None:
    #             continue
    #         block_idx = frame_idx // baseline_window
    #         blocks.setdefault(block_idx, []).append((frame.x_normalized, frame.y_normalized))

    #     # Compute baseline per block
    #     for block_idx, coords in blocks.items():
    #         xs, ys = zip(*coords)
    #         self.baselines[block_idx] = (
    #             float(np.median(xs)),
    #             float(np.median(ys))
    #         )

    def compute_baselines(
        self,
        alpha=0.998,
        # alpha=0.995,
        max_update_dist=0.25
    ):
        """
        Computes a self-updating baseline using EMA, per frame.

        Parameters
        ----------
        alpha : float
            EMA smoothing factor (close to 1 = slow baseline)
        max_update_dist : float
            Max distance from baseline to allow baseline update.
            Prevents gestures from polluting baseline.
        """

        self.baselines = {}

        if not self.frames:
            return

        frame_indices = sorted(self.frames.keys())

        # Initialize baseline from first valid frame
        baseline_x = None
        baseline_y = None

        for frame_idx in frame_indices:
            # frame = self.frames[frame_idx]
            x_normalized, y_normalized = self.get_normalized_coordinates(frame_idx)
            if x_normalized is None or y_normalized is None:
                continue

            if baseline_x is None:
                baseline_x = x_normalized
                baseline_y = y_normalized
                self.baselines[frame_idx] = (baseline_x, baseline_y)
                continue

            dx = x_normalized - baseline_x
            dy = y_normalized - baseline_y
            dist = np.hypot(dx, dy)
            # if self.part_name == "RWrist":
            #     print(dist)
            # Only update baseline if movement is small
            if dist < max_update_dist:
            # if True:
                baseline_x = alpha * baseline_x + (1 - alpha) * x_normalized
                baseline_y = alpha * baseline_y + (1 - alpha) * y_normalized

            # Store baseline (even if frozen)
            self.baselines[frame_idx] = (baseline_x, baseline_y)
    def __len__(self):
        return len(self.frames)

    def __repr__(self):
        return (f"BodyPart('{self.part_name}', frames={len(self.frames)}")

    def display_frames(self):
        """
        Print all frames for this gesture part in order.
        """
        for fn in sorted(self.frames.keys()):
            print(self.frames[fn])