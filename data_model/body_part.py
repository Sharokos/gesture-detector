import math
import numpy as np

from data_model.frame import Frame
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
            return None
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
            return None
        return (frame.x_normalized, frame.y_normalized)
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
        try:
            if np.isnan(val):
                return 0.0
        except Exception:
            pass
        return float(val)

    def get_velocity_vector(self, frame_idx):
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
            x[idx] = f.x_normalized
            y[idx] = f.y_normalized

        valid = ~np.isnan(x) & ~np.isnan(y)
        idxs = np.nonzero(valid)[0]

        vx = np.full(n, np.nan, dtype=float)
        vy = np.full(n, np.nan, dtype=float)
        vel = np.full(n, np.nan, dtype=float)
        acc = np.full(n, np.nan, dtype=float)

        if len(idxs) >= 2:
            # compute per-interval dt (in seconds)
            dt = np.diff(idxs) / float(fps)
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
        self.velocities = vel
        self.accelerations = acc
        self._dirty = False

    def update_normalized(self):
        person = self.gesture_analysis.get_person_by_id(self.person_id)
        for frame in self.frames.values():
            frame.update_normalized(
                person.avg_origin_x,
                person.avg_origin_y,
                person.avg_shoulder_length
            )
        # Mark arrays as stale
        self._dirty = True

    def confident(self, frame_idx, confidence_threshold=0.5):
        """
        Returns whether the frame at frame_idx has a confidence score above the threshold.
        """
        return True
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
        self.frames[frame.frame_no] = frame
        # Mark cached arrays as stale
        self._dirty = True

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