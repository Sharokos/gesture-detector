import numpy as np
import pandas as pd

from data_model.frame import Frame
from config import REFERENCE_FPS


def fill_short_gaps(a, max_gap):
    """Linearly interpolate runs of NaN that are at most `max_gap` long (inside the signal only)."""
    s = pd.Series(a)
    isna = s.isna()
    if not isna.any() or isna.all():
        return a.copy()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    interp = s.interpolate(limit_area="inside")
    fill = (isna & (run_len <= max_gap)).to_numpy()
    out = a.copy()
    out[fill] = interp.to_numpy()[fill]
    return out


def centered_nanmean(a, half_window):
    """Centered moving average over [i - w, i + w] ignoring NaN (same as the old loop-based smoothing)."""
    if half_window <= 0 or len(a) == 0:
        return a.copy()
    return (
        pd.Series(a)
        .rolling(2 * half_window + 1, center=True, min_periods=1)
        .mean()
        .to_numpy(copy=True)  # pandas 3 returns read-only views otherwise
    )


class BodyPart:
    """
    One keypoint tracked across the whole video. Data is stored as numpy arrays indexed by frame
    number (NaN = missing / not confident).
    """

    def __init__(self, part_name, person_id, gesture_analysis=None):
        self.part_name = part_name
        self.person_id = person_id
        self.gesture_analysis = gesture_analysis

        empty = np.array([], dtype=float)
        self.raw_x = self.raw_y = self.conf = empty
        self.x = self.y = empty          # cleaned + smoothed pixel coordinates
        self.valid = np.array([], dtype=bool)
        self.xn = self.yn = empty        # normalized coordinates
        self.vx = self.vy = empty
        self.velocities = empty
        self.accelerations = empty
        self.baseline_x = self.baseline_y = empty
        self.baselines = {}
        self._vel_m = self._acc_m = self._vx_m = self._vy_m = empty
        self._frames = None

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def set_raw(self, xyc):
        """xyc: array (n_frames, 3) with x, y, confidence; NaN where the person was not detected."""
        self.raw_x = xyc[:, 0].astype(float)
        self.raw_y = xyc[:, 1].astype(float)
        self.conf = xyc[:, 2].astype(float)
        self._frames = None

    def __len__(self):
        return int(np.count_nonzero(~np.isnan(self.conf)))

    def has_data(self):
        return len(self) > 0

    def clean_and_smooth(self, conf_threshold, max_gap, half_window):
        with np.errstate(invalid="ignore"):
            ok = self.conf >= conf_threshold
        x = np.where(ok, self.raw_x, np.nan)
        y = np.where(ok, self.raw_y, np.nan)
        x = fill_short_gaps(x, max_gap)
        y = fill_short_gaps(y, max_gap)
        self.valid = ~np.isnan(x) & ~np.isnan(y)
        xs = centered_nanmean(x, half_window)
        ys = centered_nanmean(y, half_window)
        xs[~self.valid] = np.nan
        ys[~self.valid] = np.nan
        self.x, self.y = xs, ys
        self._frames = None

    def update_normalized(self, x_origin, y_origin, scale):
        """x_origin / y_origin: per-frame arrays, scale: scalar (e.g. shoulder width in pixels)."""
        if scale is None or not np.isfinite(scale) or scale == 0:
            self.xn = np.full_like(self.x, np.nan)
            self.yn = np.full_like(self.y, np.nan)
        else:
            self.xn = (self.x - x_origin) / scale
            self.yn = (self.y - y_origin) / scale
        self._frames = None

    def build_velocities_and_accelerations(self, fps=None, half_window=0):
        """
        Velocity is expressed in normalized units per REFERENCE frame (i.e. per 1/REFERENCE_FPS s),
        which keeps the old magnitudes at 30 fps but makes them independent of the video frame rate.
        Velocity between frames i-1 and i is stored at i. Acceleration = |d velocity| / dt(seconds).
        """
        fps = fps or getattr(self.gesture_analysis, "frame_rate", REFERENCE_FPS)
        n = len(self.xn)
        vx = np.full(n, np.nan)
        vy = np.full(n, np.nan)
        vel = np.full(n, np.nan)
        acc = np.full(n, np.nan)

        valid = ~np.isnan(self.xn) & ~np.isnan(self.yn)
        idxs = np.nonzero(valid)[0]
        if len(idxs) >= 2:
            dt = np.diff(idxs)
            scale = fps / REFERENCE_FPS
            vx_i = np.diff(self.xn[idxs]) / dt * scale
            vy_i = np.diff(self.yn[idxs]) / dt * scale
            mag_i = np.hypot(vx_i, vy_i)
            vx[idxs[1:]] = vx_i
            vy[idxs[1:]] = vy_i
            vel[idxs[1:]] = mag_i
            if len(mag_i) >= 2:
                dt2 = np.diff(idxs[1:]) / float(fps)
                acc[idxs[2:]] = np.abs(np.diff(mag_i) / dt2)

        self.vx, self.vy = vx, vy
        self.velocities = centered_nanmean(vel, half_window)
        self.accelerations = centered_nanmean(acc, half_window)

        # Masked versions used by the features: 0 where not valid / not computable
        def masked(a):
            return np.where(valid & np.isfinite(a), a, 0.0)

        self._vel_m = masked(self.velocities)
        self._acc_m = masked(self.accelerations)
        self._vx_m = masked(vx)
        self._vy_m = masked(vy)

    def compute_baselines(self, alpha=0.998, max_update_dist=0.14, fps=None):
        """
        Self-updating rest position (EMA) per frame. The baseline only moves while the keypoint is
        close to it, so gestures don't pollute it. It starts at the median position over the whole
        video (for a seated person that is the rest pose) instead of the first frame, which could be
        in the middle of a gesture and freeze the baseline in the wrong place.
        """
        n = len(self.xn)
        bx_arr = np.full(n, np.nan)
        by_arr = np.full(n, np.nan)
        self.baselines = {}
        valid = ~np.isnan(self.xn) & ~np.isnan(self.yn)
        if not valid.any():
            self.baseline_x, self.baseline_y = bx_arr, by_arr
            return

        fps = fps or getattr(self.gesture_analysis, "frame_rate", REFERENCE_FPS)
        a = alpha ** (REFERENCE_FPS / fps)  # same time constant at any fps

        bx = float(np.median(self.xn[valid]))
        by = float(np.median(self.yn[valid]))
        xs, ys = self.xn, self.yn
        for i in np.nonzero(valid)[0]:
            x, y = xs[i], ys[i]
            if np.hypot(x - bx, y - by) < max_update_dist:
                bx = a * bx + (1 - a) * x
                by = a * by + (1 - a) * y
            bx_arr[i] = bx
            by_arr[i] = by
            self.baselines[int(i)] = (bx, by)
        self.baseline_x, self.baseline_y = bx_arr, by_arr

    # ------------------------------------------------------------------
    # Window slices used by the features (vectorized)
    # ------------------------------------------------------------------
    def window(self, start, end):
        """Slice [start, end) of the masked per-frame arrays."""
        return {
            "vel": self._vel_m[start:end],
            "acc": self._acc_m[start:end],
            "vx": self._vx_m[start:end],
            "vy": self._vy_m[start:end],
            "xn": self.xn[start:end],
            "yn": self.yn[start:end],
        }

    # ------------------------------------------------------------------
    # Per-frame accessors (kept for the debug exporters / plots)
    # ------------------------------------------------------------------
    def _in_range(self, frame_idx):
        return 0 <= frame_idx < len(self.valid)

    def confident(self, frame_idx, confidence_threshold=None):
        return self._in_range(frame_idx) and bool(self.valid[frame_idx])

    def get_coordinates(self, frame_idx):
        if not self._in_range(frame_idx) or not self.valid[frame_idx]:
            return (None, None)
        return (float(self.x[frame_idx]), float(self.y[frame_idx]))

    def get_normalized_coordinates(self, frame_idx):
        if not self._in_range(frame_idx) or np.isnan(self.xn[frame_idx]) or np.isnan(self.yn[frame_idx]):
            return (None, None)
        return (float(self.xn[frame_idx]), float(self.yn[frame_idx]))

    def get_velocity_magnitude(self, frame_idx):
        return float(self._vel_m[frame_idx]) if 0 <= frame_idx < len(self._vel_m) else 0.0

    compute_velocity_magnitude = get_velocity_magnitude

    def get_velocity_vector(self, frame_idx):
        if not 0 <= frame_idx < len(self._vx_m):
            return (0.0, 0.0)
        return (float(self._vx_m[frame_idx]), float(self._vy_m[frame_idx]))

    def get_acceleration_magnitude(self, frame_idx):
        return float(self._acc_m[frame_idx]) if 0 <= frame_idx < len(self._acc_m) else 0.0

    compute_acceleration_magnitude = get_acceleration_magnitude

    @property
    def frames(self):
        """dict[frame_idx -> Frame] built on demand (only the debug exporters need it)."""
        if self._frames is None:
            self._frames = {}
            for i in np.nonzero(~np.isnan(self.conf))[0]:
                i = int(i)
                x = self.x[i] if len(self.x) and not np.isnan(self.x[i]) else self.raw_x[i]
                y = self.y[i] if len(self.y) and not np.isnan(self.y[i]) else self.raw_y[i]
                f = Frame(i, float(x), float(y), float(self.conf[i]))
                f.add_body_part_reference(self)
                if len(self.xn) and not np.isnan(self.xn[i]):
                    f.x_normalized = float(self.xn[i])
                    f.y_normalized = float(self.yn[i])
                self._frames[i] = f
        return self._frames

    def __repr__(self):
        return f"BodyPart('{self.part_name}', frames={len(self)})"

    def display_frames(self):
        for fn in sorted(self.frames.keys()):
            print(self.frames[fn])
