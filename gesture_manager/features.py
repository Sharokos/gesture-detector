import numpy as np

from config import REFERENCE_FPS
from data_model.body_part import BodyPart

MIN_VELOCITY = 0.03
EPSILON = 1e-6


class Features:
    """Per-window features of one body part. Same definitions as before, computed on numpy slices."""

    def __init__(self, body_part: BodyPart, sw):
        self.body_part = body_part
        self.sw = sw
        self.build_common_measures()

    def build_common_measures(self):
        # The window covers frames start_frame+1 .. end_frame (velocity at i = motion from i-1 to i)
        w = self.body_part.window(self.sw.start_frame + 1, self.sw.end_frame + 1)
        v, acc, vx, vy = w["vel"], w["acc"], w["vx"], w["vy"]
        xn, yn = w["xn"], w["yn"]
        n = len(v)
        fps = self.sw.fps

        if n == 0:
            v = np.zeros(1)
            acc = np.zeros(1)

        mean_v = float(v.mean())
        # Energy: sum(v^2) per second, rescaled so it does not grow with the frame rate
        self.motion_energy = float(np.sum(v ** 2) / self.sw.duration_seconds * (REFERENCE_FPS / fps))
        self.mean_velocity = mean_v
        self.motion_persistence = float(np.count_nonzero(v > MIN_VELOCITY) / self.sw.duration_frames)
        self.velocity_variance = float(v.var())
        self.max_acceleration = float(acc.max())

        coords_ok = ~np.isnan(xn) & ~np.isnan(yn)
        if coords_ok.any():
            self.median_x = float(np.median(xn[coords_ok]))
            self.median_y = float(np.median(yn[coords_ok]))
        else:
            self.median_x, self.median_y = None, None

        self.motion_saliency = float(v.max() / (mean_v + EPSILON))
        self.burstiness = float(v.std() / (mean_v + EPSILON))

        # Directional measures only use frames with real motion
        mag = np.hypot(vx, vy)
        moving = mag >= MIN_VELOCITY
        mvx, mvy = vx[moving], vy[moving]
        mag_sum = mag[moving].sum()
        self.directional_consistency = (
            float(np.hypot(mvx.sum(), mvy.sum()) / mag_sum) if mag_sum > 0 else 0.0
        )
        if len(mvx) >= 2:
            dots = mvx[1:] * mvx[:-1] + mvy[1:] * mvy[:-1]
            self.direction_changes = int(np.count_nonzero(dots < 0))
        else:
            self.direction_changes = 0

        px, py = xn[coords_ok], yn[coords_ok]
        if len(px) >= 2:
            total = np.hypot(np.diff(px), np.diff(py)).sum()
            disp = np.hypot(px[-1] - px[0], py[-1] - py[0])
            self.path_efficiency = float(disp / total) if total > 0 else 0.0
        else:
            self.path_efficiency = 0.0
