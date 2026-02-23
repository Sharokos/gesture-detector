from data_model.body_part import BodyPart
import numpy as np
class Features():            
    def __init__(self, body_part: BodyPart, sw):
        self.body_part = body_part
        self.sw = sw
        self.build_common_measures()

    def build_common_measures(self):
        self.motion_energy = self.compute_motion_energy()
        self.mean_velocity = self.compute_mean_velocity()
        self.motion_persistence = self.compute_motion_persistence()
        self.velocity_variance = self.compute_velocity_variance()
        # self.directional_consistency = self.compute_directional_consistency()
        self.max_acceleration = self.compute_max_acceleration()
        self.median_x, self.median_y = self.compute_median_coords()


    def compute_median_coords(self):
        xs = []
        ys = []

        for frame_idx in range(self.sw.start_frame + 1, self.sw.end_frame + 1):
            coords = self.body_part.get_normalized_coordinates(frame_idx)
            if coords is None:
                continue

            x, y = coords
            if x is None or y is None:
                continue

            xs.append(x)
            ys.append(y)

        if not xs:
            return None, None

        return float(np.median(xs)), float(np.median(ys))
    def compute_motion_energy(self):

        part_energy = 0.0
        for frame_idx in range(self.sw.start_frame + 1, self.sw.end_frame + 1):

            v = self.body_part.get_velocity_magnitude(frame_idx)
            part_energy += v ** 2
        part_energy /= (self.sw.duration_seconds)
        return part_energy
    
    def compute_mean_velocity(self):
        total_velocity = 0.0
        count = 0
        # origin = self.sw.person.get_origin_part()
        for frame_idx in range(self.sw.start_frame + 1, self.sw.end_frame + 1):
            velocity = self.body_part.get_velocity_magnitude(frame_idx)
            # velocity -= origin.get_velocity_magnitude(frame_idx)
            total_velocity += velocity
            count += 1
        return np.median(total_velocity)
    
    def compute_max_acceleration(self):
        # Not calculating mean because we are interested in a "burst" of movement and we want to see the maximum value
        # TODO: temporarly calculating the mean instead of max
        accelerations = []
        count = 0
        for frame_idx in range(self.sw.start_frame + 1, self.sw.end_frame + 1):
            acc = self.body_part.get_acceleration_magnitude(frame_idx)
            accelerations.append(acc)
            count += 1
        return sum(accelerations)/count
    
    def compute_motion_persistence(self, threshold=0.01):
        # Ratio of how many frames contain movement in a window
        total_frames = self.sw.WINDOW_SIZE

        moving_frames = 0
        for frame_idx in range(self.sw.start_frame + 1, self.sw.end_frame + 1):
            v = self.body_part.get_velocity_magnitude(frame_idx)
            if v > threshold:
                moving_frames += 1
        persistence_per_part = moving_frames / total_frames 
        return persistence_per_part
    
    def compute_velocity_variance(self):
        velocities = []
        for frame_idx in range(self.sw.start_frame + 1, self.sw.end_frame + 1):
            velocity = self.body_part.get_velocity_magnitude(frame_idx)
            velocities.append(velocity)

        if len(velocities) == 0:
            return 0.0

        mean_velocity = sum(velocities) / len(velocities)
        variance = sum((v - mean_velocity) ** 2 for v in velocities) / len(velocities)

        return variance

    def compute_directional_consistency(self, min_velocity_threshold=0.1):
        vx_sum, vy_sum = 0.0, 0.0
        magnitude_sum = 0.0
        for frame_idx in range(self.sw.start_frame + 1, self.sw.end_frame + 1):
            vx, vy = self.body_part.get_velocity_vector(frame_idx)
            vel_mag = (vx**2 + vy**2)**0.5
            if vel_mag < min_velocity_threshold:
                continue  # ignore tiny motions
            vx_sum += vx
            vy_sum += vy
            magnitude_sum += vel_mag

        if magnitude_sum == 0:
            return 0.0  # no motion

        # Ratio of net motion to total motion
        consistency = (vx_sum**2 + vy_sum**2)**0.5 / magnitude_sum
        return min(max(consistency, 0.0), 1.0)  # limit the return between 0 and 1