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
        self.max_acceleration = self.compute_max_acceleration()
        self.median_x, self.median_y = self.compute_median_coords()
        self.motion_saliency = self.compute_motion_saliency()
        self.burstiness = self.compute_burstiness()
        self.directional_consistency = self.compute_directional_consistency()
        self.direction_changes = self.compute_direction_changes()
        self.path_efficiency = self.compute_path_efficiency()


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
        return total_velocity / count if count > 0 else 0.0
    
    def compute_max_acceleration(self):
        # Not calculating mean because we are interested in a "burst" of movement and we want to see the maximum value
        # TODO: temporarly calculating the mean instead of max
        accelerations = []
        count = 0
        for frame_idx in range(self.sw.start_frame + 1, self.sw.end_frame + 1):
            acc = self.body_part.get_acceleration_magnitude(frame_idx)
            accelerations.append(acc)
            count += 1
        # return sum(accelerations)/count
        return max(accelerations)
    def compute_motion_saliency(self, epsilon=1e-6):
        velocities = []
        for frame_idx in range(self.sw.start_frame + 1, self.sw.end_frame + 1):
            v = self.body_part.get_velocity_magnitude(frame_idx)
            velocities.append(v)

        if not velocities:
            return 0.0

        max_v = max(velocities)
        mean_v = sum(velocities) / len(velocities)

        return max_v / (mean_v + epsilon)
    def compute_motion_persistence(self, threshold=0.03):
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

    def compute_directional_consistency(self, min_velocity_threshold=0.03):
        vx_sum, vy_sum = 0.0, 0.0
        magnitude_sum = 0.0

        for frame_idx in range(self.sw.start_frame + 1, self.sw.end_frame + 1):
            vx, vy = self.body_part.get_velocity_vector(frame_idx)
            mag = (vx**2 + vy**2)**0.5

            if mag < min_velocity_threshold:
                continue

            vx_sum += vx
            vy_sum += vy
            magnitude_sum += mag

        if magnitude_sum == 0:
            return 0.0

        net_motion = (vx_sum**2 + vy_sum**2)**0.5
        return net_motion / magnitude_sum
    
    def compute_burstiness(self, epsilon=1e-6):
        velocities = []
        for frame_idx in range(self.sw.start_frame + 1, self.sw.end_frame + 1):
            velocities.append(self.body_part.get_velocity_magnitude(frame_idx))

        if not velocities:
            return 0.0

        mean_v = sum(velocities) / len(velocities)
        std_v = np.std(velocities)

        return std_v / (mean_v + epsilon)
    

    def compute_direction_changes(self, min_velocity_threshold=0.03):
        prev_vx, prev_vy = None, None
        changes = 0

        for frame_idx in range(self.sw.start_frame + 1, self.sw.end_frame + 1):
            vx, vy = self.body_part.get_velocity_vector(frame_idx)
            mag = (vx**2 + vy**2)**0.5

            if mag < min_velocity_threshold:
                continue

            if prev_vx is not None:
                dot = vx * prev_vx + vy * prev_vy
                if dot < 0:  # opposite direction
                    changes += 1

            prev_vx, prev_vy = vx, vy

        return changes
    

    def compute_path_efficiency(self):
        total_dist = 0.0
        start = None
        end = None
        prev = None

        for frame_idx in range(self.sw.start_frame + 1, self.sw.end_frame + 1):
            coords = self.body_part.get_normalized_coordinates(frame_idx)

            if coords is None:
                continue

            x, y = coords

            if x is None or y is None:
                continue

            if start is None:
                start = (x, y)

            if prev is not None:
                dx = x - prev[0]
                dy = y - prev[1]
                total_dist += (dx**2 + dy**2) ** 0.5

            prev = (x, y)
            end = (x, y)

        if start is None or end is None or total_dist == 0:
            return 0.0

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        displacement = (dx**2 + dy**2) ** 0.5

        return displacement / total_dist