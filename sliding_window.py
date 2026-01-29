class SlidingWindow():

    WINDOW_SIZE = 15 # frames 
    STEP_SIZE = 5   # frames 
    INTEREST_PARTS = ["LElbow", "RShoulder", "LShoulder", "RElbow", "RWrist", "LWrist" ]

    def __init__(self, sliding_window_id, start_frame, end_frame, person):

        self.id = sliding_window_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.person = person
    
    def build_features(self):
        """
        Build all features for this sliding window.
        """
        self.motion_energy = self.compute_motion_energy()
        self.mean_velocity = self.compute_mean_velocity()
        self.motion_persistance_ratio = self.compute_motion_persistance_ratio()
        self.velocity_variance = self.compute_velocity_variance()
        self.distal_proximal_motion_ratio = self.compute_distal_proximal_motion_ratio()
        
    def compute_motion_energy(self):
        """
        Compute motion energy for this sliding window.

        Returns:
            Float: motion energy value.
        """
        energy = 0.0

        for part_name, body_part in self.person.body.items():
            if part_name not in SlidingWindow.INTEREST_PARTS:
                continue
            for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                velocity = (body_part.get_velocity_magnitude(frame_idx))**2
                energy += velocity
        return energy
    def compute_mean_velocity(self):
        """
        Compute mean velocity for this sliding window.

        Returns:
            Float: mean velocity value.
        """
        total_velocity = 0.0
        count = 0

        for part_name, body_part in self.person.body.items():
            if part_name not in SlidingWindow.INTEREST_PARTS:
                continue
            for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                velocity = body_part.get_velocity_magnitude(frame_idx)
                total_velocity += velocity
                count += 1

        return total_velocity / count if count > 0 else 0.0
    
    def compute_motion_persistance_ratio(self):
        """
        Compute motion persistance ratio for this sliding window.

        Returns:
            Float: motion persistance ratio value.
        """
        moving_frames = 0
        total_frames = self.end_frame - self.start_frame + 1

        threshold = 0.1  # Define a velocity threshold to consider as "moving"

        for part_name, body_part in self.person.body.items():
            if part_name not in SlidingWindow.INTEREST_PARTS:
                continue
            for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                velocity = body_part.get_velocity_magnitude(frame_idx)
                if velocity > threshold:
                    moving_frames += 1

        return moving_frames / (total_frames * len(SlidingWindow.INTEREST_PARTS)) if total_frames > 0 else 0.0
    def compute_velocity_variance(self):
        """
        Compute velocity variance for this sliding window.

        Returns:
            Float: velocity variance value.
        """
        velocities = []

        for part_name, body_part in self.person.body.items():
            if part_name not in SlidingWindow.INTEREST_PARTS:
                continue
            for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                velocity = body_part.get_velocity_magnitude(frame_idx)
                velocities.append(velocity)

        if len(velocities) == 0:
            return 0.0

        mean_velocity = sum(velocities) / len(velocities)
        variance = sum((v - mean_velocity) ** 2 for v in velocities) / len(velocities)

        return variance
    
    def compute_distal_proximal_motion_ratio(self):
        """
        Compute distal to proximal motion ratio for this sliding window.

        Returns:
            Float: distal to proximal motion ratio value.
        """
        distal_parts = ["RWrist", "LWrist", "RElbow", "LElbow"]
        proximal_parts = ["RShoulder", "LShoulder"]

        distal_motion = 0.0
        proximal_motion = 0.0

        for part_name, body_part in self.person.body.items():
            if part_name in distal_parts:
                for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                    distal_motion += body_part.get_velocity_magnitude(frame_idx)
            elif part_name in proximal_parts:
                for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                    proximal_motion += body_part.get_velocity_magnitude(frame_idx)

        if proximal_motion == 0:
            return 0.0

        return distal_motion / proximal_motion
    
    def contains_gesture(self, 
                        motion_energy_threshold=0.05,
                        mean_velocity_threshold=0.025,
                        motion_persistance_threshold=0.0,
                        distal_proximal_threshold=1.0):
        """
        Determine whether this sliding window contains a gesture.
        
        A gesture is detected if multiple motion features exceed their thresholds.
        
        Args:
            motion_energy_threshold (float): Minimum motion energy to indicate gesture activity.
            mean_velocity_threshold (float): Minimum mean velocity to indicate gesture.
            motion_persistance_threshold (float): Minimum ratio of frames with motion.
            distal_proximal_threshold (float): Minimum ratio of hand/elbow to shoulder motion.
        
        Returns:
            bool: True if gesture detected, False otherwise.
        """
        # Ensure features have been computed
        if not hasattr(self, 'motion_energy'):
            self.build_features()
        
        # Gesture detected if at least 3 of these conditions are met
        conditions = [
            self.motion_energy > motion_energy_threshold,
            self.mean_velocity > mean_velocity_threshold,
            self.motion_persistance_ratio > motion_persistance_threshold,
            self.distal_proximal_motion_ratio > distal_proximal_threshold
        ]
        
        # Require at least 3 out of 4 conditions to classify as gesture
        return sum(conditions) >= 3
    
    def should_merge_with(self, other_window, 
                         max_temporal_gap=6,
                         feature_similarity_threshold=0.7):
        """
        Determine if this window should be merged with another window.
        
        Two windows are merged if:
        1. They are temporally close (within max_temporal_gap frames)
        2. Their motion features are similar (above similarity threshold)
        
        Args:
            other_window (SlidingWindow): The other window to compare with.
            max_temporal_gap (int): Maximum frames allowed between windows to merge.
            feature_similarity_threshold (float): Minimum similarity (0-1) of normalized features.
        
        Returns:
            bool: True if windows should be merged, False otherwise.
        """
        # Ensure both windows have features computed
        if not hasattr(self, 'motion_energy'):
            self.build_features()
        if not hasattr(other_window, 'motion_energy'):
            other_window.build_features()
        
        # Check temporal proximity
        temporal_gap = min(
            abs(other_window.start_frame - self.end_frame),
            abs(self.start_frame - other_window.end_frame)
        )
        
        if temporal_gap > max_temporal_gap:
            return False
        
        # Compute feature similarity (normalized L2 distance)
        features_self = [
            self.motion_energy,
            self.mean_velocity,
            self.motion_persistance_ratio,
            self.distal_proximal_motion_ratio
        ]
        
        features_other = [
            other_window.motion_energy,
            other_window.mean_velocity,
            other_window.motion_persistance_ratio,
            other_window.distal_proximal_motion_ratio
        ]
        
        # Normalize features to avoid scale issues
        max_vals = [max(f1, f2) if max(f1, f2) > 0 else 1.0 
                   for f1, f2 in zip(features_self, features_other)]
        
        norm_self = [f / m for f, m in zip(features_self, max_vals)]
        norm_other = [f / m for f, m in zip(features_other, max_vals)]
        
        # Calculate similarity as 1 - normalized distance
        distance = sum((f1 - f2) ** 2 for f1, f2 in zip(norm_self, norm_other)) ** 0.5
        similarity = 1.0 - (distance / len(features_self))
        
        return similarity >= feature_similarity_threshold