import math
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
        Compute velocity magnitude between the given frame and the previous frame.

        Args:
            frame_idx (int): Frame number.
        Returns:
            Float: velocity magnitude, or 0 if not computable.
        """ 
        if frame_idx == 0 or frame_idx not in self.frames or (frame_idx - 1) not in self.frames:
            return 0.0
        x,y = self.get_normalized_coordinates(frame_idx)
        x0,y0 = self.get_normalized_coordinates(frame_idx-1)

        dx = x - x0
        dy = y - y0
        return math.hypot(dx, dy)
    def get_velocity_vector(self, frame_idx):
        """
        Returns (dx, dy) between current and previous frame.
        If previous frame not available, returns (0.0, 0.0)
        """
        if frame_idx == 0 or frame_idx not in self.frames or (frame_idx - 1) not in self.frames:
            return (0.0, 0.0)
        x,y = self.get_normalized_coordinates(frame_idx)
        x0,y0 = self.get_normalized_coordinates(frame_idx-1)

        dx = x - x0
        dy = y - y0
        return (dx, dy)
    def get_velocity_magnitude(self, frame_idx):
        """
        Retrieve precomputed velocity magnitude for a specific frame index.

        Args:
            frame_idx (int): Frame number.
        Returns:
            Float: velocity magnitude, or 0 if not available.
        """
        if frame_idx < len(self.velocities):
            return self.velocities[frame_idx]
        return 0.0
    def get_acceleration_magnitude(self, frame_idx):
        """
        Retrieve precomputed acceleration magnitude for a specific frame index.

        Args:
            frame_idx (int): Frame number.
        Returns:
            Float: acceleration magnitude, or 0 if not available.   
        """
        if frame_idx < len(self.accelerations):
            return self.accelerations[frame_idx]
        return 0.0
    def compute_acceleration_magnitude(self, frame_idx):
        """
        Compute acceleration magnitude between the given frame and the previous two frames.

        Args:
            frame_idx (int): Frame number.
        Returns:
            Float: acceleration magnitude, or 0 if not computable.
        """
        if frame_idx < 2 or frame_idx not in self.frames or (frame_idx - 1) not in self.frames or (frame_idx - 2) not in self.frames:
            return 0.0
        displacement_current = self.get_velocity_magnitude(frame_idx)
        displacement_previous = self.get_velocity_magnitude(frame_idx - 1)
    
        # Acceleration is the change in displacement
        return abs(displacement_current - displacement_previous)
    def build_velocities_and_accelerations(self):
        """
        Build velocity and acceleration lists for the frames of this body part.
        """
        # THis should use the built in variable. THe no of frames should be saved as soon as all the frames have been added
        num_frames = len(self.frames)
        self.velocities = [self.compute_velocity_magnitude(i) for i in range(num_frames)]

        self.accelerations = [self.compute_acceleration_magnitude(i) for i in range(num_frames)]  
    def update_normalized(self):
        person = self.gesture_analysis.get_person_by_id(self.person_id)
        for frame in self.frames.values():
            frame.update_normalized(
                person.avg_origin_x,
                person.avg_origin_y,
                person.avg_shoulder_length
            )
    def confident(self, frame_idx):
        return True
        return self.frames[frame_idx].is_valid(0.3)
    def add_keyframe(self, frame):
        """
        Add or update a frame observation.

        Args:
            frame (Frame): Frame object.
        """
        # store by frame_no for O(1) lookup
        self.frames[frame.frame_no] = frame

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