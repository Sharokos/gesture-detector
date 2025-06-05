class GesturePart:
    def __init__(self, part_name):
        """
        Initialize a GesturePart to track a single keypoint across frames.

        Args:
            part_name (str): Name of the keypoint (e.g., "RWrist", "Thumb_4").
        """
        self.part_name = part_name
        self.frames = []  # List of frame indices

    def add_keyframe(self, frame):
        """
        Add a new frame observation.

        Args:
            frame (Frame): Frame object.
        """
        self.frames.append(frame)

    def get_motion_deltas(self):
        """
        Compute Euclidean distance (motion) between consecutive frames.

        Returns:
            List of float: motion magnitudes between consecutive frames.
        """
        import math
        deltas = []
        for i in range(1, len(self.positions)):
            x0, y0 = self.positions[i - 1]
            x1, y1 = self.positions[i]
            dx = x1 - x0
            dy = y1 - y0
            deltas.append(math.hypot(dx, dy))
        return deltas


    def __len__(self):
        return len(self.frames)

    def __repr__(self):
        return (f"GesturePart('{self.part_name}', frames={len(self.frames)}")

    def display_frames(self):
        """
        Print all frames for this gesture part.
        """
        for frame in self.frames:
            print(frame)