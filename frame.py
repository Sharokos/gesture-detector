class Frame:
    FRAME_RATE = 24
    def __init__(self, frame_no,x, y, confidence):
        """
        Initialize a Frame to track a single keypoint frame.

        Args:
            frame_no (int): Number of the frame.
        """
        self.frame_no = frame_no
        self.x = x
        self.y = y
        self.confidence = confidence
        self.x_normalized = 0
        self.y_normalized = 0

    def __repr__(self):
        return (f"Frame: {self.frame_no}, Position: ({self.x:.2f}, {self.y:.2f}), Confidence: {self.confidence:.2f}, timestamp: {self.get_timestamp()}")
    
    def update_normalized(self, x_origin, y_origin, shoulder_length):
        """
        Update normalized coordinates for this frame.

        Args:
            x_origin (float): x coordinate of the origin.
            y_origin (float): y coordinate of the origin.
            shoulder_length (float): shoulder length for normalization.
        """
        self.x_normalized = (self.x - x_origin) / shoulder_length if shoulder_length != 0 else 0
        self.y_normalized = (self.y - y_origin) / shoulder_length if shoulder_length != 0 else 0
    def get_timestamp(self):
        """
        Returns the timestamp for this frame in HH:MM:SS.mmm format.
        """
        total_seconds = self.frame_no / self.FRAME_RATE
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"