class FrameNormalization:

    def __init__(self, frame_no, x, y, shoulder_length):
        """
        Initialize the normalization object - per frame

        Args:
            frame_no (int): Number of the frame.
        """
        self.frame_no = frame_no
        self.x_origin = x
        self.y_origin = y
        self.shoulder_length = shoulder_length
        
    def __repr__(self):
        return (f"Frame: {self.frame_no}, Position: ({self.x_origin:.2f}, {self.y_origin:.2f}), shoulder length: {self.shoulder_length:.2f}")
    