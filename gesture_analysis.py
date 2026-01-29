from pathlib import Path
import json
import csv
from person import PersonGesture
from sliding_window import SlidingWindow

class GestureAnalysis:
    # COCO body parts minimal
    COCO_PARTS = [
        "Neck", "RShoulder", "RElbow", "RWrist", 
        "LShoulder", "LElbow", "LWrist", "MidHip", "RHip","LHip"
    ]
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.persons = {}
        self.sliding_windows = []
        self.number_of_frames = 0
        self.frame_rate = 24  # Default FPS, can be adjusted

    def create_sliding_windows(self):
        """
        Create sliding windows for all persons.
        """
        for person in self.persons.values():
            total_frames = self.number_of_frames
            window_size = SlidingWindow.WINDOW_SIZE
            step_size = SlidingWindow.STEP_SIZE

            for start in range(0, total_frames - window_size + 1, step_size):
                end = start + window_size - 1
                sliding_window_id = len(self.sliding_windows)
                window = SlidingWindow(sliding_window_id, start, end, person)
                self.sliding_windows.append(window)
    def build_all_data(self):
        """
        Build all necessary data for all persons.
        """
        for person in self.persons.values():
            person.build_all_data()

    def parse_openpose_and_populate_persons(self) -> dict:
        """
        Parses OpenPose JSONs and builds a dictionary of PersonGesture instances.

        Returns:
            Dict[int, PersonGesture]: Mapping from person_id to their tracked gesture data.
        """
        folder_path = Path(self.input_folder)
        json_files = sorted(folder_path.glob("*.json"))

        # Each file represents a frame, and each person in the frame has keypoints
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            frame_index = int(json_file.stem.split("_")[1])  # e.g., '000000000001'
            self.number_of_frames += 1
            # print(f"Processing frame {frame_index} from {json_file.name}")
            for person_id, person in enumerate(data.get("people", [])):
                # Ensure person object exists
                if person_id not in self.persons:
                    # print(f"Creating new PersonGesture for person_id {person_id}")
                    self.persons[person_id] = PersonGesture(person_id, gesture_analysis=self)

                # Extract and organize keypoints
                body_data = person.get("pose_keypoints_2d", [])
                # left_hand_data = person.get("hand_left_keypoints_2d", [])
                # right_hand_data = person.get("hand_right_keypoints_2d", [])

                keypoint_data = {
                    "body": {
                        self.COCO_PARTS[i]: (
                            body_data[i * 3],
                            body_data[i * 3 + 1],
                            body_data[i * 3 + 2]
                        )
                        for i in range(len(self.COCO_PARTS))
                    },
                    # "left_hand": {
                    #     HAND_PARTS[i]: (
                    #         left_hand_data[i * 3],
                    #         left_hand_data[i * 3 + 1],
                    #         left_hand_data[i * 3 + 2]
                    #     )
                    #     for i in range(len(HAND_PARTS)) if left_hand_data
                    # },
                    # "right_hand": {
                    #     HAND_PARTS[i]: (
                    #         right_hand_data[i * 3],
                    #         right_hand_data[i * 3 + 1],
                    #         right_hand_data[i * 3 + 2]
                    #     )
                    #     for i in range(len(HAND_PARTS)) if right_hand_data
                    # }
                }

                self.persons[person_id].add_frame_data(frame_index, keypoint_data)

    def get_person_by_id(self,person_id):
        """
        Retrieve a PersonGesture by their ID.

        Args:
            person_id (int): ID of the person to retrieve.

        Returns:
            PersonGesture or None: The PersonGesture object if found, else None.
        """
        return self.persons.get(person_id, None)
    
    def merge_gesture_windows(self, gesture_windows,
                              max_temporal_gap=6,
                              feature_similarity_threshold=0.7):
        """
        Merge adjacent gesture windows that are similar and temporally close.
        
        Args:
            gesture_windows (list): List of SlidingWindow objects that contain gestures.
            max_temporal_gap (int): Maximum frames allowed between windows to merge.
            feature_similarity_threshold (float): Minimum similarity (0-1) of features for merging.
        
        Returns:
            list: List of gesture groups, where each group is a list of merged windows.
        """
        if not gesture_windows:
            return []
        
        # Sort windows by start frame
        sorted_windows = sorted(gesture_windows, key=lambda w: w.start_frame)
        
        gesture_groups = []
        current_group = [sorted_windows[0]]
        
        for window in sorted_windows[1:]:
            # Check if this window should merge with the last window in current group
            if current_group[-1].should_merge_with(window, 
                                                    max_temporal_gap=max_temporal_gap,
                                                    feature_similarity_threshold=feature_similarity_threshold):
                current_group.append(window)
            else:
                # Start a new group
                gesture_groups.append(current_group)
                current_group = [window]
        
        # Don't forget the last group
        if current_group:
            gesture_groups.append(current_group)
        
        return gesture_groups
    
    def format_gesture_summary(self, gesture_groups):
        """
        Format gesture groups into a readable summary with frame numbers and timestamps.
        
        Args:
            gesture_groups (list): List of gesture groups from merge_gesture_windows().
        
        Returns:
            list: List of dicts with gesture information including start/end frames and timestamps.
        """
        summary = []
        
        for gesture_id, group in enumerate(gesture_groups):
            start_frame = group[0].start_frame
            end_frame = group[-1].end_frame
            
            # Calculate timestamps (in seconds)
            start_time = start_frame / self.frame_rate
            end_time = end_frame / self.frame_rate
            duration = end_time - start_time
            
            gesture_info = {
                'gesture_id': gesture_id,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': f"{start_time:.2f}s",
                'end_time': f"{end_time:.2f}s",
                'duration': f"{duration:.2f}s",
                'num_windows': len(group),
                'person_id': group[0].person.person_id if hasattr(group[0].person, 'person_id') else 'unknown'
            }
            summary.append(gesture_info)
        
        return summary
    
    def print_gesture_summary(self, gesture_groups):
        """
        Print a formatted gesture summary to the console.
        
        Args:
            gesture_groups (list): List of gesture groups from merge_gesture_windows().
        """
        summary = self.format_gesture_summary(gesture_groups)
        
        if not summary:
            print("No gestures detected.")
            return
        
        print(f"\n{'='*80}")
        print(f"GESTURE SUMMARY ({len(summary)} gestures detected)")
        print(f"{'='*80}")
        
        for gesture in summary:
            print(f"\nGesture {gesture['gesture_id']} (Person {gesture['person_id']}):")
            print(f"  Frames:   {gesture['start_frame']} - {gesture['end_frame']}")
            print(f"  Time:     {gesture['start_time']} - {gesture['end_time']}")
            print(f"  Duration: {gesture['duration']}")
            print(f"  Windows:  {gesture['num_windows']} merged windows")
        
        print(f"\n{'='*80}\n")
    
    def export_person_windows_to_csv(self, person_id, output_file=None):
        """
        Export all sliding windows for a person to CSV with gesture classification.
        
        Creates a CSV file with columns: window_id, start_frame, end_frame, contains_gesture,
        motion_energy, mean_velocity, motion_persistance_ratio, velocity_variance, 
        distal_proximal_motion_ratio
        
        Args:
            person_id (int): ID of the person to export.
            output_file (str): Path to output CSV file. If None, uses 'person_{person_id}_windows.csv'.
            motion_energy_threshold (float): Threshold for motion energy in gesture detection.
            mean_velocity_threshold (float): Threshold for mean velocity in gesture detection.
            motion_persistance_threshold (float): Threshold for motion persistance ratio.
            distal_proximal_threshold (float): Threshold for distal/proximal motion ratio.
        
        Returns:
            str: Path to the created CSV file.
        """
        if output_file is None:
            output_file = f"person_{person_id}_windows.csv"
        
        # Get all windows for this person
        person_windows = [w for w in self.sliding_windows if w.person.person_id == person_id]
        
        if not person_windows:
            print(f"No windows found for person {person_id}")
            return None
        
        # Prepare data
        rows = []
        for window in person_windows:
            # Build features if not already built
            if not hasattr(window, 'motion_energy'):
                window.build_features()
            
            # Determine if window contains gesture
            is_gesture = window.contains_gesture()
            
            
            rows.append({
                'window_id': window.id,
                'start_frame': window.start_frame,
                'end_frame': window.end_frame,
                'contains_gesture': 'Yes' if is_gesture else 'No',
                'motion_energy': f"{window.motion_energy:.4f}",
                'mean_velocity': f"{window.mean_velocity:.4f}",
                'motion_persistance_ratio': f"{window.motion_persistance_ratio:.4f}",
                'velocity_variance': f"{window.velocity_variance:.4f}",
                'distal_proximal_motion_ratio': f"{window.distal_proximal_motion_ratio:.4f}"
            })
        
        # Write to CSV
        csv_columns = [
            'window_id', 'start_frame', 'end_frame', 'contains_gesture',
            'motion_energy', 'mean_velocity', 'motion_persistance_ratio',
            'velocity_variance', 'distal_proximal_motion_ratio'
        ]
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\nExported {len(rows)} windows to: {output_file}")
        return output_file
    
    def export_gesture_groups_to_json(self, gesture_groups, output_file=None):
        """
        Export gesture groups to JSON format compatible with the video helper.
        
        Args:
            gesture_groups (list): List of gesture groups from merge_gesture_windows().
            output_file (str): Path to output JSON file. If None, uses 'gestures.json'.
        
        Returns:
            str: Path to the created JSON file.
        """
        if output_file is None:
            output_file = "gestures.json"
        
        gestures = []
        for gesture_id, group in enumerate(gesture_groups):
            start_frame = group[0].start_frame
            end_frame = group[-1].end_frame
            
            # Calculate timestamps (in seconds)
            start_time = round(start_frame / self.frame_rate, 3)
            end_time = round(end_frame / self.frame_rate, 3)
            
            gestures.append({
                "gesture": f"gesture_{gesture_id}",
                "start": start_time,
                "end": end_time
            })
        
        # Write to JSON
        with open(output_file, 'w') as jsonfile:
            json.dump(gestures, jsonfile, indent=2)
        
        print(f"\nExported {len(gestures)} gestures to: {output_file}")
        return output_file
    
    def plot_person_sliding_windows(self, person_id, output_html=None):
        """
        Create an interactive Plotly plot with all parameters of a person's sliding windows.
        
        Creates a multi-line plot with all sliding window parameters on the same graph.
        X-axis represents the start frame of each window.
        Y-axis contains normalized parameter values.
        
        Parameters plotted:
        - Motion Energy
        - Mean Velocity
        - Motion Persistance Ratio
        - Velocity Variance
        - Distal Proximal Motion Ratio
        
        Args:
            person_id (int): ID of the person to plot.
            output_html (str): Path to output HTML file. If None, uses 'person_{person_id}_plot.html'.
        
        Returns:
            str: Path to the created HTML file.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not installed. Install it with: pip install plotly")
            return None
        
        if output_html is None:
            output_html = f"person_{person_id}_plot.html"
        
        # Get all windows for this person
        person_windows = [w for w in self.sliding_windows if w.person.person_id == person_id]
        
        if not person_windows:
            print(f"No windows found for person {person_id}")
            return None
        
        # Sort windows by start frame
        person_windows = sorted(person_windows, key=lambda w: w.start_frame)
        
        # Build features for all windows
        for window in person_windows:
            if not hasattr(window, 'motion_energy'):
                window.build_features()
        
        # Extract data
        start_frames = [w.start_frame for w in person_windows]
        motion_energy = [w.motion_energy for w in person_windows]
        mean_velocity = [w.mean_velocity for w in person_windows]
        motion_persistance = [w.motion_persistance_ratio for w in person_windows]
        velocity_variance = [w.velocity_variance for w in person_windows]
        distal_proximal = [w.distal_proximal_motion_ratio for w in person_windows]
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each parameter
        fig.add_trace(go.Scatter(
            x=start_frames,
            y=motion_energy,
            mode='lines+markers',
            name='Motion Energy',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=start_frames,
            y=mean_velocity,
            mode='lines+markers',
            name='Mean Velocity',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=start_frames,
            y=motion_persistance,
            mode='lines+markers',
            name='Motion Persistance Ratio',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=start_frames,
            y=velocity_variance,
            mode='lines+markers',
            name='Velocity Variance',
            line=dict(color='#d62728', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=start_frames,
            y=distal_proximal,
            mode='lines+markers',
            name='Distal/Proximal Motion Ratio',
            line=dict(color='#9467bd', width=2),
            marker=dict(size=6)
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Sliding Window Parameters - Person {person_id}',
            xaxis_title='Start Frame',
            yaxis_title='Parameter Value',
            hovermode='x unified',
            template='plotly_white',
            width=1200,
            height=600,
            font=dict(size=12),
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            )
        )
        
        # Save to HTML
        fig.write_html(output_html)
        print(f"\nPlot saved to: {output_html}")
        print(f"Number of windows: {len(person_windows)}")
        
        return output_html