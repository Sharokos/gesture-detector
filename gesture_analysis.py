
import json
import csv
from data_model.person import PersonGesture
from sliding_window import SlidingWindow
import pandas as pd
import os
class GestureAnalysis:
    # COCO body parts minimal
    COCO_PARTS = [
        "Neck", "RShoulder", "RElbow", "RWrist", 
        "LShoulder", "LElbow", "LWrist", "MidHip", "RHip","LHip"
    ]
    HAND_PARTS = [
        "Wrist",
        "Thumb_1", "Thumb_2", "Thumb_3", "Thumb_4",
        "Index_1", "Index_2", "Index_3", "Index_4",
        "Middle_1", "Middle_2", "Middle_3", "Middle_4",
        "Ring_1", "Ring_2", "Ring_3", "Ring_4",
        "Pinky_1", "Pinky_2", "Pinky_3", "Pinky_4"
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
            print(f"Building data for Person {person.person_id}")
            person.build_all_data()

    

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
        if output_file is None:
            output_file = f"person_{person_id}_windows.csv"

        # Get all windows for this person
        person_windows = [w for w in self.sliding_windows if w.person.person_id == person_id]

        if not person_windows:
            print(f"No windows found for person {person_id}")
            return None

        # Prepare data as a list of dicts
        rows = []
        for window in person_windows:
            # Build features if not already done
            if not hasattr(window, 'motion_energy'):
                window.build_features()

            rows.append({
                'window_id': window.id,
                'start_frame': window.start_frame,
                'end_frame': window.end_frame,
                'contains_gesture': 'Yes' if window.contains_gesture() else 'No',
                'motion_energy': window.motion_energy,
                'rh_energy': window.rh_energy,
                'lh_energy': window.lh_energy,
                'mean_velocity': window.mean_velocity,
                'motion_persistence': window.motion_persistence,
                'velocity_variance': window.velocity_variance,
                'distal_proximal_motion_ratio': window.distal_proximal_motion_ratio,
                'directional_consistency': window.directional_consistency,
                'max_angular_velocity': window.max_angular_velocity,
                'max_angle': window.max_angle,
                'acc': window.max_acceleration,
                'score': window.score,
                # 'le': window.angle_le,
                # 're': window.angle_re,
                # 'ls': window.angle_ls,
                # 'rs': window.angle_rs,
                # 'vle': window.vel_le,
                # 'vre': window.vel_re,
                # 'vls': window.vel_ls,
                # 'vrs': window.vel_rs,
            })

        # Convert to pandas DataFrame
        df = pd.DataFrame(rows)

        # Optional: round numeric columns for readability
        numeric_cols = [
            'motion_energy','rh_energy','lh_energy', 'mean_velocity', 'motion_persistence',
            'velocity_variance', 'distal_proximal_motion_ratio', 'directional_consistency','max_angle','max_angular_velocity','score','acc'
        ]
        df[numeric_cols] = df[numeric_cols].round(4)

        # Export to CSV
        df.to_csv(output_file, index=False)

        print(f"\nExported {len(df)} windows to: {output_file}")
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
        motion_persistence = [w.motion_persistence for w in person_windows]
        velocity_variance = [w.velocity_variance for w in person_windows]
        distal_proximal = [w.distal_proximal_motion_ratio for w in person_windows]
        directional_consistency = [w.directional_consistency for w in person_windows]
        max_angle = [w.max_angle for w in person_windows]
        max_angular_veloctiy = [w.max_angular_velocity for w in person_windows]
        score = [w.score for w in person_windows]
        acc = [w.max_acceleration for w in person_windows]
        r_hand_energy = [w.rh_energy for w in person_windows]
        l_hand_energy = [w.lh_energy for w in person_windows]
        
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
            y=acc,
            mode='lines+markers',
            name='Max acceleration',
            line=dict(color="#733f96", width=2),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=start_frames,
            y=r_hand_energy,
            mode='lines+markers',
            name='Right hand energy',
            line=dict(color="#00ff55", width=2),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=start_frames,
            y=l_hand_energy,
            mode='lines+markers',
            name='Left hand energy',
            line=dict(color="#ff7dff", width=2),
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
            y=motion_persistence,
            mode='lines+markers',
            name='Motion persistence Ratio',
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
        fig.add_trace(go.Scatter(
            x=start_frames,
            y=directional_consistency,
            mode='lines+markers',
            name='Directional consistency',
            line=dict(color="#eeff00", width=2),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=start_frames,
            y=max_angle,
            mode='lines+markers',
            name='Max angle',
            line=dict(color="#00FFDD", width=2),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=start_frames,
            y=max_angular_veloctiy,
            mode='lines+markers',
            name='Max angular velocity',
            line=dict(color="#df4ee4", width=2),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=start_frames,
            y=score,
            mode='lines+markers',
            name='Score',
            line=dict(color="#1100ff", width=2),
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
    

    def export_debug_data_for_person(self, person_id):
        person = self.get_person_by_id(person_id)
        output_path = "debug_person"
        rows = []

        for part_name, body_part in person.body.items():
            if part_name not in SlidingWindow.INTEREST_PARTS:
                continue

            for frame_idx, frame in body_part.frames.items():
                rows.append({
                    "person_id": person_id,
                    "body_part": part_name,
                    "frame_idx": frame_idx,
                    "X": frame.x,
                    "Y": frame.y,
                    "X_normalized": frame.x_normalized,
                    "Y_normalized": frame.y_normalized,
                    "confidence": frame.confidence,
                })
        for part_name, body_part in person.left_hand.items():
            for frame_idx, frame in body_part.frames.items():
                rows.append({
                    "person_id": person_id,
                    "body_part": part_name,
                    "frame_idx": frame_idx,
                    "X": frame.x,
                    "Y": frame.y,
                    "X_normalized": frame.x_normalized,
                    "Y_normalized": frame.y_normalized,
                    "confidence": frame.confidence,
                })
        for part_name, body_part in person.right_hand.items():
            for frame_idx, frame in body_part.frames.items():
                rows.append({
                    "person_id": person_id,
                    "body_part": part_name,
                    "frame_idx": frame_idx,
                    "X": frame.x,
                    "Y": frame.y,
                    "X_normalized": frame.x_normalized,
                    "Y_normalized": frame.y_normalized,
                    "confidence": frame.confidence,
                })
        df = pd.DataFrame(rows)

        # Export reminders:
        df.to_csv(f"{output_path}.csv", index=False)
        # df.to_excel(f"{output_path}.xlsx", index=False)

        return df

                
    def export_person_bodyparts_to_csv(self,person, output_dir="bodypart_csvs"):

        os.makedirs(output_dir, exist_ok=True)
        csv_files = []

        for part_name, body_part in person.body.items():
            rows = []
            for frame_idx, frame in body_part.frames.items():
                vx, vy = body_part.get_velocity_vector(frame_idx)
                v_mag =  body_part.get_velocity_magnitude(frame_idx)
                rows.append({
                    "frame_idx": frame_idx,
                    "x": frame.x,
                    "y": frame.y,
                    "x_normalized": frame.x_normalized,
                    "y_normalized": frame.y_normalized,
                    "vx": vx,
                    "vy": vy,
                    "velocity_magnitude": v_mag,
                    "confidence": frame.confidence
                })

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Optional: sort by frame_idx to ensure order
            df = df.sort_values("frame_idx").reset_index(drop=True)

            # Save CSV
            csv_file = os.path.join(output_dir, f"{person.person_id}_{part_name}.csv")
            df.to_csv(csv_file, index=False)
            csv_files.append(csv_file)

            print(f"Exported {len(df)} frames for {part_name} -> {csv_file}")

        return csv_files