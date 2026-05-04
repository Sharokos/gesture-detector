from gesture_manager.sliding_window import SlidingWindow
from data_manager.exporter import export_gesture_groups_to_json
from data_manager import input_parser
import numbers
from math_utility import remove_outliers_mad
from config import TEMPORAL_GAP
import csv
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
        # There will always be only one instance of gesture_analysis per video, so it's safe to store the FPS
        self.frame_rate = 24
        input_parser.parse_openpose_and_populate_persons(gesture_analysis=self)       
        print(f"Detected {len(self.persons)} persons")
        
        
    def execute(self):
        # Building and computing all data
        self.build_all_data()
        self.create_sliding_windows()

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
                window = SlidingWindow(sliding_window_id, start, end, person, self.frame_rate)
                self.sliding_windows.append(window)
        print(f"Created {len(self.sliding_windows)} sliding windows")
    
    def build_all_data(self):
        """
        Build all necessary data for all persons.
        """
        for person in self.persons.values():
            print(f"Building data for Person {person.person_id}")
            person.build_all_data()
    
    def determine_gestures_for_person(self,person_id,output_path = None):
        self.clean_features_outliers_for_person(person_id)
        # Keep only the sliding windows of that person
        person_windows = self.get_windows_for_person(person_id)
        for w in person_windows:
            w.recompute_score()
        # Keep only sliding windows marked as containing gestures
        gesture_windows = [w for w in person_windows if w.contains_gesture()]
        print(f"\nPerson {person_id}: Detected {len(gesture_windows)} gesture windows")
        gesture_groups = self.merge_gesture_windows(
            gesture_windows,
            max_temporal_gap=TEMPORAL_GAP,
        )
        self.print_gesture_summary(gesture_groups)
        # Export to JSON for video helper
        export_gesture_groups_to_json(self,gesture_groups, os.path.join(output_path , "gestures.json"))

    def get_windows_for_person(self,person_id):
        person_windows = []
        person = self.get_person_by_id(person_id)
        if person:
            person_windows = [w for w in self.sliding_windows if w.person.person_id == person_id]
        return person_windows
    
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
                              ):
        if not gesture_windows:
            return []
        
        # Sort windows by start frame
        sorted_windows = sorted(gesture_windows, key=lambda w: w.start_frame)
        
        gesture_groups = []
        current_group = [sorted_windows[0]]
        
        for window in sorted_windows[1:]:
            # Check if this window should merge with the last window in current group
            if self.are_windows_mergeable(current_group[-1], window, max_temporal_gap):
                current_group.append(window)
            else:
                # Start a new group
                gesture_groups.append(current_group)
                current_group = [window]
        
        # Don't forget the last group
        if current_group:
            gesture_groups.append(current_group)
        
        return gesture_groups
    
    def are_windows_mergeable(self, first_window, second_window, 
                         max_temporal_gap):

        # Check temporal proximity
        temporal_gap = min(
            abs(first_window.start_frame - second_window.end_frame),
            abs(second_window.start_frame - first_window.end_frame)
        )

        if temporal_gap > max_temporal_gap:
            return False
        return True
    
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
    
    def clean_features_outliers_for_person(self, person_id, k=3.5):
        """
        Applies MAD-based outlier clipping to all numeric FeaturesManager
        parameters across a list of sliding windows.

        Parameters
        ----------
        sliding_windows : list
            List of SlidingWindow instances.
        k : float
            MAD threshold.

        Returns
        -------
        None (FeaturesManagers are modified in-place)
        """
        sliding_windows = self.get_windows_for_person(person_id)
        if not sliding_windows:
            return

        # Collect all numeric FeaturesManager attributes
        fm_attrs = set()

        for w in sliding_windows:
            fm = w.features_manager
            for attr, value in vars(fm).items():
                if isinstance(value, numbers.Number):
                    fm_attrs.add(attr)

        # Apply MAD clipping per attribute
        for attr in fm_attrs:
            values = []
            windows_with_attr = []

            for w in sliding_windows:
                fm = w.features_manager
                if hasattr(fm, attr):
                    v = getattr(fm, attr)
                    if isinstance(v, numbers.Number):
                        values.append(v)
                        windows_with_attr.append(fm)

            if len(values) < 3:
                continue

            cleaned_values = remove_outliers_mad(values, k=k)

            # Write back clipped values
            for fm, v_clean in zip(windows_with_attr, cleaned_values):
                setattr(fm, attr, v_clean)


    def export_windows_to_csv(self, output_path):
        if not self.sliding_windows:
            return

        fm_fields = [
            "velocity",
            "velocity_variance",
            "max_energy",
            "motion_persistance",
            'max_angular',
            "left_hand_energy",
            "right_hand_energy",
            "l_elbow_angle",
            "r_elbow_angle",
            "l_shoulder_angle",
            "r_shoulder_angle",
            "l_elbow_angular_velocity",
            "r_elbow_angular_velocity",
            "l_shoulder_angular_velocity",
            "r_shoulder_angular_velocity",
            "distal_proximal_ratio",
            "max_acceleration",
            "mean_baseline_distance"
        ]

        fieldnames = [
            "window_id",
            "person_id",
            "start_frame",
            "end_frame",
            "duration_seconds",
            "label"
        ] + fm_fields

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for w in self.sliding_windows:
                fm = w.features_manager

                row = {
                    "window_id": w.id,
                    "person_id": w.person.person_id,
                    "start_frame": w.start_frame,
                    "end_frame": w.end_frame,
                    "duration_seconds": w.duration_seconds,
                    "label": getattr(w, "label", None)
                }

                for field in fm_fields:
                    row[field] = getattr(fm, field, None)

                writer.writerow(row)
