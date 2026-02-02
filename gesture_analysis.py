
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
    
