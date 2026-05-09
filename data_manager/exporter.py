import json

def export_gesture_groups_to_json(gesture_analysis, gesture_groups, output_file=None):
        """
        Export gesture groups to JSON format compatible with the video helper.
        
        Args:
            gesture_groups (list): List of gesture groups from merge_gesture_windows().
            output_file (str): Path to output JSON file. If None, uses 'gestures.json'.
        
        Returns:
            str: JSON contents
        """
        if output_file is None:
            output_file = "gestures.json"
        
        gestures = []
        # TODO: better to calculate the start/end time for each window at creation time so you won't pass the gesture analysis obj here
        for gesture_id, group in enumerate(gesture_groups):

            start_frame = (group[0].start_frame + group[0].end_frame)//2
            
            # start_frame = group[0].start_frame
            end_frame = group[-1].end_frame
            
            # Calculate timestamps (in seconds)
            start_time = round(start_frame / gesture_analysis.frame_rate, 3)
            end_time = round(end_frame / gesture_analysis.frame_rate, 3)
            
            gestures.append({
                "gesture": f"gesture_{gesture_id}",
                "start": start_time,
                "end": end_time
            })
        
        # Write to JSON
        with open(output_file, 'w') as jsonfile:
            json.dump(gestures, jsonfile, indent=2)
        
        print(f"\nExported {len(gestures)} gestures to: {output_file}")
        return gestures