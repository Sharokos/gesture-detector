from ..sliding_window import SlidingWindow
import pandas as pd
import os
import re

def export_person_windows_to_csv(gesture_analysis, person_id, output_file=None):
    if output_file is None:
        output_file = f"person_{person_id}_windows.csv"

    # Get all windows for this person
    person_windows = [w for w in gesture_analysis.sliding_windows if w.person.person_id == person_id]

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
                
def export_person_bodyparts_to_csv(gesture_analysis, person, output_dir="bodypart_csvs"):

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    def _safe_sheet_name(name):
        # Excel sheet names must be <=31 chars and cannot contain: : \/ ? * [ ]
        name = re.sub(r'[:\\\/?\*\[\]]', '_', name)
        return name[:31]

    categories = {
        'body': person.body,
        'left_hand': person.left_hand,
        'right_hand': person.right_hand
    }

    for cat_name, parts in categories.items():
        file_path = os.path.join(output_dir, f"{person.person_id}_{cat_name}.xlsx")
        try:
            # Try writing a single Excel workbook with one sheet per body part
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                for part_name, body_part in parts.items():
                    # Ensure velocities/accelerations are available
                    if hasattr(body_part, 'build_velocities_and_accelerations'):
                        body_part.build_velocities_and_accelerations()

                    rows = []
                    for frame_idx, frame in body_part.frames.items():
                        vx, vy = body_part.get_velocity_vector(frame_idx)
                        v_mag = body_part.get_velocity_magnitude(frame_idx)
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

                    df = pd.DataFrame(rows)
                    df = df.sort_values("frame_idx").reset_index(drop=True)

                    sheet_name = _safe_sheet_name(part_name)
                    if df.empty:
                        # create empty sheet with column headers to keep structure
                        df = pd.DataFrame(columns=["frame_idx", "x", "y", "x_normalized", "y_normalized", "vx", "vy", "velocity_magnitude", "confidence"])
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"Exported {cat_name} parts to {file_path}")
            saved_files.append(file_path)

        except Exception as e:
            # Fallback to per-part CSV files if Excel writing fails
            print(f"Failed to write excel {file_path} ({e}). Falling back to per-part CSVs.")
            for part_name, body_part in parts.items():
                csv_file = os.path.join(output_dir, f"{person.person_id}_{part_name}.csv")
                if hasattr(body_part, 'build_velocities_and_accelerations'):
                    body_part.build_velocities_and_accelerations()

                rows = []
                for frame_idx, frame in body_part.frames.items():
                    vx, vy = body_part.get_velocity_vector(frame_idx)
                    v_mag = body_part.get_velocity_magnitude(frame_idx)
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

                df = pd.DataFrame(rows)
                df = df.sort_values("frame_idx").reset_index(drop=True)
                df.to_csv(csv_file, index=False)
                saved_files.append(csv_file)
                print(f"Exported {len(df)} frames for {part_name} -> {csv_file}")

    return saved_files