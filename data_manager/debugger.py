import pandas as pd
import os
import re
from collections import defaultdict
from pathlib import Path
from config import DEBUG_DIR,BASELINE_WINDOW
FEATURE_COLUMNS = [
    "motion_energy",
    "mean_velocity",
    "motion_persistence",
    "velocity_variance",
    "max_acceleration",
]

MANAGER_COLUMNS = [
    "velocity",
    "velocity_variance",
    "max_energy",
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
    "mean_baseline_distance",
]

def _safe_sheet_name(name):
        # Excel sheet names must be <=31 chars and cannot contain: : \/ ? * [ ]
        name = re.sub(r'[:\\\/?\*\[\]]', '_', name)
        return name[:31]
def export_person_features_data(gesture_analysis, person_id, deep_debug):

    output_dir = DEBUG_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    person_windows = gesture_analysis.get_windows_for_person(person_id)

    body_data = defaultdict(list)
    left_hand_data = defaultdict(list)
    right_hand_data = defaultdict(list)
    manager_rows = []

    # ----------------------------
    # Parse sliding windows
    # ----------------------------
    for window in person_windows:
        fm = window.features_manager

        base_info = {
            "window_id": window.id,
            "start_frame": window.start_frame,
            "end_frame": window.end_frame,
            "duration": window.duration_seconds,
        }
        if deep_debug:
            # ---- BODY PARTS ----
            for part, feat in fm.body_features.items():
                row = base_info.copy()
                for col in FEATURE_COLUMNS:
                    row[col] = getattr(feat, col, None)
                body_data[part].append(row)

            # ---- LEFT HAND ----
            for part, feat in fm.left_hand_features.items():
                row = base_info.copy()
                for col in FEATURE_COLUMNS:
                    row[col] = getattr(feat, col, None)
                left_hand_data[part].append(row)

            # ---- RIGHT HAND ----
            for part, feat in fm.right_hand_features.items():
                row = base_info.copy()
                for col in FEATURE_COLUMNS:
                    row[col] = getattr(feat, col, None)
                right_hand_data[part].append(row)

        # ---- FEATURES MANAGER OVERVIEW ----
        mgr_row = base_info.copy()
        for col in MANAGER_COLUMNS:
            mgr_row[col] = getattr(fm, col, None)
        manager_rows.append(mgr_row)

    
    if deep_debug:
        # ----------------------------
        # Write BODY excel
        # ----------------------------
        with pd.ExcelWriter(output_dir / "body.xlsx", engine="openpyxl") as writer:
            for part, rows in body_data.items():
                pd.DataFrame(rows).to_excel(
                    writer, sheet_name=_safe_sheet_name(part), index=False
                )

        # ----------------------------
        # Write LEFT HAND excel
        # ----------------------------
        with pd.ExcelWriter(output_dir / "left_hand.xlsx", engine="openpyxl") as writer:
            overview_rows = []

            for part, rows in left_hand_data.items():
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=_safe_sheet_name(part), index=False)

                overview = df[FEATURE_COLUMNS].mean().to_dict()
                overview.update({
                    "window_id": "mean",
                    "start_frame": "",
                    "part": part
                })
                overview_rows.append(overview)

            if overview_rows:
                pd.DataFrame(overview_rows).to_excel(
                    writer, sheet_name="overview", index=False
                )

        # ----------------------------
        # Write RIGHT HAND excel
        # ----------------------------
        with pd.ExcelWriter(output_dir / "right_hand.xlsx", engine="openpyxl") as writer:
            overview_rows = []

            for part, rows in right_hand_data.items():
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=_safe_sheet_name(part), index=False)

                overview = df[FEATURE_COLUMNS].mean().to_dict()
                overview.update({
                    "window_id": "mean",
                    "start_frame": "",
                    "part": part
                })
                overview_rows.append(overview)

        if overview_rows:
            pd.DataFrame(overview_rows).to_excel(
                writer, sheet_name="overview", index=False
            )

    # ----------------------------
    # Write FEATURES MANAGER overview
    # ----------------------------
    pd.DataFrame(manager_rows).to_excel(
        output_dir / "overview.xlsx",
        index=False
    )
 
                
def export_person_bodyparts_data(gesture_analysis, person_id, deep_debug, output_dir="bodypart_csvs", ):
    output_dir = DEBUG_DIR  + "/" + output_dir
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    person = gesture_analysis.get_person_by_id(person_id)
    if person is None: 
        print("No person found in export function")
        return
    categories = {
        'body': person.body
    }
    if deep_debug:
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
                    # if hasattr(body_part, 'build_velocities_and_accelerations'):
                    #     body_part.build_velocities_and_accelerations()
                    rows = []

                    for frame_idx, frame in body_part.frames.items():

                        vx, vy = body_part.get_velocity_vector(frame_idx)
                        v_mag = body_part.get_velocity_magnitude(frame_idx)
                        # block_idx = frame_idx // BASELINE_WINDOW
                        bx, by = body_part.baselines.get(frame_idx, (None, None))
                        if bx is None:
                            bx = 0
                        if by is None:
                            by = 0
                        x_normalized, y_normalized = body_part.get_normalized_coordinates(frame_idx)
                        if x_normalized is None:
                            x_normalized = 0
                        if y_normalized is None:
                            y_normalized = 0
                        rows.append({
                            "frame_idx": frame_idx,
                            "x": frame.x,
                            "y": frame.y,
                            "x_normalized": x_normalized,
                            "y_normalized": y_normalized,
                            "vx": vx,
                            "vy": vy,
                            "velocity_magnitude": v_mag,
                            "baseline_x": bx,
                            "baseline_y": by,
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