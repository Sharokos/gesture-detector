import plotly.graph_objects as go
import os
import itertools

def plot_person_sliding_windows(gesture_analysis, person_id, output_dir=None):

    output_html = os.path.join(output_dir,f"person_{person_id}_plot.html")

    # Get all windows for this person
    person_windows = gesture_analysis.get_windows_for_person(person_id)

    if not person_windows:
        print(f"No windows found for person {person_id}")
        return None

    # Sort windows by start frame
    person_windows.sort(key=lambda w: w.start_frame)

    # -------------------------
    # FeaturesManager parameters to plot
    # -------------------------
    fm_params = [
        "velocity",
        "velocity_variance",
        "max_energy",
        "motion_persistance",
        'max_angular',
        # "left_hand_energy",
        # "right_hand_energy",
        # "l_elbow_angle",
        # "r_elbow_angle",
        # "l_shoulder_angle",
        # "r_shoulder_angle",
        # "l_elbow_angular_velocity",
        # "r_elbow_angular_velocity",
        # "l_shoulder_angular_velocity",
        # "r_shoulder_angular_velocity",
        "distal_proximal_ratio",
        "max_acceleration",
        "mean_baseline_distance"
    ]

    # -------------------------
    # Collect data
    # -------------------------
    data = {param: [] for param in fm_params}
    data["window_id"] = []
    data["start_frame"] = []
    data["score"] = []

    for w in person_windows:
        fm = w.features_manager
        data["window_id"].append(w.id)
        data["start_frame"].append(w.start_frame)
        data["score"].append(w.score)
        for param in fm_params:
            data[param].append(getattr(fm, param, None))

    # -------------------------
    # Create figure
    # -------------------------
    import itertools
    import plotly.graph_objects as go

    fig = go.Figure()
    color_cycle = itertools.cycle([
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#ff00ff', '#00ff55', '#1100ff', '#733f96'
    ])

    for param in fm_params:
        fig.add_trace(go.Scatter(
            x=data["start_frame"],
            y=data[param],
            mode='lines+markers',
            name=param,
            line=dict(color=next(color_cycle), width=2),
            marker=dict(size=6)
        ))
    fig.add_trace(go.Scatter(
            x=data["start_frame"],
            y=data['score'],
            mode='lines+markers',
            name='Score',
            line=dict(color=next(color_cycle), width=2),
            marker=dict(size=6)
        ))
    # -------------------------
    # Layout
    # -------------------------
    fig.update_layout(
        title=f'FeaturesManager Parameters - Person {person_id}',
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

    # -------------------------
    # Save HTML
    # -------------------------
    fig.write_html(output_html)
    print(f"\nPlot saved to: {output_html}")
    return output_html


def plot_body_part_features(gesture_analysis, person_id,output_dir=None):
    """
    Plots features for all body parts of a person.

    Parameters
    ----------
    person : Person
        Person instance with body_parts dict[part_name -> BodyPart]
    output_dir : str, optional
        If given, saves HTML plots to this directory. Files: <part_name>.html
    """
    person = gesture_analysis.get_person_by_id(person_id)
    if person is None:
        return
    
    output_dir = os.path.join(output_dir,"plots")

    for part_name, body_part in person.body.items():
        frame_indices = sorted(body_part.frames.keys())

        if not frame_indices:
            continue

        # Gather data
        x = []
        y = []
        x_norm = []
        y_norm = []
        conf = []
        vx = []
        vy = []
        bx = []
        by = []

        for idx in frame_indices:
            frame = body_part.frames[idx]

            # Original coordinates
            x.append(frame.x if frame.x is not None else None)
            y.append(frame.y if frame.y is not None else None)

            # Normalized coordinates
            x_norm.append(getattr(frame, "x_normalized", None))
            y_norm.append(getattr(frame, "y_normalized", None))

            # Confidence
            conf.append(getattr(frame, "confidence", None))

            # Velocities
            vx.append(getattr(body_part, "vx", [None]*len(frame_indices))[idx] if hasattr(body_part, "vx") else None)
            vy.append(getattr(body_part, "vy", [None]*len(frame_indices))[idx] if hasattr(body_part, "vy") else None)

            # Baseline
            if hasattr(body_part, "baselines"):
                bx_val, by_val = body_part.baselines.get(idx, (None, None))
            else:
                bx_val, by_val = None, None
            bx.append(bx_val)
            by.append(by_val)

        # Create figure
        fig = go.Figure()

        # Add traces
        fig.add_trace(go.Scatter(x=frame_indices, y=x, mode='lines+markers', name='x'))
        fig.add_trace(go.Scatter(x=frame_indices, y=y, mode='lines+markers', name='y'))
        fig.add_trace(go.Scatter(x=frame_indices, y=x_norm, mode='lines+markers', name='x_normalized'))
        fig.add_trace(go.Scatter(x=frame_indices, y=y_norm, mode='lines+markers', name='y_normalized'))
        fig.add_trace(go.Scatter(x=frame_indices, y=conf, mode='lines+markers', name='confidence'))
        fig.add_trace(go.Scatter(x=frame_indices, y=vx, mode='lines+markers', name='vx'))
        fig.add_trace(go.Scatter(x=frame_indices, y=vy, mode='lines+markers', name='vy'))
        fig.add_trace(go.Scatter(x=frame_indices, y=bx, mode='lines+markers', name='baseline_x'))
        fig.add_trace(go.Scatter(x=frame_indices, y=by, mode='lines+markers', name='baseline_y'))

        # Layout
        fig.update_layout(
            title=f"{person.person_id} - {part_name}",
            xaxis_title="Frame index",
            yaxis_title="Value",
            template="plotly_white",
            width=1200,
            height=600,
            hovermode="x unified"
        )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{person.person_id}_{part_name}.html")
            fig.write_html(filename)


def plot_normalization_data(gesture_analysis, person_id,output_dir=None):

    person = gesture_analysis.get_person_by_id(person_id)
    if person is None:
        return

    # Initialize lists outside the loop
    x = []
    y = []
    shoulder_length = []
    frame_indices = []

    for frame_idx, norm_data in sorted(person.normalization_data.items()):
        frame_indices.append(frame_idx)
        x.append(norm_data.x_origin if norm_data.x_origin is not None else None)
        y.append(norm_data.y_origin if norm_data.y_origin is not None else None)
        shoulder_length.append(norm_data.shoulder_length if norm_data.shoulder_length is not None else None)

    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame_indices, y=x, mode='lines+markers', name='x'))
    fig.add_trace(go.Scatter(x=frame_indices, y=y, mode='lines+markers', name='y'))
    fig.add_trace(go.Scatter(x=frame_indices, y=shoulder_length, mode='lines+markers', name='shoulder_length'))

    # Layout
    fig.update_layout(
        title=f"{person.person_id} - normalization data",
        xaxis_title="Frame index",
        yaxis_title="Value",
        template="plotly_white",
        width=1200,
        height=600,
        hovermode="x unified"
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{person.person_id}_normalization_data.html")
        fig.write_html(filename)
        print(f"Saved normalization plot: {filename}")
    else:
        fig.show()



def plot_person_bodypart_features(
    gesture_analysis,
    person_id,
    output_dir=None,
    ):
    output_dir = os.path.join(output_dir,"plots_features")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Get windows
    # -------------------------
    person_windows = gesture_analysis.get_windows_for_person(person_id)
    if not person_windows:
        print(f"No windows found for person {person_id}")
        return

    person_windows.sort(key=lambda w: w.start_frame)

    fm0 = person_windows[0].features_manager
    interest_parts = fm0.INTEREST_PARTS

    # -------------------------
    # Discover feature fields
    # -------------------------
    sample_part = next(
        p for p in fm0.body_features.values() if p is not None
    )

    feature_fields = [
        k for k, v in vars(sample_part).items()
        if isinstance(v, (int, float))
    ]

    # -------------------------
    # One plot per body part
    # -------------------------
    for part in interest_parts:
        fig = go.Figure()
        color_cycle = itertools.cycle([
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
        ])

        for field in feature_fields:
            x_vals = []
            y_vals = []

            for w in person_windows:
                feat = w.features_manager.body_features.get(part)
                x_vals.append(w.start_frame)
                y_vals.append(
                    getattr(feat, field, None) if feat else None
                )

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=field,
                line=dict(width=2, color=next(color_cycle)),
                marker=dict(size=6)
            ))

        # -------------------------
        # Layout
        # -------------------------
        fig.update_layout(
            title=f"{part} — all features (Person {person_id})",
            xaxis_title="Window start frame",
            yaxis_title="Feature value",
            hovermode="x unified",
            template="plotly_white",
            width=1200,
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            )
        )

        out_path = os.path.join(
            output_dir, f"person_{person_id}_{part}_features.html"
        )
        fig.write_html(out_path)
        print(f"Saved: {out_path}")
