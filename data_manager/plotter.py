def plot_person_sliding_windows(gesture_analysis, person_id, output_html=None):
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not installed. Install it with: pip install plotly")
            return None
        
        if output_html is None:
            output_html = f"person_{person_id}_plot.html"
        
        # Get all windows for this person
        person_windows = [w for w in gesture_analysis.sliding_windows if w.person.person_id == person_id]
        
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