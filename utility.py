from pathlib import Path
import json
from person import PersonGesture
import plotly.graph_objs as go



# def get_person_by_id(person_id):
#     """
#     Retrieve a PersonGesture by their ID.

#     Args:
#         person_id (int): ID of the person to retrieve.

#     Returns:
#         PersonGesture or None: The PersonGesture object if found, else None.
#     """
#     return persons.get(person_id, None)
# print(persons[1].body["RWrist"])
# persons[1].body["RWrist"].display_frames()


# def plot_gesture_part_xy_vs_frame(gesture_part):
#     """
#     Plots x and y coordinates of a BodyPart vs frame number using matplotlib.

#     Args:
#         gesture_part (BodyPart): The gesture part to plot.
#     """
#     filtered_frames = [frame for frame in gesture_part.frames if frame.confidence > 0.1]
#     frame_numbers = [frame.frame_no for frame in filtered_frames]
#     x_coords = [frame.x for frame in filtered_frames]
#     y_coords = [frame.y for frame in filtered_frames]

#     plt.figure(figsize=(10, 5))
#     plt.plot(frame_numbers, x_coords, marker='o', label='X')
#     plt.plot(frame_numbers, y_coords, marker='o', label='Y')
#     plt.title(f"{gesture_part.part_name} coordinates vs Frame")
#     plt.xlabel("Frame Number")
#     plt.ylabel("Coordinate Value")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

def plot_gesture_part_xy_vs_frame(gesture_part):
    """
    Plots x and y coordinates of a BodyPart vs frame number using Plotly.
    Shows timestamp as hover text.
    """
    filtered_frames = [frame for frame in gesture_part.frames if frame.confidence > 0.5]
    frame_numbers = [frame.get_timestamp() for frame in filtered_frames]
    x_coords = [frame.x for frame in filtered_frames]
    y_coords = [frame.y for frame in filtered_frames]
    timestamps = [frame.get_timestamp() for frame in filtered_frames]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame_numbers, y=x_coords, mode='lines+markers', name='X',
        text=[f"Time: {ts}" for ts in timestamps], hoverinfo='text+y+x'
    ))
    fig.add_trace(go.Scatter(
        x=frame_numbers, y=y_coords, mode='lines+markers', name='Y',
        text=[f"Time: {ts}" for ts in timestamps], hoverinfo='text+y+x'
    ))
    fig.update_layout(
        title=f"{gesture_part.part_name} coordinates vs Frame (confidence > 0.7)",
        xaxis_title="Frame Number",
        yaxis_title="Coordinate Value",
        legend_title="Coordinate"
    )
    fig.show()

def plot_velocity_magnitudes(velocities, part_name=""):
    """
    Plots velocity magnitudes vs frame number using Plotly.

    Args:
        velocities (List[float]): List of velocity magnitudes.
        part_name (str): Name of the body part for title.
    """
    frame_numbers = list(range(len(velocities)))
    timestamps = [f"{int(fn/23.976//3600):02}:{int((fn/23.976%3600)//60):02}:{int((fn/23.976)%60):02}.{int((fn/23.976 - int(fn/23.976))*1000):03}" for fn in frame_numbers]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame_numbers, y=velocities, mode='lines+markers', name='Velocity Magnitude',
        text=[f"Frame: {timestamp}, Velocity: {vel:.2f}" for timestamp, vel in zip(frame_numbers, velocities)],
        hoverinfo='text+y+x'    
    ))
    fig.update_layout(
        title=f"Velocity Magnitudes vs Frame for {part_name}",
        xaxis_title="Timestamp",
        yaxis_title="Velocity Magnitude",
        legend_title="Velocity"
    )   
    fig.show()
