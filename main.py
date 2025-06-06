from open_pose_handler import run_openpose

run_openpose(
    video_path=r"G:\OpenPose\detect_gestures\video.mp4",
    output_json_dir=r"G:\OpenPose\detect_gestures\output_jsons",
    openpose_dir=r"G:/OpenPose/openpose/",
    openpose_args="--hand --net_resolution 176x-1"
)