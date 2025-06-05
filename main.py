from open_pose_handler import run_openpose

run_openpose(
    video_path=r"G:\OpenPose\detect_gestures\video.avi",
    output_json_dir=r"G:\OpenPose\openpose\output_jsons",
    openpose_dir=r"G:/OpenPose/openpose/",
    openpose_args="--hand --net_resolution 320x176"
)