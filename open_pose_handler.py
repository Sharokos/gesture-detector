import subprocess
import time
from pathlib import Path
from tqdm import tqdm
import cv2

def run_openpose(video_path, output_json_dir, openpose_dir, openpose_args=""):
    """
    Runs OpenPose on a video using the CLI and shows a progress bar.

    Args:
        video_path (str): Path to the input video.
        output_json_dir (str): Where OpenPose should write JSONs.
        openpose_dir (str or Path): OpenPose root directory (where OpenPoseDemo.exe lives).
        openpose_args (str): Extra CLI args like '--hand --face'.
    """
    openpose_dir = Path(openpose_dir)
    output_json_dir = Path(output_json_dir)
    output_json_dir.mkdir(parents=True, exist_ok=True)

    # Path to the .exe
    openpose_exe = openpose_dir / "bin" / "OpenPoseDemo.exe"
    total_frames = get_video_frame_count(video_path)
    # Build command
    cmd = [
        str(openpose_exe),
        "--video", str(video_path),
        "--write_json", str(output_json_dir),
        "--display", "0",
        "--render_pose", "0",
    ] + openpose_args.split()

    # Launch OpenPose with cwd set to OpenPose directory
    process = subprocess.Popen(cmd, cwd=str(openpose_dir))#, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

    # Progress bar logic
    print("Running OpenPose...")
    pbar = tqdm(total=total_frames, desc="Processing frames", ncols=70)
    json_count = 0

    while process.poll() is None:
        new_count = len(list(output_json_dir.glob("*.json")))
        if new_count > json_count:
            delta = new_count - json_count
            pbar.update(delta)
            json_count = new_count
        time.sleep(0.5)

    pbar.close()
    print(f"✅ Done. {json_count} frames processed.")


def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count