import os
import subprocess
import time
from pathlib import Path

from tqdm import tqdm


def count_jsons(output_json_dir):
    with os.scandir(output_json_dir) as it:
        return sum(1 for e in it if e.name.endswith("_keypoints.json"))


def run_openpose(
    video_path,
    output_json_dir,
    openpose_dir,
    net_resolution="368x-1",
    hands=False,
    write_video=False,
    total_frames=None,
    extra_args=None,
):
    """
    Runs OpenPose on a video using the CLI and shows a progress bar.

    Speed notes:
      - `hands=False`: the hand network is by far the most expensive part, and the gesture score
        only uses the body wrists.
      - `write_video=False`: skips skeleton rendering + video encoding.
      - `--number_people_max 1`: one seated subject; also avoids picking up bystanders.

    Args:
        video_path (str): Path to the input video.
        output_json_dir (str): Where OpenPose should write JSONs.
        openpose_dir (str or Path): OpenPose root directory (where bin/OpenPoseDemo.exe lives).
        net_resolution (str): OpenPose --net_resolution.
        hands (bool): also run the hand keypoint detector.
        write_video (bool): write a rendered skeleton video next to the jsons.
        total_frames (int): number of frames in the video, for the progress bar.
        extra_args (list[str]): any extra OpenPose CLI arguments.
    """
    openpose_dir = Path(openpose_dir)
    output_json_dir = Path(output_json_dir)
    output_json_dir.mkdir(parents=True, exist_ok=True)

    openpose_exe = openpose_dir / "bin" / "OpenPoseDemo.exe"
    if not openpose_exe.exists():
        raise FileNotFoundError(f"OpenPose executable not found: {openpose_exe}")

    cmd = [
        str(openpose_exe),
        "--video", str(Path(video_path).resolve()),
        "--write_json", str(output_json_dir.resolve()),
        "--display", "0",
        "--net_resolution", net_resolution,
        "--number_people_max", "1",
    ]
    if write_video:
        cmd += ["--write_video", str((output_json_dir / "openpose_output.avi").resolve())]
    else:
        cmd += ["--render_pose", "0"]
    if hands:
        cmd += ["--hand"]
    if extra_args:
        cmd += list(extra_args)

    print("Running OpenPose:", " ".join(cmd))
    # OpenPose must run from its root folder to find the models
    process = subprocess.Popen(cmd, cwd=str(openpose_dir))

    start = time.time()
    pbar = tqdm(total=total_frames or None, desc="OpenPose frames", ncols=80)
    json_count = 0
    while process.poll() is None:
        new_count = count_jsons(output_json_dir)
        if new_count > json_count:
            pbar.update(new_count - json_count)
            json_count = new_count
        time.sleep(1.0)
    new_count = count_jsons(output_json_dir)
    pbar.update(new_count - json_count)
    pbar.close()

    if process.returncode != 0:
        raise RuntimeError(f"OpenPose failed with exit code {process.returncode}")
    elapsed = time.time() - start
    print(f"Done. {new_count} frames in {elapsed:.0f}s ({new_count / max(elapsed, 1e-6):.1f} fps).")
