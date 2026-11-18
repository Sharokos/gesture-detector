import os
import time
import traceback

from open_pose_handler import run_openpose, count_jsons
from gesture_manager.gesture_analysis import GestureAnalysis
from data_manager.debugger import export_person_bodyparts_data, export_person_features_data
from data_manager.plotter import (
    plot_person_sliding_windows,
    plot_body_part_features,
    plot_normalization_data,
    plot_person_bodypart_features
)
from config import (
    OPEN_POSE_DIR,
    GESTURE_DETECTION,
    GENERATE_JSONS,
    RESOLUTION,
    INPUT_DIR,
    POSE_BACKEND,
    POSE_HANDS,
    WRITE_POSE_VIDEO,
    SKIP_EXISTING_JSONS,
    RTMPOSE_MODE,
    RTMPOSE_DEVICE,
    RTMPOSE_DET_EVERY,
)
from utility import get_video_info


DEBUG_FLAG = True
DEEP_DEBUG = False
HAND_DEBUG = False

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def find_videos(input_root):
    """All videos under input_root (sub-folders included), as paths relative to input_root."""
    videos = []
    for dirpath, _, filenames in os.walk(input_root):
        for f in filenames:
            if f.lower().endswith(VIDEO_EXTENSIONS):
                videos.append(os.path.relpath(os.path.join(dirpath, f), input_root))
    return sorted(videos)


def export_debug(gesture_analysis, person_id, output_path):
    gesture_analysis.export_windows_to_csv(os.path.join(output_path, "windows.csv"))
    export_person_bodyparts_data(gesture_analysis, person_id, HAND_DEBUG, output_path)
    export_person_features_data(gesture_analysis, person_id, DEEP_DEBUG, output_path)
    plot_person_sliding_windows(gesture_analysis, person_id, output_path)
    plot_normalization_data(gesture_analysis, person_id, output_path)
    plot_person_bodypart_features(gesture_analysis, person_id, output_path)
    if DEEP_DEBUG:
        plot_body_part_features(gesture_analysis, person_id, output_path)


def extract_keypoints(video_path, json_dir, frame_count):
    if POSE_BACKEND == "rtmpose":
        from pose_rtmpose import run_rtmpose

        run_rtmpose(
            video_path=video_path,
            output_json_dir=json_dir,
            mode=RTMPOSE_MODE,
            hands=POSE_HANDS,
            device=RTMPOSE_DEVICE,
            det_every=RTMPOSE_DET_EVERY,
            write_video=WRITE_POSE_VIDEO,
            total_frames=frame_count,
        )
    elif POSE_BACKEND == "openpose":
        run_openpose(
            video_path=video_path,
            output_json_dir=json_dir,
            openpose_dir=OPEN_POSE_DIR,
            net_resolution=RESOLUTION,
            hands=POSE_HANDS,
            write_video=WRITE_POSE_VIDEO,
            total_frames=frame_count,
        )
    else:
        raise ValueError(f"Unknown POSE_BACKEND {POSE_BACKEND!r} (use 'rtmpose' or 'openpose')")


def process_video(video_path, video_output_root):
    # OpenPose output stays in "json" (as before), other backends get their own folder
    json_dir = os.path.join(
        video_output_root, "json" if POSE_BACKEND == "openpose" else f"json_{POSE_BACKEND}"
    )
    output_path = os.path.join(video_output_root, "results")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    video_fps, frame_count = get_video_info(video_path)
    print(f"\nProcessing video: {video_path}")
    print(f"Detected FPS: {video_fps:.3f}, frames: {frame_count}")

    # --- KEYPOINTS ---
    if GENERATE_JSONS:
        existing = count_jsons(json_dir)
        if SKIP_EXISTING_JSONS and frame_count and existing >= 0.99 * frame_count:
            print(f"Skipping keypoint extraction: {existing} jsons already in {json_dir}")
        else:
            print(f"Extracting keypoints ({POSE_BACKEND}): {video_path}")
            extract_keypoints(video_path, json_dir, frame_count)

    # --- GESTURE DETECTION ---
    if GESTURE_DETECTION:
        if count_jsons(json_dir) == 0:
            print(f"No keypoint jsons in {json_dir} (set GENERATE_JSONS=True to create them), skipping")
            return
        print(f"Processing (Gesture detection): {json_dir}")
        start = time.time()

        gesture_analysis = GestureAnalysis(input_folder=json_dir, frame_rate=video_fps)
        gesture_analysis.execute()

        person_id = 0
        gesture_analysis.determine_gestures_for_person(person_id, output_path, media_path=video_path)
        print(f"Gesture detection took {time.time() - start:.1f}s")

        if DEBUG_FLAG:
            export_debug(gesture_analysis, person_id, output_path)


def default_output_root(input_root):
    """OUTPUT_<input folder name>, next to the input folder."""
    input_root = os.path.normpath(input_root)
    return os.path.join(os.path.dirname(input_root), "OUTPUT_" + os.path.basename(input_root))


def is_inside(path, folder):
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(folder)]) == os.path.abspath(folder)
    except ValueError:  # different drives on Windows
        return False


def resolve_inputs(path=None, output=None):
    """
    Returns (input_root, output_root, [video paths relative to input_root]).

    path=None   -> every video under INPUT_DIR (from .env)
    path=folder -> every video under that folder (replaces INPUT_DIR)
    path=video  -> only that video.
    A folder or video inside INPUT_DIR keeps the usual output layout (existing keypoint jsons are
    reused); anything else gets OUTPUT_<its folder>/<video name>/ next to it.
    output      -> overrides the output root folder.
    """
    if path is None:
        if not INPUT_DIR or not os.path.isdir(INPUT_DIR):
            raise SystemExit(f"INPUT_DIR is not a folder: {INPUT_DIR!r} (set it in .env or pass a path)")
        path = INPUT_DIR

    in_input_dir = bool(INPUT_DIR) and os.path.isdir(INPUT_DIR) and is_inside(path, INPUT_DIR)

    if os.path.isdir(path):
        if in_input_dir:
            # Sub-folder of INPUT_DIR: keep the usual output layout so existing jsons are reused
            input_root = INPUT_DIR
            videos = [os.path.relpath(os.path.join(path, v), input_root) for v in find_videos(path)]
        else:
            input_root = path
            videos = find_videos(input_root)
    elif os.path.isfile(path):
        if not path.lower().endswith(VIDEO_EXTENSIONS):
            raise SystemExit(f"Not a supported video ({', '.join(VIDEO_EXTENSIONS)}): {path}")
        if in_input_dir:
            input_root = INPUT_DIR
        else:
            input_root = os.path.dirname(os.path.abspath(path))
        videos = [os.path.relpath(os.path.abspath(path), os.path.abspath(input_root))]
    else:
        raise SystemExit(f"Path does not exist: {path}")

    return input_root, output or default_output_root(input_root), videos


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Detect gestures in videos and export them for ELAN.")
    parser.add_argument(
        "path", nargs="?", default=None,
        help="a single video, or a folder of videos (default: INPUT_DIR from .env)",
    )
    parser.add_argument("-o", "--output", default=None, help="output folder (default: OUTPUT_<input folder>)")
    return parser.parse_args()


def main():
    args = parse_args()
    input_root, output_root, videos = resolve_inputs(args.path, args.output)
    os.makedirs(output_root, exist_ok=True)

    if not videos:
        print(f"No videos found in {input_root}")
        return
    print(f"Found {len(videos)} video(s), output in {output_root}")

    failed = []
    for rel_path in videos:
        video_path = os.path.join(input_root, rel_path)
        # OUTPUT_<input>/<same sub-folders>/<video name>/{json,results}
        rel_dir, video_file = os.path.split(rel_path)
        video_name = os.path.splitext(video_file)[0]
        video_output_root = os.path.join(output_root, rel_dir, video_name)
        try:
            process_video(video_path, video_output_root)
        except Exception:
            # One broken video should not stop the batch
            traceback.print_exc()
            failed.append(rel_path)

    if failed:
        print(f"\n{len(failed)} video(s) failed:")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
