"""
Keypoint extraction with RTMPose (via rtmlib + onnxruntime), as a faster and more accurate
replacement for OpenPose. Writes one OpenPose-style json per frame
(`<video>_<frame:012d>_keypoints.json`, 18-keypoint COCO layout), so the gesture detection reads
it exactly like OpenPose output.

Only ONE person is tracked (the seated subject):
  - the person detector (YOLOX) runs every `det_every` frames, or when the subject is lost;
  - on the first detection, the person with the most confident upper body is chosen as subject
    (so a partially visible interviewer in the foreground is ignored);
  - in between, the subject's box is derived from its keypoints of the previous frame and the pose
    model runs on that box only.

Models are downloaded on first use into ./models (TORCH_HOME), not into the user profile.

Standalone usage:
    python pose_rtmpose.py path/to/video.mp4 path/to/json_dir [--mode balanced] [--hands] [--write-video]
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"

# OpenPose COCO-18 indices of the upper body (Neck, shoulders, elbows, wrists)
UPPER_BODY = [1, 2, 3, 4, 5, 6, 7]
NECK = 1
# Keypoints above this score are used to follow the subject between detections
TRACK_SCORE = 0.3
BOX_EXPANSION = 1.25
# Frames without a usable subject before we pick the subject again from scratch
LOST_AFTER = 10


def _load_models(mode, hands, device, models_dir):
    # rtmlib reads TORCH_HOME when it downloads the models
    os.environ.setdefault("TORCH_HOME", str(models_dir))
    from rtmlib import Body, Wholebody

    solution = Wholebody if hands else Body
    model = solution(mode=mode, to_openpose=True, backend="onnxruntime", device=device)
    return model.det_model, model.pose_model


def _box_from_keypoints(kpts, scores):
    good = scores > TRACK_SCORE
    if good.sum() < 4:
        return None
    pts = kpts[good]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) / 2 * BOX_EXPANSION, (y1 - y0) / 2 * BOX_EXPANSION
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh])


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _match_detection(boxes, kpts, scores, current_box):
    """Detection box of the tracked subject: must contain its neck, best overlap wins."""
    neck = kpts[NECK] if scores[NECK] > TRACK_SCORE else kpts[scores > TRACK_SCORE].mean(axis=0)
    best, best_iou = None, -1.0
    for box in boxes:
        if box[0] <= neck[0] <= box[2] and box[1] <= neck[1] <= box[3]:
            iou = _iou(box, current_box)
            if iou > best_iou:
                best, best_iou = box, iou
    return best


def _to_openpose_person(kpts, scores, hands):
    """kpts (K, 2), scores (K,) in rtmlib's OpenPose order -> OpenPose json person dict."""
    scores = np.clip(scores, 0.0, 1.0)  # SimCC scores can slightly exceed 1

    def flat(sl):
        return np.column_stack([kpts[sl], scores[sl]]).round(3).ravel().tolist()

    person = {
        "person_id": [-1],
        "pose_keypoints_2d": flat(slice(0, 18)),
        "face_keypoints_2d": [],
        "hand_left_keypoints_2d": [],
        "hand_right_keypoints_2d": [],
    }
    if hands:
        # rtmlib wholebody OpenPose layout: 0-17 body, 18-23 feet, 24-91 face, 92-112 L hand, 113-133 R hand
        person["hand_left_keypoints_2d"] = flat(slice(92, 113))
        person["hand_right_keypoints_2d"] = flat(slice(113, 134))
    return person


def run_rtmpose(
    video_path,
    output_json_dir,
    mode="balanced",
    hands=False,
    device="cpu",
    det_every=15,
    write_video=False,
    total_frames=None,
    models_dir=DEFAULT_MODELS_DIR,
    max_frames=None,
):
    """
    Args:
        video_path: input video.
        output_json_dir: where the per-frame jsons are written.
        mode: "lightweight" (fastest), "balanced", or "performance" (most accurate, slowest).
        hands: use the whole-body model (body + hands, slower). The gesture score does not need it.
        device: "cpu" or "cuda" (needs onnxruntime-gpu instead of onnxruntime).
        det_every: run the person detector every N frames (it is the expensive part).
        write_video: also write a skeleton overlay video (rtmpose_output.mp4) for visual checks.
        total_frames: frame count, for the progress bar.
    """
    output_json_dir = Path(output_json_dir)
    output_json_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    det_model, pose_model = _load_models(mode, hands, device, models_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = total_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    if max_frames and total_frames:
        total_frames = min(total_frames, max_frames)

    writer = None
    if write_video:
        from rtmlib import draw_skeleton

        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_json_dir / "rtmpose_output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    subject_box = None   # box used for the pose model on the next frame
    prev = None          # (kpts, scores) of the subject in the previous frame
    frames_lost = 0
    frame_idx = 0
    start = time.time()
    pbar = tqdm(total=total_frames, desc="RTMPose frames", ncols=80)

    try:
        while max_frames is None or frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            kpts = scores = None
            if subject_box is None or frame_idx % det_every == 0:
                boxes = det_model(frame)
                boxes = [] if boxes is None else list(np.asarray(boxes).reshape(-1, 4))
                if boxes and (subject_box is None or prev is None):
                    # (Re)acquire the subject: pose for every person, keep the best upper body
                    all_k, all_s = pose_model(frame, bboxes=boxes)
                    best = int(np.argmax(all_s[:, UPPER_BODY].sum(axis=1)))
                    kpts, scores = all_k[best], all_s[best]
                elif boxes:
                    matched = _match_detection(boxes, *prev, subject_box)
                    if matched is not None:
                        subject_box = matched

            if kpts is None and subject_box is not None:
                k, s = pose_model(frame, bboxes=[subject_box])
                kpts, scores = k[0], s[0]

            people = []
            if kpts is not None and (scores[UPPER_BODY] > TRACK_SCORE).sum() >= 3:
                people.append(_to_openpose_person(kpts, scores, hands))
                prev = (kpts, scores)
                subject_box = _box_from_keypoints(kpts, scores)
                frames_lost = 0
            else:
                frames_lost += 1
                if frames_lost >= LOST_AFTER:
                    subject_box, prev = None, None  # pick the subject again at the next frame

            out_file = output_json_dir / f"{stem}_{frame_idx:012d}_keypoints.json"
            with open(out_file, "w") as f:
                json.dump({"version": 1.3, "people": people}, f)

            if writer is not None:
                if people:
                    frame = draw_skeleton(frame, kpts[None], scores[None], openpose_skeleton=True,
                                          kpt_thr=TRACK_SCORE)
                writer.write(frame)

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = time.time() - start
    print(f"Done. {frame_idx} frames in {elapsed:.0f}s ({frame_idx / max(elapsed, 1e-6):.1f} fps).")
    return frame_idx


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("output_json_dir")
    parser.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--hands", action="store_true", help="whole-body model with hand keypoints")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (needs onnxruntime-gpu)")
    parser.add_argument("--det-every", type=int, default=15, help="run the person detector every N frames")
    parser.add_argument("--write-video", action="store_true", help="write a skeleton overlay video")
    parser.add_argument("--max-frames", type=int, default=None, help="only process the first N frames (testing)")
    args = parser.parse_args()
    run_rtmpose(args.video, args.output_json_dir, mode=args.mode, hands=args.hands, device=args.device,
                det_every=args.det_every, write_video=args.write_video, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
