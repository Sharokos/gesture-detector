from pathlib import Path
import json

import numpy as np

from config import BODY_25_INDEX, COCO_18_INDEX, COCO_PARTS, HAND_PARTS
from data_model.person import PersonGesture

# Upper-body keypoints used to decide which detected person is "the" subject
_SELECTION_PARTS = ["Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist"]


def _frame_index(json_file):
    # OpenPose names files "<video>_000000000123_keypoints.json"
    return int(json_file.stem.split("_")[-2])


def _keypoint_index(n_keypoints):
    if n_keypoints == 25:
        return BODY_25_INDEX
    if n_keypoints == 18:
        return COCO_18_INDEX
    raise ValueError(
        f"Unsupported OpenPose body model with {n_keypoints} keypoints (expected BODY_25 or COCO)"
    )


def _select_person(people, index, prev_neck):
    """
    OpenPose does not track identities: the order of `people` can change between frames and
    background/false detections appear. Pick the subject: the most confident upper body,
    preferring the detection closest to where the subject was in the previous frame.
    """
    candidates = []
    for person in people:
        kp = np.asarray(person.get("pose_keypoints_2d", []), dtype=float).reshape(-1, 3)
        if kp.size == 0:
            continue
        conf = sum(kp[index[p], 2] for p in _SELECTION_PARTS)
        neck = kp[index["Neck"]]
        if prev_neck is not None and neck[2] > 0:
            dist = np.hypot(neck[0] - prev_neck[0], neck[1] - prev_neck[1])
        else:
            dist = np.inf
        candidates.append((conf, dist, person, kp))
    if not candidates:
        return None

    # Among detections with a reasonably confident upper body, take the one closest to the
    # previous position of the subject; without history, take the most confident one.
    max_conf = max(c[0] for c in candidates)
    plausible = [c for c in candidates if c[0] >= 0.7 * max_conf]
    if prev_neck is not None and any(np.isfinite(c[1]) for c in plausible):
        chosen = min(plausible, key=lambda c: c[1])
    else:
        chosen = max(plausible, key=lambda c: c[0])
    return chosen[2], chosen[3]


def _hand_array(person, key):
    data = person.get(key) or []
    if not data:
        return None
    return np.asarray(data, dtype=float).reshape(-1, 3)


def parse_openpose_and_populate_persons(gesture_analysis=None):
    """
    Parses OpenPose JSONs and builds the PersonGesture of the (single) subject in the video.
    The subject is always stored as person 0.
    """
    folder_path = Path(gesture_analysis.input_folder)
    json_files = sorted(folder_path.glob("*_keypoints.json"), key=_frame_index)
    if not json_files:
        raise FileNotFoundError(f"No OpenPose json files in {folder_path}")

    n_frames = _frame_index(json_files[-1]) + 1
    n_body = len(COCO_PARTS)
    n_hand = len(HAND_PARTS)

    body = np.full((n_frames, n_body, 3), np.nan)
    left = np.full((n_frames, n_hand, 3), np.nan)
    right = np.full((n_frames, n_hand, 3), np.nan)
    has_hands = False

    index = None
    prev_neck = None
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
        people = data.get("people", [])
        if not people:
            continue

        if index is None:
            index = _keypoint_index(len(people[0]["pose_keypoints_2d"]) // 3)

        selected = _select_person(people, index, prev_neck)
        if selected is None:
            continue
        person, kp = selected
        frame_idx = _frame_index(json_file)

        for i, part in enumerate(COCO_PARTS):
            if part in index:
                body[frame_idx, i] = kp[index[part]]
            elif part == "MidHip":
                # COCO model has no MidHip: derive it from the hips
                rh, lh = kp[index["RHip"]], kp[index["LHip"]]
                body[frame_idx, i] = [(rh[0] + lh[0]) / 2, (rh[1] + lh[1]) / 2, min(rh[2], lh[2])]

        neck = kp[index["Neck"]]
        if neck[2] > 0:
            prev_neck = neck

        lh_kp = _hand_array(person, "hand_left_keypoints_2d")
        rh_kp = _hand_array(person, "hand_right_keypoints_2d")
        if lh_kp is not None:
            left[frame_idx] = lh_kp[:n_hand]
            has_hands = True
        if rh_kp is not None:
            right[frame_idx] = rh_kp[:n_hand]
            has_hands = True

    subject = PersonGesture(0, gesture_analysis=gesture_analysis)
    subject.set_keypoints(body, left if has_hands else None, right if has_hands else None)

    gesture_analysis.number_of_frames = n_frames
    gesture_analysis.persons = {0: subject}
