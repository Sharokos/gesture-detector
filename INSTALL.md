# Installing the gesture detector

This guide sets up the gesture detector from scratch on a Windows 10/11 laptop. When you are done,
you can point it at a folder of videos (or a single video) and get gesture time spans as
`gestures.json` and as an ELAN file (`gestures.eaf`).

**Contents**

1. [What you need](#1-what-you-need)
2. [Install Python 3.11](#2-install-python-311)
3. [Install Git and get the code](#3-install-git-and-get-the-code)
4. [Create the Python environment](#4-create-the-python-environment)
5. [Configure `.env`](#5-configure-env)
6. [Add your videos](#6-add-your-videos)
7. [First run](#7-first-run)
8. [Where the results are and how to check them](#8-where-the-results-are-and-how-to-check-them)
9. [Optional: NVIDIA GPU](#9-optional-nvidia-gpu)
10. [Optional: OpenPose instead of RTMPose](#10-optional-openpose-instead-of-rtmpose)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What you need

| | |
|---|---|
| Operating system | Windows 10 or 11, 64-bit (the commands below use PowerShell) |
| Python | **3.11** (3.11.9 is what this project is tested with; 3.10 or older will not work) |
| GPU | not needed. Everything runs on the CPU. An NVIDIA GPU makes keypoint extraction faster ([section 9](#9-optional-nvidia-gpu)) |
| ELAN | to open the `.eaf` results: <https://archive.mpi.nl/tla/elan/download> |

You do **not** need OpenPose. Keypoints are extracted with RTMPose, which is installed as a normal
Python package.

## 2. Install Python 3.11

1. Download the **Windows installer (64-bit)** of Python 3.11 from
   <https://www.python.org/downloads/windows/> (any 3.11.x release).
2. Run it. On the first screen, tick **"Add python.exe to PATH"**, then click **Install Now**.
3. Open a new PowerShell window (Start menu → type `PowerShell`) and check:

   ```powershell
   py -3.11 --version
   ```

   It should print `Python 3.11.x`.

> Alternative: `winget install Python.Python.3.11` does the same from PowerShell.

## 3. Install Git and get the code

1. Install Git from <https://git-scm.com/download/win> (the default options are fine), or run
   `winget install Git.Git`. Open a **new** PowerShell window afterwards.
2. Go to the folder where you want the project and clone it:

   ```powershell
   cd $HOME\Documents
   git clone https://github.com/Sharokos/gesture-detector.git
   cd gesture-detector
   ```
All following commands are run **inside the `gesture-detector` folder**.

## 4. Create the Python environment

The project keeps its own copy of all libraries in a `.venv` folder, so nothing else on the laptop is
affected.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

After `Activate.ps1`, the prompt starts with `(.venv)`. You need to activate the environment again
in every new PowerShell window before running the scripts.

> **"running scripts is disabled on this system"** when activating? Allow scripts for your user once,
> then activate again:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
>
> Or skip activation entirely and call the environment's Python directly, e.g.
> `.\.venv\Scripts\python.exe main.py`.

## 5. Configure `.env`

The settings live in a file named `.env` in the project folder. It is not part of the repository, so create it from the template:

```powershell
Copy-Item .env.example .env
notepad .env
```

At minimum, set `INPUT_DIR` to the **full path** of the folder that will hold your videos, e.g.:

```
INPUT_DIR=C:\Users\<you>\Documents\gesture-detector\INPUT_VIDEOS
```

Leave the rest as it is for now. What the main settings do:

| Setting | Meaning |
|---|---|
| `INPUT_DIR` | folder with the videos; sub-folders are searched too |
| `GENERATE_JSONS` | `True` = extract keypoints from the videos (the slow step). Set to `False` to only re-run the gesture detection on keypoints you already have, which takes seconds |
| `GESTURE_DETECTION` | `True` = detect gestures from the keypoints |
| `POSE_BACKEND` | `rtmpose` (recommended) or `openpose` |
| `RTMPOSE_MODE` | `lightweight` (fastest), `balanced` (default), `performance` (most accurate, slowest) |
| `RTMPOSE_DEVICE` | `cpu`, or `cuda` for an NVIDIA GPU ([section 9](#9-optional-nvidia-gpu)) |
| `SKIP_EXISTING_JSONS` | don't extract keypoints again for videos that already have them |
| `THRESHOLD_METHOD` | `otsu` (adapts to each video, default) or `percentile` (not recommended) |

The detection parameters themselves (window sizes, weights, …) are in `config.py`.

## 6. Add your videos

Videos are not stored in Git. Copy them into the `INPUT_DIR` folder. One sub-folder per video is
fine (that is how the existing data is organised):

```
INPUT_VIDEOS\
    Emotional_LA_Pianist_L2_EM_AV_FS_P31\
        Emotional_LA_Pianist_L2_EM_AV_FS_P31.mp4
    another_recording.mp4
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`.

> **Already installed somewhere else?** Copy these too, to save time:
>
> - the `OUTPUT_INPUT_VIDEOS` folder (next to `INPUT_VIDEOS`): it contains the keypoints that were
>   already extracted, so those videos are not processed again;
> - the `models` folder from the project: then nothing needs to be downloaded (useful without
>   internet access).

## 7. First run

Start with **one** video to check that everything works:

```powershell
python main.py "INPUT_VIDEOS\Emotional_LA_Pianist_L2_EM_AV_FS_P31\Emotional_LA_Pianist_L2_EM_AV_FS_P31.mp4"
```

On the very first run, the two RTMPose models (≈90 MB + 50 MB download, ≈190 MB unpacked) are
downloaded into the project's `models` folder. You will see a progress bar for each. This happens
only once.

Then you should see something like:

```
Found 1 video(s), output in ...\OUTPUT_INPUT_VIDEOS
Processing video: ...
Detected FPS: 29.970, frames: 7596
Extracting keypoints (rtmpose): ...
RTMPose frames: 100%|████████████| 7596/7596
Done. 7596 frames in 470s (16.2 fps).
...
Exported 27 gestures to: ...\results\gestures.json
Gesture detection took 5.0s
```

**How long it takes.** Keypoint extraction is the slow part. On the CPU of the laptop this was
tested on (12 threads), a 1080p video was processed at about 16 frames per second in `balanced` mode,
so roughly twice the length of the video. `lightweight` mode ran at about 40 frames per second
(faster than real time). Gesture detection itself takes a few seconds, plus about 15–20 seconds for
the debug exports (see [Troubleshooting](#11-troubleshooting) to turn those off).

Once that works, process everything:

```powershell
python main.py                     # all videos in INPUT_DIR
python main.py "D:\other\folder"   # all videos in another folder
python main.py video.mp4 -o "D:\results"   # choose where the results go
python main.py --help
```

If one video fails, the others are still processed and the failures are listed at the end.

## 8. Where the results are and how to check them

For each video you get a folder in `OUTPUT_<name of the input folder>`, next to the input folder:

```
OUTPUT_INPUT_VIDEOS\
    <sub-folder>\<video name>\
        json_rtmpose\      keypoints, one file per frame (input for the detection)
        results\
            gestures.eaf   gestures for ELAN
            gestures.json  the same gestures as start/end times in seconds
            windows.csv, person_0_plot.html, feature_data\, ...   debug data
```

A video outside `INPUT_DIR` gets its results in `OUTPUT_<its folder name>` next to that video instead.

**In ELAN:** open `results\gestures.eaf`. The video is linked inside the file, so ELAN opens it
with the annotations on the `Gesture_Unit` tier. If you moved the video after processing, ELAN asks
you to locate it.

**In the browser:** open `video_helper\index.html`, choose the video, then load
`results\gestures.json` to see the detected gestures while the video plays. You can load a second
JSON (e.g. manual annotations) to compare them.

**Against your manual annotations:** save the manual ELAN annotation of a video next to it with the
same name (`my_video.mp4` → `my_video.eaf`, gestures on a tier called `Gesture_Unit`), process the
videos, then run:

```powershell
python evaluate.py
```

It prints precision, recall and F1 per video and on average. Use this average when you change
settings in `config.py`, rather than tuning on a single video. Options: `--tier <name>` if your
tier has another name, `--tolerance 0.3` to accept 0.3 s of slack when matching gestures.

## 9. Optional: NVIDIA GPU

With an NVIDIA GPU, keypoint extraction can be several times faster. You need:

- a recent NVIDIA driver;
- the CUDA and cuDNN versions that match the installed onnxruntime version. The table at
  <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements>
  lists them (onnxruntime 1.31 is built for CUDA 12 and cuDNN 9).

Then swap the CPU package for the GPU one (both installed at the same time do not work):

```powershell
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.31.1
python -c "import onnxruntime as o; print(o.get_available_providers())"
```

The last line must list `CUDAExecutionProvider`. Then set `RTMPOSE_DEVICE=cuda` in `.env`.

> `requirements.txt` installs the CPU version. If you run `pip install -r requirements.txt` again
> later, repeat the swap above.

## 10. Optional: OpenPose instead of RTMPose

Only needed to reproduce older results. RTMPose is faster, needs no separate install and produced
nearly identical gestures in our comparison.

1. Download the Windows portable demo from the OpenPose releases page
   (<https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases>, the GPU build needs an
   NVIDIA GPU) and unpack it, e.g. to `C:\openpose`. The folder must contain `bin\OpenPoseDemo.exe`.
2. Download the models with `models\getModels.bat` inside that folder (if the download servers do not
   respond, the model files have to be obtained separately; copy the `models` folder of the old
   OpenPose install).
3. In `.env`: `POSE_BACKEND=openpose`, `OPEN_POSE_DIR=C:\openpose`, and `RESOLUTION=176x-1`
   (lower = faster, e.g. `368x-1` = more accurate).

OpenPose keypoints are stored in a `json` folder per video (RTMPose uses `json_rtmpose`), so both can
exist side by side.

## 11. Troubleshooting

| Problem | Solution |
|---|---|
| `No module named 'cv2'` / `'rtmlib'` / … | The environment is not active. Run `.\.venv\Scripts\Activate.ps1` (prompt shows `(.venv)`), or use `.\.venv\Scripts\python.exe main.py`. |
| `py` is not recognised / wrong Python version | Python was not added to PATH. Reinstall Python 3.11 and tick "Add python.exe to PATH", then open a new PowerShell window. |
| `INPUT_DIR is not a folder` | `.env` is missing or the path in `INPUT_DIR` is wrong. Use a full path, without quotes. |
| `No keypoint jsons in ... skipping` | No keypoints exist yet for that video and `GENERATE_JSONS=False`. Set `GENERATE_JSONS=True` in `.env`. |
| Model download fails (proxy / no internet) | Copy the `models` folder from another install, so that `models\hub\checkpoints` contains `yolox_m_8xb8-300e_humanart-c2c7a14a.onnx` and `rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx` (other modes use other files). |
| `cv2` reports version 5.x or OpenCV errors after installing something else | `pip install --force-reinstall opencv-python==4.13.0.92 opencv-contrib-python==4.13.0.92`. Both packages must have the same version. |
| Keypoint extraction too slow | Use `RTMPOSE_MODE=lightweight`, keep `WRITE_POSE_VIDEO=False`, or use a GPU ([section 9](#9-optional-nvidia-gpu)). |
| Each video takes ~20 s extra after the detection | Those are the debug exports (Excel files and plots). Set `DEBUG_FLAG = False` at the top of `main.py` if you only need `gestures.json` / `gestures.eaf`. |
| Wrong person tracked (e.g. an interviewer in the frame) | Run once with `WRITE_POSE_VIDEO=True` and check `json_rtmpose\rtmpose_output.mp4`. The subject is chosen as the person with the most confidently detected upper body. |
| Variable-frame-rate video (phone, screen recording) | Timestamps can drift. Convert to a constant frame rate first, e.g. `ffmpeg -i in.mp4 -r 30 -fps_mode cfr out.mp4`. |

To update to the latest version of the code later:

```powershell
git pull
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
