"""Mouth ROI extractor for AV-HuBERT input.

Ported from av_hubert/avhubert/preparation/align_mouth.py (Facebook AI Research)
with two backend options:

  backend='dlib'  (PRODUCTION — identical to av-hubert's official pipeline)
  ─────────────────────────────────────────────────────────────────────────────
  • Uses dlib frontal + CNN face detectors + 68-point shape predictor
  • Requires model files (download links in __init__ docstring)
  • Applies mean-face affine warp before cropping → most stable ROI
  • Output matches the precomputed .npy files in data/utts/

  backend='haar'  (FALLBACK — no external model files needed)
  ─────────────────────────────────────────────────────────────────────────────
  • Uses cv2 built-in Haar cascades (bundled with opencv-python)
  • Works out of the box in any environment
  • Less robust for tilted/profile faces; fine for LRS2/LRS3 (face-centred)
  • No mean-face affine — crops around detected mouth centre directly

Both backends return a tensor of shape  [T, 1, 96, 96]  float32 in [0, 1]
ready to pass directly into AVHubertVSR.extract().

Usage
─────
    from avsd_ger.frontend.mouth_roi import MouthROIExtractor

    # Production (dlib):
    extractor = MouthROIExtractor(
        backend='dlib',
        face_predictor_path='checkpoints/shape_predictor_68_face_landmarks.dat',
        cnn_detector_path='checkpoints/mmod_human_face_detector.dat',
        mean_face_path='av_hubert/avhubert/preparation/data/20words_mean_face.npy',
    )

    # Fallback (no downloads):
    extractor = MouthROIExtractor(backend='haar')

    video_frames = extractor.extract_from_file('path/to/video.mp4')
    # video_frames: torch.Tensor [T, 1, 96, 96] float32
"""
from __future__ import annotations

import os
from collections import deque
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────── shared crop helpers ──
# Directly ported from av_hubert/avhubert/preparation/align_mouth.py

def _linear_interpolate(landmarks, start_idx, stop_idx):
    start = landmarks[start_idx]
    stop  = landmarks[stop_idx]
    delta = stop - start
    for i in range(1, stop_idx - start_idx):
        landmarks[start_idx + i] = start + i / float(stop_idx - start_idx) * delta
    return landmarks


def _landmarks_interpolate(landmarks):
    """Fill None entries by linear interpolation; extend at boundaries."""
    valid = [i for i, v in enumerate(landmarks) if v is not None]
    if not valid:
        return None
    for k in range(1, len(valid)):
        if valid[k] - valid[k - 1] > 1:
            landmarks = _linear_interpolate(landmarks, valid[k - 1], valid[k])
    valid = [i for i, v in enumerate(landmarks) if v is not None]
    landmarks[: valid[0]]  = [landmarks[valid[0]]]  * valid[0]
    landmarks[valid[-1] :] = [landmarks[valid[-1]]] * (len(landmarks) - valid[-1])
    assert all(v is not None for v in landmarks), "interpolation failed"
    return landmarks


def _cut_patch(img, landmarks, height, width, threshold=5):
    """Crop a (2*height) × (2*width) patch centred on the landmark mean."""
    cx, cy = np.mean(landmarks, axis=0)
    cy = float(np.clip(cy, height, img.shape[0] - height))
    cx = float(np.clip(cx, width,  img.shape[1] - width))
    return np.copy(
        img[int(round(cy) - round(height)): int(round(cy) + round(height)),
            int(round(cx) - round(width)) : int(round(cx) + round(width))]
    )


def _warp_img(src, dst, img, std_size):
    from skimage import transform as sktf
    tform  = sktf.estimate_transform("similarity", src, dst)
    warped = sktf.warp(img, inverse_map=tform.inverse, output_shape=std_size)
    return (warped * 255).astype(np.uint8), tform


def _apply_transform(transform, img, std_size):
    from skimage import transform as sktf
    warped = sktf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    return (warped * 255).astype(np.uint8)


def _frames_to_tensor(frames: list[np.ndarray]):
    """Convert list of (H, W) uint8 grey frames to [T, 1, 96, 96] float32 [0,1].

    Returns a torch.Tensor when torch is available, otherwise a numpy ndarray
    with the same shape / dtype so callers work in both environments.
    """
    arr = np.stack(frames, axis=0)           # [T, H, W]
    arr = arr[:, np.newaxis, :, :]           # [T, 1, H, W]
    arr = arr.astype(np.float32) / 255.0
    try:
        import torch
        return torch.from_numpy(arr)
    except ImportError:
        return arr


# ──────────────────────────────────────────────────────────────────────────────
class MouthROIExtractor:
    """Extract 96×96 grayscale mouth-ROI clips from a video file.

    Args
    ────
    backend : 'dlib' | 'haar'
        Landmark detection backend. See module docstring.
    crop_height : int
        Half-height of the mouth crop (default 48 → full crop 96 px).
    crop_width  : int
        Half-width  of the mouth crop (default 48 → full crop 96 px).
    window_margin : int
        Sliding window for temporal smoothing of landmarks (av-hubert default 12).

    dlib-only args
    ──────────────
    face_predictor_path : str
        Path to shape_predictor_68_face_landmarks.dat
        Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    cnn_detector_path : str
        Path to mmod_human_face_detector.dat
        Download: http://dlib.net/files/mmod_human_face_detector.dat.bz2
    mean_face_path : str
        Path to 20words_mean_face.npy
        Download: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
                  blob/master/preprocessing/20words_mean_face.npy
    mouth_start_idx : int  (default 48)
    mouth_stop_idx  : int  (default 68)
        Dlib 68-pt landmark range for the mouth.
    """

    # Dlib stable points for affine alignment (nose bridge + eye corners)
    _DLIB_STABLE_IDS = [33, 36, 39, 42, 45]
    _STD_SIZE        = (256, 256)

    def __init__(
        self,
        backend: str = "dlib",
        crop_height: int = 48,
        crop_width:  int = 48,
        window_margin: int = 12,
        # dlib-only
        face_predictor_path: Optional[str] = None,
        cnn_detector_path:   Optional[str] = None,
        mean_face_path:      Optional[str] = None,
        mouth_start_idx: int = 48,
        mouth_stop_idx:  int = 68,
    ):
        self.backend       = backend.lower()
        self.crop_height   = crop_height
        self.crop_width    = crop_width
        self.window_margin = window_margin
        self.start_idx     = mouth_start_idx
        self.stop_idx      = mouth_stop_idx

        if self.backend == "dlib":
            self._init_dlib(face_predictor_path, cnn_detector_path, mean_face_path)
        elif self.backend == "haar":
            self._init_haar()
        else:
            raise ValueError(f"Unknown backend {backend!r}; choose 'dlib' or 'haar'")

    # ──────────────────────────────────────────────── dlib backend init ───────
    def _init_dlib(self, face_predictor_path, cnn_detector_path, mean_face_path):
        try:
            import dlib
        except ImportError:
            raise RuntimeError(
                "dlib is required for backend='dlib'. "
                "Install it with: pip install dlib  (needs cmake + C++ compiler). "
                "Or use backend='haar' for a dependency-free fallback."
            )
        if not face_predictor_path or not os.path.isfile(face_predictor_path):
            raise FileNotFoundError(
                f"shape_predictor file not found: {face_predictor_path!r}\n"
                "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        if not cnn_detector_path or not os.path.isfile(cnn_detector_path):
            raise FileNotFoundError(
                f"CNN detector file not found: {cnn_detector_path!r}\n"
                "Download from: http://dlib.net/files/mmod_human_face_detector.dat.bz2"
            )
        if not mean_face_path or not os.path.isfile(mean_face_path):
            raise FileNotFoundError(
                f"mean_face file not found: {mean_face_path!r}\n"
                "Download from: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks"
                "/blob/master/preprocessing/20words_mean_face.npy"
            )
        self._detector     = dlib.get_frontal_face_detector()
        self._cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
        self._predictor    = dlib.shape_predictor(face_predictor_path)
        self._mean_face    = np.load(mean_face_path)

    def _detect_landmarks_dlib(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return (68,2) int32 array or None if no face found."""
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        rects = self._detector(gray, 1)
        if not rects:
            rects = [d.rect for d in self._cnn_detector(gray)]
        if not rects:
            return None
        shape = self._predictor(gray, rects[0])
        pts   = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            pts[i] = (shape.part(i).x, shape.part(i).y)
        return pts

    def _crop_sequence_dlib(self, frames: list[np.ndarray]) -> Optional[np.ndarray]:
        """Full av-hubert pipeline: detect → interpolate → warp → crop."""
        landmarks_raw = [self._detect_landmarks_dlib(f) for f in frames]
        landmarks = _landmarks_interpolate(landmarks_raw)
        if landmarks is None:
            return None

        margin  = min(len(frames), self.window_margin)
        q_frame, q_lm = deque(), deque()
        sequence = []
        trans    = None

        for idx, (frame, lm) in enumerate(zip(frames, landmarks)):
            q_frame.append(frame)
            q_lm.append(lm)

            if len(q_frame) == margin:
                smooth_lm = np.mean(q_lm, axis=0)
                cur_frame  = q_frame.popleft()
                cur_lm     = q_lm.popleft()

                warped, trans = _warp_img(
                    smooth_lm[self._DLIB_STABLE_IDS, :],
                    self._mean_face[self._DLIB_STABLE_IDS, :],
                    cur_frame,
                    self._STD_SIZE,
                )
                trans_lm = trans(cur_lm)
                patch = _cut_patch(
                    warped, trans_lm[self.start_idx: self.stop_idx],
                    self.crop_height, self.crop_width,
                )
                sequence.append(patch)

            if idx == len(frames) - 1 and trans is not None:
                while q_frame:
                    wf = _apply_transform(trans, q_frame.popleft(), self._STD_SIZE)
                    tl = trans(q_lm.popleft())
                    sequence.append(
                        _cut_patch(wf, tl[self.start_idx: self.stop_idx],
                                   self.crop_height, self.crop_width)
                    )

        return np.array(sequence) if sequence else None

    # ──────────────────────────────────────────────── haar backend init ───────
    def _init_haar(self):
        data_dir = cv2.data.haarcascades
        self._face_cascade  = cv2.CascadeClassifier(
            os.path.join(data_dir, "haarcascade_frontalface_default.xml")
        )
        self._eye_cascade   = cv2.CascadeClassifier(
            os.path.join(data_dir, "haarcascade_eye.xml")
        )
        # Mouth cascade (smile) as a soft cue; we mainly use face geometry
        smile_xml = os.path.join(data_dir, "haarcascade_smile.xml")
        self._smile_cascade = cv2.CascadeClassifier(smile_xml) if os.path.isfile(smile_xml) else None
        self._last_mouth_cx: Optional[float] = None
        self._last_mouth_cy: Optional[float] = None

    def _detect_mouth_centre_haar(self, frame_bgr: np.ndarray):
        """Return (cx, cy) mouth centre in pixel coords, or None."""
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H, W  = gray.shape

        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(W // 4, H // 4),
        )
        if len(faces) == 0:
            return None

        # Largest face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

        # Mouth is in the lower 40 % of the face bounding box, horizontally centred
        mouth_y = y + int(h * 0.72)   # ~72 % down from face top
        mouth_x = x + w // 2
        return float(mouth_x), float(mouth_y)

    def _crop_sequence_haar(self, frames: list[np.ndarray]) -> np.ndarray:
        """Detect mouth per frame, smooth positions, crop & resize."""
        centres: list[Optional[tuple]] = []
        for f in frames:
            centres.append(self._detect_mouth_centre_haar(f))

        # Temporal interpolation for missed frames
        cx_arr = [c[0] if c else None for c in centres]
        cy_arr = [c[1] if c else None for c in centres]

        def _interp_1d(vals):
            arr = np.array([v if v is not None else np.nan for v in vals], dtype=float)
            nans = np.isnan(arr)
            if nans.all():
                arr[:] = frames[0].shape[1] / 2  # fallback to frame centre
            else:
                ok = np.where(~nans)[0]
                arr[nans] = np.interp(np.where(nans)[0], ok, arr[ok])
            return arr

        cx_arr = _interp_1d(cx_arr)
        cy_arr = _interp_1d(cy_arr)

        # Temporal smoothing (sliding mean over window_margin)
        margin = min(len(frames), self.window_margin)
        kernel = np.ones(margin) / margin
        cx_smooth = np.convolve(cx_arr, kernel, mode="same")
        cy_smooth = np.convolve(cy_arr, kernel, mode="same")

        crop_h = self.crop_height
        crop_w = self.crop_width
        target = self.crop_height * 2   # 96

        sequence = []
        for frame, cx, cy in zip(frames, cx_smooth, cy_smooth):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape
            y0 = int(np.clip(round(cy) - crop_h, 0, H - target))
            x0 = int(np.clip(round(cx) - crop_w, 0, W - target))
            patch = gray[y0: y0 + target, x0: x0 + target]
            if patch.shape != (target, target):
                patch = cv2.resize(patch, (target, target))
            sequence.append(patch)

        return np.array(sequence)   # [T, 96, 96] uint8

    # ──────────────────────────────────────────────────── public API ──────────
    def extract_from_file(self, video_path: str):
        """Extract mouth ROI tensor from a video file.

        Args:
            video_path: Path to an MP4 (or any cv2-readable) video file.

        Returns:
            torch.Tensor of shape [T, 1, 96, 96], float32 in [0, 1]
            (numpy ndarray with same shape/dtype if torch is not installed).
            T = number of video frames.

        Raises:
            RuntimeError if no frames could be read.
        """
        frames = self._read_frames(video_path)
        if not frames:
            raise RuntimeError(f"Could not read any frames from {video_path!r}")
        return self.extract_from_frames(frames)

    def extract_from_frames(self, frames: list[np.ndarray]):
        """Extract mouth ROI from a list of BGR numpy frames (H×W×3 uint8).

        Returns:
            torch.Tensor [T, 1, 96, 96] float32 [0, 1]
            (numpy ndarray with same shape/dtype if torch is not installed).
        """
        if self.backend == "dlib":
            grey_seq = self._crop_sequence_dlib(frames)
            if grey_seq is None:
                raise RuntimeError("dlib: landmark detection failed on all frames")
            # grey_seq is [T, 96, 96] uint8 (already greyscale via warp)
        else:
            grey_seq = self._crop_sequence_haar(frames)   # [T, 96, 96] uint8

        # Ensure greyscale (in case warp returned colour)
        if grey_seq.ndim == 4:      # [T, H, W, C]
            grey_seq = grey_seq.mean(axis=-1).astype(np.uint8)

        return _frames_to_tensor(list(grey_seq))   # [T, 1, 96, 96]

    # ──────────────────────────────────────────────────── helpers ─────────────
    @staticmethod
    def _read_frames(video_path: str) -> list[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
