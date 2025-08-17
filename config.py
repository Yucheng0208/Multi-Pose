# -*- coding: utf-8 -*-
from pathlib import Path

# 根目錄
ROOT = Path(__file__).resolve().parent

# 路徑
OUTPUT_DIR  = ROOT / "outputs"
VIDEO_DIR   = OUTPUT_DIR / "video"
JSON_DIR    = OUTPUT_DIR / "json"
CSV_DIR     = OUTPUT_DIR / "csv"
NPY_DIR     = OUTPUT_DIR / "npy"
TENSOR_DIR  = OUTPUT_DIR / "tensor"

# I/O 與視覺化
WRITE_VIDEO   = True
DRAW_BOX_ID   = True
DRAW_FPS      = True
DRAW_KEYPOINT = True       # 是否在輸出影像上畫 keypoints

# 存檔開關（可被 CLI 覆蓋）
SAVE_JSON   = True
SAVE_CSV    = True
SAVE_NPY    = False
SAVE_TENSOR = False

# 來源
YOUTUBE_ENABLE = True
VIDEO_FPS_FALLBACK = 30.0

# YOLO/Tracker
YOLO_WEIGHTS = ROOT / "models" / "yolo11n.pt"  # 你已下載
YOLO_CONF    = 0.5
YOLO_IOU     = 0.5
YOLO_DEVICE  = "0"                  # RTX3090 => "0"; 若要CPU改 "cpu"
YOLO_TRACKER = "bytetrack.yaml"     # 務必是 .yaml/.yml
YOLO_PERSON_CLASS = 0               # COCO: person

# Mediapipe
USE_FACE  = True
USE_POSE  = True
USE_HANDS = True
FRONT_ONLY = True

# 臉點採 468（為了總點數 0..542）；若想用 478，改 False
FACE_USE_468 = True

KP_FACE_COUNT = 1
KP_POSE_COUNT = 1

# 模型路徑（你已提供）
FACE_TASK = ROOT / "models" / "face_landmarker.task"
POSE_TASK = ROOT / "models" / "pose_landmarker_full.task"
HAND_TASK = ROOT / "models" / "hand_landmarker.task"

# 流水號位數
ID_PAD = 14
