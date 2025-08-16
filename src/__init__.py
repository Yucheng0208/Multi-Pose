# YOLOv11-pose 骨架偵測與追蹤記錄器套件
# 支援直接執行與模組 import 使用

from .pose_recorder import PoseRecorder, run_inference, KEYPOINTS_C17
from .face_recorder import FaceMeshRecorder, FACE_MESH_POINTS
from .hand_recorder import HandRecorder, HAND_LANDMARKS, HandGesture
from .multimodal_data import MultimodalDataIntegrator, MultimodalAnalyzer, ModalityType
from .multimodal_processor import MultimodalProcessor

__version__ = "1.0.0"
__all__ = [
    "PoseRecorder", "run_inference", "KEYPOINTS_C17",
    "FaceMeshRecorder", "FACE_MESH_POINTS",
    "HandRecorder", "HAND_LANDMARKS", "HandGesture",
    "MultimodalDataIntegrator", "MultimodalAnalyzer", "ModalityType",
    "MultimodalProcessor"
]