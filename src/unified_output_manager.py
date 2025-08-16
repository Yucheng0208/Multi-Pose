# unified_output_manager.py
# 統一輸出管理器：標準化所有模組的輸出格式
# 輸出格式：<person_id><keypoints><coor_x><coor_y><confidence>
# 檔案格式：<modelname>_framenum.npy

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import pickle


class ModalityType(Enum):
    """模態類型"""
    POSE = "pose"
    FACE = "face"
    HAND = "hand"
    MULTIMODAL = "multimodal"


@dataclass
class UnifiedKeypoint:
    """統一的關鍵點資料結構"""
    person_id: int
    keypoint_name: str
    coor_x: float
    coor_y: float
    confidence: float
    modality: ModalityType
    frame_num: int
    timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "person_id": self.person_id,
            "keypoint_name": self.keypoint_name,
            "coor_x": self.coor_x,
            "coor_y": self.coor_y,
            "confidence": self.confidence,
            "modality": self.modality.value,
            "frame_num": self.frame_num,
            "timestamp": self.timestamp
        }
    
    def to_array(self) -> np.ndarray:
        """轉換為numpy陣列格式"""
        return np.array([
            self.person_id,
            hash(self.keypoint_name) % 10000,  # 將字串轉換為數字ID
            self.coor_x,
            self.coor_y,
            self.confidence,
            self.frame_num
        ], dtype=np.float32)


class UnifiedOutputManager:
    """統一輸出管理器"""
    
    def __init__(self, output_dir: Union[str, Path] = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 儲存每幀的資料
        self.frame_data: Dict[int, List[UnifiedKeypoint]] = {}
        self.current_frame = 0
        
        # 模態對應表
        self.modality_mapping = {
            "pose": ModalityType.POSE,
            "face": ModalityType.FACE,
            "hand": ModalityType.HAND,
            "multimodal": ModalityType.MULTIMODAL
        }
        
        logging.info(f"統一輸出管理器已初始化，輸出目錄：{self.output_dir}")
    
    def add_pose_data(self, frame_num: int, pose_results: List[Dict[str, Any]]) -> None:
        """添加骨架檢測資料"""
        if frame_num not in self.frame_data:
            self.frame_data[frame_num] = []
        
        for result in pose_results:
            # 解析骨架檢測結果
            person_id = result.get("id", 0)
            keypoints = result.get("keypoints", [])
            
            if isinstance(keypoints, np.ndarray):
                # 處理numpy陣列格式的關鍵點
                for i, kp in enumerate(keypoints):
                    if len(kp) >= 3:  # x, y, confidence
                        keypoint_name = f"pose_kp_{i}"
                        keypoint = UnifiedKeypoint(
                            person_id=person_id,
                            keypoint_name=keypoint_name,
                            coor_x=float(kp[0]),
                            coor_y=float(kp[1]),
                            confidence=float(kp[2]),
                            modality=ModalityType.POSE,
                            frame_num=frame_num
                        )
                        self.frame_data[frame_num].append(keypoint)
            else:
                # 處理字典格式的關鍵點
                for kp_name, kp_data in keypoints.items():
                    if isinstance(kp_data, dict) and "x" in kp_data and "y" in kp_data:
                        keypoint = UnifiedKeypoint(
                            person_id=person_id,
                            keypoint_name=f"pose_{kp_name}",
                            coor_x=float(kp_data["x"]),
                            coor_y=float(kp_data["y"]),
                            confidence=float(kp_data.get("confidence", 1.0)),
                            modality=ModalityType.POSE,
                            frame_num=frame_num
                        )
                        self.frame_data[frame_num].append(keypoint)
    
    def add_face_data(self, frame_num: int, face_results: List[Dict[str, Any]]) -> None:
        """添加臉部檢測資料"""
        if frame_num not in self.frame_data:
            self.frame_data[frame_num] = []
        
        for result in face_results:
            person_id = result.get("id", 0)
            landmarks = result.get("landmarks", [])
            
            if isinstance(landmarks, np.ndarray):
                # 處理numpy陣列格式的臉部關鍵點
                for i, lm in enumerate(landmarks):
                    if len(lm) >= 3:  # x, y, z
                        keypoint_name = f"face_lm_{i}"
                        keypoint = UnifiedKeypoint(
                            person_id=person_id,
                            keypoint_name=keypoint_name,
                            coor_x=float(lm[0]),
                            coor_y=float(lm[1]),
                            confidence=float(lm[2]) if len(lm) > 2 else 1.0,
                            modality=ModalityType.FACE,
                            frame_num=frame_num
                        )
                        self.frame_data[frame_num].append(keypoint)
            else:
                # 處理字典格式的臉部關鍵點
                for lm_name, lm_data in landmarks.items():
                    if isinstance(lm_data, dict) and "x" in lm_data and "y" in lm_data:
                        keypoint = UnifiedKeypoint(
                            person_id=person_id,
                            keypoint_name=f"face_{lm_name}",
                            coor_x=float(lm_data["x"]),
                            coor_y=float(lm_data["y"]),
                            confidence=float(lm_data.get("confidence", 1.0)),
                            modality=ModalityType.FACE,
                            frame_num=frame_num
                        )
                        self.frame_data[frame_num].append(keypoint)
    
    def add_hand_data(self, frame_num: int, hand_results: List[Dict[str, Any]]) -> None:
        """添加手部檢測資料"""
        if frame_num not in self.frame_data:
            self.frame_data[frame_num] = []
        
        for result in hand_results:
            person_id = result.get("id", 0)
            landmarks = result.get("landmarks", [])
            
            if isinstance(landmarks, np.ndarray):
                # 處理numpy陣列格式的手部關鍵點
                for i, lm in enumerate(landmarks):
                    if len(lm) >= 3:  # x, y, z
                        keypoint_name = f"hand_lm_{i}"
                        keypoint = UnifiedKeypoint(
                            person_id=person_id,
                            keypoint_name=keypoint_name,
                            coor_x=float(lm[0]),
                            coor_y=float(lm[1]),
                            confidence=float(lm[2]) if len(lm) > 2 else 1.0,
                            modality=ModalityType.HAND,
                            frame_num=frame_num
                        )
                        self.frame_data[frame_num].append(keypoint)
            else:
                # 處理字典格式的手部關鍵點
                for lm_name, lm_data in landmarks.items():
                    if isinstance(lm_data, dict) and "x" in lm_data and "y" in lm_data:
                        keypoint = UnifiedKeypoint(
                            person_id=person_id,
                            keypoint_name=f"hand_{lm_name}",
                            coor_x=float(lm_data["x"]),
                            coor_y=float(lm_data["y"]),
                            confidence=float(lm_data.get("confidence", 1.0)),
                            modality=ModalityType.HAND,
                            frame_num=frame_num
                        )
                        self.frame_data[frame_num].append(keypoint)
    
    def save_frame_data(self, frame_num: int, model_name: str) -> Path:
        """保存單幀資料為npy檔案"""
        if frame_num not in self.frame_data or not self.frame_data[frame_num]:
            logging.warning(f"幀 {frame_num} 沒有資料可保存")
            return None
        
        # 轉換為numpy陣列格式
        frame_keypoints = self.frame_data[frame_num]
        data_array = np.array([kp.to_array() for kp in frame_keypoints], dtype=np.float32)
        
        # 保存為npy檔案
        filename = f"{model_name}_{frame_num:06d}.npy"
        filepath = self.output_dir / filename
        np.save(filepath, data_array)
        
        # 同時保存元資料（JSON格式）
        metadata = {
            "frame_num": frame_num,
            "model_name": model_name,
            "num_keypoints": len(frame_keypoints),
            "modalities": list(set(kp.modality.value for kp in frame_keypoints)),
            "timestamp": frame_keypoints[0].timestamp if frame_keypoints else None,
            "keypoint_names": [kp.keypoint_name for kp in frame_keypoints]
        }
        
        metadata_file = filepath.with_suffix('.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logging.info(f"已保存幀 {frame_num} 資料到 {filename} ({len(frame_keypoints)} 個關鍵點)")
        return filepath
    
    def save_all_frames(self, model_name: str) -> List[Path]:
        """保存所有幀的資料"""
        saved_files = []
        for frame_num in sorted(self.frame_data.keys()):
            if self.frame_data[frame_num]:  # 只保存有資料的幀
                filepath = self.save_frame_data(frame_num, model_name)
                if filepath:
                    saved_files.append(filepath)
        
        logging.info(f"已保存 {len(saved_files)} 個幀檔案")
        return saved_files
    
    def save_summary_csv(self, model_name: str) -> Path:
        """保存摘要CSV檔案"""
        all_keypoints = []
        for frame_num, keypoints in self.frame_data.items():
            for kp in keypoints:
                all_keypoints.append(kp.to_dict())
        
        if not all_keypoints:
            logging.warning("沒有資料可保存為CSV")
            return None
        
        df = pd.DataFrame(all_keypoints)
        csv_file = self.output_dir / f"{model_name}_summary.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logging.info(f"已保存摘要CSV到 {csv_file} ({len(df)} 行)")
        return csv_file
    
    def get_frame_statistics(self) -> Dict[str, Any]:
        """獲取幀統計資訊"""
        total_frames = len(self.frame_data)
        total_keypoints = sum(len(keypoints) for keypoints in self.frame_data.values())
        
        modality_counts = {}
        for keypoints in self.frame_data.values():
            for kp in keypoints:
                modality = kp.modality.value
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        return {
            "total_frames": total_frames,
            "total_keypoints": total_keypoints,
            "modality_counts": modality_counts,
            "frames_with_data": [f for f, kps in self.frame_data.items() if kps]
        }
    
    def clear_frame_data(self, frame_num: int) -> None:
        """清除指定幀的資料"""
        if frame_num in self.frame_data:
            del self.frame_data[frame_num]
            logging.debug(f"已清除幀 {frame_num} 的資料")
    
    def clear_all_data(self) -> None:
        """清除所有資料"""
        self.frame_data.clear()
        self.current_frame = 0
        logging.info("已清除所有幀資料")
    
    def load_frame_data(self, filepath: Path) -> np.ndarray:
        """載入npy檔案資料"""
        try:
            data = np.load(filepath)
            logging.info(f"已載入 {filepath.name}，資料形狀：{data.shape}")
            return data
        except Exception as e:
            logging.error(f"載入 {filepath.name} 失敗：{e}")
            return None
    
    def export_to_pickle(self, model_name: str) -> Path:
        """匯出為pickle格式（包含完整資料結構）"""
        pickle_file = self.output_dir / f"{model_name}_complete.pkl"
        
        export_data = {
            "frame_data": self.frame_data,
            "statistics": self.get_frame_statistics(),
            "model_name": model_name,
            "export_timestamp": pd.Timestamp.now().isoformat()
        }
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(export_data, f)
        
        logging.info(f"已匯出完整資料到 {pickle_file}")
        return pickle_file


def create_unified_output_manager(output_dir: Union[str, Path] = "output") -> UnifiedOutputManager:
    """創建統一輸出管理器實例"""
    return UnifiedOutputManager(output_dir)
