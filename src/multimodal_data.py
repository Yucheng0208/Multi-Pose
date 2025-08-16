# multimodal_data.py
# 多模態資料結構與整合模組
# 提供臉部、姿態、手部資料的統一介面和整合功能

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime

# 導入各模組的資料類型
try:
    from .face_recorder import FaceMeshRecorder, FACE_MESH_POINTS
    from .pose_recorder import PoseRecorder, KEYPOINTS_C17
    from .hand_recorder import HandRecorder, HAND_LANDMARKS, HandGesture
except ImportError:
    # 如果無法導入，定義基本類型
    FACE_MESH_POINTS = []
    KEYPOINTS_C17 = []
    HAND_LANDMARKS = []
    HandGesture = Enum('HandGesture', ['UNKNOWN'])


class ModalityType(Enum):
    """模態類型"""
    FACE = "face"
    POSE = "pose"
    HAND = "hand"


class DataQuality(Enum):
    """資料品質等級"""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 70-89%
    FAIR = "fair"           # 50-69%
    POOR = "poor"           # <50%


@dataclass
class FrameData:
    """單幀的多模態資料"""
    frame_id: int
    timestamp: float
    face_data: Optional[Dict[str, Any]] = None
    pose_data: Optional[Dict[str, Any]] = None
    hand_data: Optional[Dict[str, Any]] = None
    
    def has_modality(self, modality: ModalityType) -> bool:
        """檢查是否包含特定模態的資料"""
        if modality == ModalityType.FACE:
            return self.face_data is not None
        elif modality == ModalityType.POSE:
            return self.pose_data is not None
        elif modality == ModalityType.HAND:
            return self.hand_data is not None
        return False
    
    def get_modality_data(self, modality: ModalityType) -> Optional[Dict[str, Any]]:
        """獲取特定模態的資料"""
        if modality == ModalityType.FACE:
            return self.face_data
        elif modality == ModalityType.POSE:
            return self.pose_data
        elif modality == ModalityType.HAND:
            return self.hand_data
        return None


@dataclass
class MultimodalSession:
    """多模態會話資料"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    source_path: Optional[Path] = None
    frame_data: List[FrameData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_frame(self, frame_data: FrameData) -> None:
        """添加幀資料"""
        self.frame_data.append(frame_data)
    
    def get_frame_count(self) -> int:
        """獲取總幀數"""
        return len(self.frame_data)
    
    def get_modality_coverage(self) -> Dict[ModalityType, float]:
        """計算各模態的覆蓋率"""
        try:
            if not self.frame_data:
                return {modality: 0.0 for modality in ModalityType}
            
            total_frames = len(self.frame_data)
            coverage = {}
            
            for modality in ModalityType:
                covered_frames = sum(1 for frame in self.frame_data if frame.has_modality(modality))
                coverage[modality] = covered_frames / total_frames
            
            return coverage
        except Exception as e:
            logging.error("計算模態覆蓋率時發生錯誤: %s", e)
            # 返回預設值
            return {modality: 0.0 for modality in ModalityType}
    
    def get_quality_score(self) -> DataQuality:
        """計算整體資料品質分數"""
        try:
            coverage = self.get_modality_coverage()
            avg_coverage = sum(coverage.values()) / len(coverage)
            
            if avg_coverage >= 0.9:
                return DataQuality.EXCELLENT
            elif avg_coverage >= 0.7:
                return DataQuality.GOOD
            elif avg_coverage >= 0.5:
                return DataQuality.FAIR
            else:
                return DataQuality.POOR
        except Exception as e:
            logging.error("計算品質分數時發生錯誤: %s", e)
            return DataQuality.POOR


class MultimodalDataIntegrator:
    """多模態資料整合器"""
    
    def __init__(self):
        self.sessions: Dict[str, MultimodalSession] = {}
        self.current_session: Optional[MultimodalSession] = None
        
    def create_session(self, session_id: str, source_path: Optional[Path] = None) -> MultimodalSession:
        """創建新的多模態會話"""
        session = MultimodalSession(
            session_id=session_id,
            start_time=datetime.now(),
            source_path=source_path
        )
        self.sessions[session_id] = session
        self.current_session = session
        logging.info("創建新會話: %s", session_id)
        return session
    
    def add_frame_data(
        self,
        frame_id: int,
        timestamp: float,
        face_data: Optional[Dict[str, Any]] = None,
        pose_data: Optional[Dict[str, Any]] = None,
        hand_data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> None:
        """添加幀資料到指定會話"""
        if session_id is None:
            if self.current_session is None:
                raise ValueError("沒有活動會話，請先創建會話")
            session = self.current_session
        else:
            if session_id not in self.sessions:
                raise ValueError(f"會話不存在: {session_id}")
            session = self.sessions[session_id]
        
        frame_data = FrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            face_data=face_data,
            pose_data=pose_data,
            hand_data=hand_data
        )
        
        session.add_frame(frame_data)
    
    def get_session(self, session_id: str) -> Optional[MultimodalSession]:
        """獲取指定會話"""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[str]:
        """列出所有會話 ID"""
        return list(self.sessions.keys())
    
    def export_session_to_csv(
        self,
        session_id: str,
        output_dir: Path,
        separate_modalities: bool = True
    ) -> List[Path]:
        """將會話資料匯出為 CSV 檔案"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"會話不存在: {session_id}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        exported_files = []
        
        if separate_modalities:
            # 分別匯出各模態的資料
            for modality in ModalityType:
                modality_data = []
                for frame in session.frame_data:
                    data = frame.get_modality_data(modality)
                    if data:
                        # 確保 data 可被展開為欄位
                        extra = data if isinstance(data, dict) else {'data': data}
                        modality_data.append({
                            'frame_id': frame.frame_id,
                            'timestamp': frame.timestamp,
                            **extra
                        })
                
                if modality_data:
                    df = pd.DataFrame(modality_data)
                    output_file = output_dir / f"{session_id}_{modality.value}.csv"
                    df.to_csv(output_file, index=False)
                    exported_files.append(output_file)
        else:
            # 匯出整合的資料
            integrated_data = []
            for frame in session.frame_data:
                frame_record = {
                    'frame_id': frame.frame_id,
                    'timestamp': frame.timestamp,
                    'has_face': frame.has_modality(ModalityType.FACE),
                    'has_pose': frame.has_modality(ModalityType.POSE),
                    'has_hand': frame.has_modality(ModalityType.HAND)
                }
                
                # 添加臉部資料
                if frame.face_data:
                    if isinstance(frame.face_data, dict):
                        for key, value in frame.face_data.items():
                            frame_record[f"face_{key}"] = value
                    else:
                        frame_record["face_data"] = frame.face_data
                
                # 添加姿態資料
                if frame.pose_data:
                    if isinstance(frame.pose_data, dict):
                        for key, value in frame.pose_data.items():
                            frame_record[f"pose_{key}"] = value
                    else:
                        frame_record["pose_data"] = frame.pose_data
                
                # 添加手部資料
                if frame.hand_data:
                    if isinstance(frame.hand_data, dict):
                        for key, value in frame.hand_data.items():
                            frame_record[f"hand_{key}"] = value
                    else:
                        frame_record["hand_data"] = frame.hand_data
                
                integrated_data.append(frame_record)
            
            if integrated_data:
                df = pd.DataFrame(integrated_data)
                output_file = output_dir / f"{session_id}_integrated.csv"
                df.to_csv(output_file, index=False)
                exported_files.append(output_file)
        
        logging.info("會話 %s 資料已匯出至: %s", session_id, output_dir)
        return exported_files
    
    def get_statistics(self, session_id: str) -> Dict[str, Any]:
        """獲取會話統計資訊"""
        try:
            session = self.get_session(session_id)
            if not session:
                return {}
            
            coverage = session.get_modality_coverage()
            quality = session.get_quality_score()
            
            stats = {
                'session_id': session_id,
                'frame_count': session.get_frame_count(),
                'duration_seconds': (session.end_time - session.start_time).total_seconds() if session.end_time else None,
                'modality_coverage': {mod.value: cov for mod, cov in coverage.items()},
                'quality_score': quality.value,
                'source_path': str(session.source_path) if session.source_path else None
            }
            
            return stats
        except Exception as e:
            logging.error("獲取會話統計資訊時發生錯誤: %s", e)
            return {
                'session_id': session_id,
                'error': str(e),
                'frame_count': 0,
                'duration_seconds': None,
                'modality_coverage': {},
                'quality_score': 'unknown',
                'source_path': None
            }
    
    def close_session(self, session_id: str) -> None:
        """關閉會話"""
        if session_id in self.sessions:
            self.sessions[session_id].end_time = datetime.now()
            if self.current_session and self.current_session.session_id == session_id:
                self.current_session = None
            logging.info("會話已關閉: %s", session_id)


class MultimodalAnalyzer:
    """多模態資料分析器"""
    
    def __init__(self, integrator: MultimodalDataIntegrator):
        self.integrator = integrator
    
    def analyze_cross_modality_correlation(
        self,
        session_id: str,
        modality1: ModalityType,
        modality2: ModalityType
    ) -> Dict[str, Any]:
        """分析兩個模態之間的相關性"""
        session = self.integrator.get_session(session_id)
        if not session:
            return {}
        
        # 獲取兩個模態都有資料的幀
        common_frames = [
            frame for frame in session.frame_data
            if frame.has_modality(modality1) and frame.has_modality(modality2)
        ]
        
        if len(common_frames) < 2:
            return {'correlation': 0.0, 'common_frames': len(common_frames)}
        
        # 這裡可以實現更複雜的相關性分析
        # 目前返回基本統計
        return {
            'correlation': len(common_frames) / session.get_frame_count(),
            'common_frames': len(common_frames),
            'total_frames': session.get_frame_count()
        }
    
    def detect_gesture_pose_patterns(
        self,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """檢測手部動作與身體姿態的組合模式"""
        session = self.integrator.get_session(session_id)
        if not session:
            return []
        
        patterns = []
        
        for frame in session.frame_data:
            if frame.has_modality(ModalityType.HAND) and frame.has_modality(ModalityType.POSE):
                # 分析手部動作與身體姿態的關係
                hand_data = frame.hand_data
                pose_data = frame.pose_data
                
                # 這裡可以實現更複雜的模式檢測邏輯
                pattern = {
                    'frame_id': frame.frame_id,
                    'timestamp': frame.timestamp,
                    'hand_gesture': hand_data.get('gesture', 'unknown'),
                    'pose_type': pose_data.get('posture', 'unknown'),
                    'pattern_type': 'hand_pose_combination'
                }
                
                patterns.append(pattern)
        
        return patterns
    
    def generate_summary_report(self, session_id: str) -> Dict[str, Any]:
        """生成會話摘要報告"""
        session = self.integrator.get_session(session_id)
        if not session:
            return {}
        
        try:
            stats = self.integrator.get_statistics(session_id)
            coverage = session.get_modality_coverage()
            
            # 確保 coverage 是字典類型
            if not isinstance(coverage, dict):
                logging.warning("coverage 不是字典類型，重置為空字典: %s", type(coverage))
                coverage = {modality: 0.0 for modality in ModalityType}
            
            report = {
                'session_summary': stats,
                'modality_analysis': {
                    'face': {
                        'coverage': coverage.get(ModalityType.FACE, 0.0),
                        'points_detected': len(FACE_MESH_POINTS) if FACE_MESH_POINTS else 0
                    },
                    'pose': {
                        'coverage': coverage.get(ModalityType.POSE, 0.0),
                        'keypoints_detected': len(KEYPOINTS_C17) if KEYPOINTS_C17 else 0
                    },
                    'hand': {
                        'coverage': coverage.get(ModalityType.HAND, 0.0),
                        'landmarks_detected': len(HAND_LANDMARKS) if HAND_LANDMARKS else 0
                    }
                },
                'recommendations': self._generate_recommendations(coverage, stats)
            }
            
            return report
        except Exception as e:
            logging.error("生成摘要報告時發生錯誤: %s", e)
            # 返回基本的錯誤報告
            return {
                'error': str(e),
                'session_summary': {},
                'modality_analysis': {
                    'face': {'coverage': 0.0, 'points_detected': 0},
                    'pose': {'coverage': 0.0, 'keypoints_detected': 0},
                    'hand': {'coverage': 0.0, 'landmarks_detected': 0}
                },
                'recommendations': ['報告生成失敗，請檢查資料完整性']
            }
    
    def _generate_recommendations(
        self,
        coverage: Dict[ModalityType, float],
        stats: Dict[str, Any]
    ) -> List[str]:
        """生成改進建議"""
        recommendations = []
        
        try:
            # 確保 coverage 是字典類型
            if not isinstance(coverage, dict):
                logging.warning("_generate_recommendations: coverage 不是字典類型: %s", type(coverage))
                return ["無法生成建議：資料格式錯誤"]
            
            # 確保 stats 是字典類型
            if not isinstance(stats, dict):
                logging.warning("_generate_recommendations: stats 不是字典類型: %s", type(stats))
                stats = {}
            
            for modality, cov in coverage.items():
                if cov < 0.5:
                    recommendations.append(f"{modality.value} 模態覆蓋率過低 ({cov:.1%})，建議檢查檢測參數")
                elif cov < 0.8:
                    recommendations.append(f"{modality.value} 模態覆蓋率中等 ({cov:.1%})，可考慮優化檢測設定")
            
            if stats.get('frame_count', 0) < 100:
                recommendations.append("幀數較少，建議收集更多資料以提高分析準確性")
                
        except Exception as e:
            logging.error("生成建議時發生錯誤: %s", e)
            recommendations.append(f"生成建議失敗: {str(e)}")
        
        return recommendations


# 便利函數
def create_multimodal_session(
    session_id: str,
    source_path: Optional[Path] = None
) -> MultimodalDataIntegrator:
    """創建多模態資料整合器並開始新會話"""
    integrator = MultimodalDataIntegrator()
    integrator.create_session(session_id, source_path)
    return integrator


def load_multimodal_data_from_csv(
    face_csv: Optional[Path] = None,
    pose_csv: Optional[Path] = None,
    hand_csv: Optional[Path] = None
) -> MultimodalDataIntegrator:
    """從 CSV 檔案載入多模態資料"""
    integrator = MultimodalDataIntegrator()
    
    # 這裡可以實現從 CSV 檔案載入資料的邏輯
    # 目前返回空的整合器
    
    return integrator
