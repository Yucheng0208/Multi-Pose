#!/usr/bin/env python3
"""
身體分析模組
提供身體姿態、動作和健康指標分析功能
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math
from enum import Enum

from .pose_visualizer import PosePerson, Keypoint


class BodyPosture(Enum):
    """身體姿態類型"""
    STANDING = "standing"
    SITTING = "sitting"
    WALKING = "walking"
    RUNNING = "running"
    LYING = "lying"
    UNKNOWN = "unknown"


class BodySymmetry(Enum):
    """身體對稱性評估"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class BodyMetrics:
    """身體測量指標"""
    height_pixels: float
    height_cm: Optional[float]
    shoulder_width: float
    hip_width: float
    arm_length: float
    leg_length: float
    torso_height: float
    head_height: float
    
    def get_bmi_estimate(self, estimated_weight_kg: float = 70.0) -> Optional[float]:
        """估算 BMI（需要體重估計值）"""
        if self.height_cm:
            height_m = self.height_cm / 100.0
            return estimated_weight_kg / (height_m * height_m)
        return None


@dataclass
class PostureAnalysis:
    """姿態分析結果"""
    posture_type: BodyPosture
    confidence: float
    symmetry_score: float
    symmetry_level: BodySymmetry
    spine_alignment: float
    shoulder_level: float
    hip_level: float
    head_position: str
    recommendations: List[str]


class BodyAnalyzer:
    """身體分析器"""
    
    def __init__(self, pixel_to_cm: Optional[float] = None):
        self.pixel_to_cm = pixel_to_cm
        self.posture_history: List[BodyPosture] = []
        self.metrics_history: List[BodyMetrics] = []
    
    def analyze_body_metrics(self, pose_person: PosePerson) -> BodyMetrics:
        """分析身體測量指標"""
        if not pose_person.keypoints:
            return BodyMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        # 獲取關鍵點
        nose = pose_person.get_keypoint_by_name("nose")
        left_shoulder = pose_person.get_keypoint_by_name("left_shoulder")
        right_shoulder = pose_person.get_keypoint_by_name("right_shoulder")
        left_hip = pose_person.get_keypoint_by_name("left_hip")
        right_hip = pose_person.get_keypoint_by_name("right_hip")
        left_elbow = pose_person.get_keypoint_by_name("left_elbow")
        left_wrist = pose_person.get_keypoint_by_name("left_wrist")
        left_knee = pose_person.get_keypoint_by_name("left_knee")
        left_ankle = pose_person.get_keypoint_by_name("left_ankle")
        
        # 計算身體高度（從頭頂到腳踝）
        if nose and left_ankle:
            height_pixels = abs(nose.y - left_ankle.y)
            height_cm = height_pixels * self.pixel_to_cm if self.pixel_to_cm else None
        else:
            height_pixels = 0
            height_cm = None
        
        # 計算肩膀寬度
        if left_shoulder and right_shoulder:
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        else:
            shoulder_width = 0
        
        # 計算髖部寬度
        if left_hip and right_hip:
            hip_width = abs(left_hip.x - right_hip.x)
        else:
            hip_width = 0
        
        # 計算手臂長度
        if left_shoulder and left_elbow and left_wrist:
            arm_length = (abs(left_shoulder.x - left_elbow.x) + 
                         abs(left_shoulder.y - left_elbow.y)) ** 0.5 + \
                        (abs(left_elbow.x - left_wrist.x) + 
                         abs(left_elbow.y - left_wrist.y)) ** 0.5
        else:
            arm_length = 0
        
        # 計算腿部長度
        if left_hip and left_knee and left_ankle:
            leg_length = (abs(left_hip.x - left_knee.x) + 
                         abs(left_hip.y - left_knee.y)) ** 0.5 + \
                        (abs(left_knee.x - left_ankle.x) + 
                         abs(left_knee.y - left_ankle.y)) ** 0.5
        else:
            leg_length = 0
        
        # 計算軀幹高度
        if left_shoulder and left_hip:
            torso_height = abs(left_shoulder.y - left_hip.y)
        else:
            torso_height = 0
        
        # 計算頭部高度（估算）
        if nose and left_shoulder:
            head_height = abs(nose.y - left_shoulder.y) * 0.3  # 頭部約佔頸部到肩膀距離的 30%
        else:
            head_height = 0
        
        return BodyMetrics(
            height_pixels=height_pixels,
            height_cm=height_cm,
            shoulder_width=shoulder_width,
            hip_width=hip_width,
            arm_length=arm_length,
            leg_length=leg_length,
            torso_height=torso_height,
            head_height=head_height
        )
    
    def analyze_posture(self, pose_person: PosePerson) -> PostureAnalysis:
        """分析身體姿態"""
        if not pose_person.keypoints:
            return PostureAnalysis(
                posture_type=BodyPosture.UNKNOWN,
                confidence=0.0,
                symmetry_score=0.0,
                symmetry_level=BodySymmetry.POOR,
                spine_alignment=0.0,
                shoulder_level=0.0,
                hip_level=0.0,
                head_position="unknown",
                recommendations=[]
            )
        
        # 獲取關鍵點
        nose = pose_person.get_keypoint_by_name("nose")
        left_shoulder = pose_person.get_keypoint_by_name("left_shoulder")
        right_shoulder = pose_person.get_keypoint_by_name("right_shoulder")
        left_hip = pose_person.get_keypoint_by_name("left_hip")
        right_hip = pose_person.get_keypoint_by_name("right_hip")
        left_knee = pose_person.get_keypoint_by_name("left_knee")
        right_knee = pose_person.get_keypoint_by_name("right_knee")
        
        # 分析姿態類型
        posture_type, confidence = self._classify_posture(pose_person)
        
        # 分析身體對稱性
        symmetry_score = self._analyze_symmetry(pose_person)
        symmetry_level = self._get_symmetry_level(symmetry_score)
        
        # 分析脊椎對齊
        spine_alignment = self._analyze_spine_alignment(pose_person)
        
        # 分析肩膀水平度
        shoulder_level = self._analyze_shoulder_level(left_shoulder, right_shoulder)
        
        # 分析髖部水平度
        hip_level = self._analyze_hip_level(left_hip, right_hip)
        
        # 分析頭部位置
        head_position = self._analyze_head_position(nose, left_shoulder, right_shoulder)
        
        # 生成建議
        recommendations = self._generate_recommendations(
            posture_type, symmetry_score, spine_alignment, shoulder_level, hip_level
        )
        
        return PostureAnalysis(
            posture_type=posture_type,
            confidence=confidence,
            symmetry_score=symmetry_score,
            symmetry_level=symmetry_level,
            spine_alignment=spine_alignment,
            shoulder_level=shoulder_level,
            hip_level=hip_level,
            head_position=head_position,
            recommendations=recommendations
        )
    
    def _classify_posture(self, pose_person: PosePerson) -> Tuple[BodyPosture, float]:
        """分類身體姿態"""
        # 獲取關鍵點
        nose = pose_person.get_keypoint_by_name("nose")
        left_hip = pose_person.get_keypoint_by_name("left_hip")
        right_hip = pose_person.get_keypoint_by_name("right_hip")
        left_knee = pose_person.get_keypoint_by_name("left_knee")
        right_knee = pose_person.get_keypoint_by_name("right_knee")
        left_ankle = pose_person.get_keypoint_by_name("left_ankle")
        right_ankle = pose_person.get_keypoint_by_name("right_ankle")
        
        if not all([nose, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
            return BodyPosture.UNKNOWN, 0.0
        
        # 計算身體各部分的相對位置
        body_height = abs(nose.y - (left_ankle.y + right_ankle.y) / 2)
        hip_y = (left_hip.y + right_hip.y) / 2
        knee_y = (left_knee.y + right_knee.y) / 2
        
        # 姿態分類邏輯
        if body_height > 0:
            # 計算身體傾斜角度
            body_angle = math.atan2(abs(nose.x - (left_hip.x + right_hip.x) / 2), body_height)
            body_angle_deg = math.degrees(body_angle)
            
            # 檢查是否為坐姿（膝蓋彎曲）
            knee_bend = abs(hip_y - knee_y) < body_height * 0.3
            
            if knee_bend and body_angle_deg < 30:
                return BodyPosture.SITTING, 0.8
            elif body_angle_deg < 15:
                return BodyPosture.STANDING, 0.9
            elif body_angle_deg < 45:
                return BodyPosture.WALKING, 0.7
            elif body_angle_deg < 60:
                return BodyPosture.RUNNING, 0.6
            else:
                return BodyPosture.LYING, 0.5
        
        return BodyPosture.UNKNOWN, 0.0
    
    def _analyze_symmetry(self, pose_person: PosePerson) -> float:
        """分析身體對稱性"""
        # 獲取左右對稱的關鍵點
        left_shoulder = pose_person.get_keypoint_by_name("left_shoulder")
        right_shoulder = pose_person.get_keypoint_by_name("right_shoulder")
        left_hip = pose_person.get_keypoint_by_name("left_hip")
        right_hip = pose_person.get_keypoint_by_name("right_hip")
        left_eye = pose_person.get_keypoint_by_name("left_eye")
        right_eye = pose_person.get_keypoint_by_name("right_eye")
        
        symmetry_scores = []
        
        # 肩膀對稱性
        if left_shoulder and right_shoulder:
            shoulder_symmetry = 1.0 - abs(left_shoulder.y - right_shoulder.y) / max(left_shoulder.y, right_shoulder.y)
            symmetry_scores.append(shoulder_symmetry)
        
        # 髖部對稱性
        if left_hip and right_hip:
            hip_symmetry = 1.0 - abs(left_hip.y - right_hip.y) / max(left_hip.y, right_hip.y)
            symmetry_scores.append(hip_symmetry)
        
        # 眼睛對稱性
        if left_eye and right_eye:
            eye_symmetry = 1.0 - abs(left_eye.y - right_eye.y) / max(left_eye.y, right_eye.y)
            symmetry_scores.append(eye_symmetry)
        
        if symmetry_scores:
            return sum(symmetry_scores) / len(symmetry_scores)
        return 0.0
    
    def _get_symmetry_level(self, symmetry_score: float) -> BodySymmetry:
        """根據對稱性分數獲取對稱性等級"""
        if symmetry_score >= 0.9:
            return BodySymmetry.EXCELLENT
        elif symmetry_score >= 0.8:
            return BodySymmetry.GOOD
        elif symmetry_score >= 0.7:
            return BodySymmetry.FAIR
        else:
            return BodySymmetry.POOR
    
    def _analyze_spine_alignment(self, pose_person: PosePerson) -> float:
        """分析脊椎對齊度"""
        # 獲取脊椎相關關鍵點
        nose = pose_person.get_keypoint_by_name("nose")
        left_shoulder = pose_person.get_keypoint_by_name("left_shoulder")
        right_shoulder = pose_person.get_keypoint_by_name("right_shoulder")
        left_hip = pose_person.get_keypoint_by_name("left_hip")
        right_hip = pose_person.get_keypoint_by_name("right_hip")
        
        if not all([nose, left_shoulder, right_shoulder, left_hip, right_hip]):
            return 0.0
        
        # 計算脊椎中線
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        
        # 計算脊椎傾斜角度
        spine_angle = math.atan2(abs(shoulder_center_x - hip_center_x), 
                                abs(left_shoulder.y - left_hip.y))
        spine_angle_deg = math.degrees(spine_angle)
        
        # 脊椎對齊度（角度越小越好）
        alignment_score = max(0, 1.0 - spine_angle_deg / 90.0)
        return alignment_score
    
    def _analyze_shoulder_level(self, left_shoulder: Optional[Keypoint], 
                               right_shoulder: Optional[Keypoint]) -> float:
        """分析肩膀水平度"""
        if not (left_shoulder and right_shoulder):
            return 0.0
        
        # 計算肩膀高度差異
        height_diff = abs(left_shoulder.y - right_shoulder.y)
        max_height = max(left_shoulder.y, right_shoulder.y)
        
        if max_height > 0:
            level_score = 1.0 - (height_diff / max_height)
            return max(0, level_score)
        return 0.0
    
    def _analyze_hip_level(self, left_hip: Optional[Keypoint], 
                          right_hip: Optional[Keypoint]) -> float:
        """分析髖部水平度"""
        if not (left_hip and right_hip):
            return 0.0
        
        # 計算髖部高度差異
        height_diff = abs(left_hip.y - right_hip.y)
        max_height = max(left_hip.y, right_hip.y)
        
        if max_height > 0:
            level_score = 1.0 - (height_diff / max_height)
            return max(0, level_score)
        return 0.0
    
    def _analyze_head_position(self, nose: Optional[Keypoint], 
                              left_shoulder: Optional[Keypoint], 
                              right_shoulder: Optional[Keypoint]) -> str:
        """分析頭部位置"""
        if not all([nose, left_shoulder, right_shoulder]):
            return "unknown"
        
        # 計算肩膀中心
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        
        # 計算頭部相對於肩膀的位置
        head_offset = nose.x - shoulder_center_x
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        
        if shoulder_width > 0:
            relative_offset = head_offset / shoulder_width
            
            if abs(relative_offset) < 0.1:
                return "centered"
            elif relative_offset > 0.1:
                return "right_tilted"
            else:
                return "left_tilted"
        
        return "unknown"
    
    def _generate_recommendations(self, posture_type: BodyPosture, 
                                symmetry_score: float, spine_alignment: float,
                                shoulder_level: float, hip_level: float) -> List[str]:
        """生成改善建議"""
        recommendations = []
        
        # 根據姿態類型給出建議
        if posture_type == BodyPosture.SITTING:
            recommendations.append("保持背部挺直，避免駝背")
            recommendations.append("調整座椅高度，確保腳掌能平放地面")
        elif posture_type == BodyPosture.STANDING:
            recommendations.append("保持身體重心在兩腳之間")
            recommendations.append("避免長時間單腳站立")
        
        # 根據對稱性給出建議
        if symmetry_score < 0.8:
            recommendations.append("注意身體左右平衡，避免單側負重")
            recommendations.append("進行對稱性運動訓練")
        
        # 根據脊椎對齊給出建議
        if spine_alignment < 0.8:
            recommendations.append("注意脊椎保持直立，避免側彎")
            recommendations.append("進行核心肌群訓練")
        
        # 根據肩膀水平度給出建議
        if shoulder_level < 0.8:
            recommendations.append("注意肩膀保持水平，避免高低肩")
            recommendations.append("檢查是否有脊椎側彎問題")
        
        # 根據髖部水平度給出建議
        if hip_level < 0.8:
            recommendations.append("注意髖部保持水平，避免骨盆傾斜")
            recommendations.append("進行髖部穩定訓練")
        
        return recommendations
    
    def update_history(self, posture: BodyPosture, metrics: BodyMetrics):
        """更新歷史記錄"""
        self.posture_history.append(posture)
        self.metrics_history.append(metrics)
        
        # 保持最近 100 條記錄
        if len(self.posture_history) > 100:
            self.posture_history.pop(0)
            self.metrics_history.pop(0)
    
    def get_posture_trend(self) -> Dict[str, Any]:
        """獲取姿態趨勢分析"""
        if not self.posture_history:
            return {}
        
        # 統計各姿態的出現頻率
        posture_counts = {}
        for posture in self.posture_history:
            posture_counts[posture.value] = posture_counts.get(posture.value, 0) + 1
        
        # 計算主要姿態
        total = len(self.posture_history)
        main_posture = max(posture_counts.items(), key=lambda x: x[1])[0]
        main_posture_ratio = posture_counts[main_posture] / total
        
        return {
            "total_samples": total,
            "main_posture": main_posture,
            "main_posture_ratio": main_posture_ratio,
            "posture_distribution": {k: v/total for k, v in posture_counts.items()}
        }
    
    def get_metrics_trend(self) -> Dict[str, Any]:
        """獲取身體指標趨勢分析"""
        if not self.metrics_history:
            return {}
        
        # 計算各指標的平均值和變化趨勢
        metrics_summary = {}
        
        for field in ["height_pixels", "shoulder_width", "hip_width", "arm_length", "leg_length"]:
            values = [getattr(m, field) for m in self.metrics_history if getattr(m, field) > 0]
            if values:
                metrics_summary[field] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "trend": "stable" if np.std(values) < np.mean(values) * 0.1 else "variable"
                }
        
        return metrics_summary


def draw_body_analysis_overlay(image: np.ndarray, analysis: PostureAnalysis, 
                              metrics: BodyMetrics, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """在影像上繪製身體分析結果"""
    if not analysis:
        return image
    
            # 創建分析文字
        lines = [
            f"Posture: {analysis.posture_type.value}",
            f"Confidence: {analysis.confidence:.2f}",
            f"Symmetry: {analysis.symmetry_level.value} ({analysis.symmetry_score:.2f})",
            f"Spine Alignment: {analysis.spine_alignment:.2f}",
            f"Shoulder Level: {analysis.shoulder_level:.2f}",
            f"Hip Level: {analysis.hip_level:.2f}",
            f"Head Position: {analysis.head_position}"
        ]
        
        # 添加身體指標
        if metrics.height_cm:
            lines.append(f"Height: {metrics.height_cm:.1f} cm")
        lines.append(f"Shoulder Width: {metrics.shoulder_width:.1f} px")
        lines.append(f"Hip Width: {metrics.hip_width:.1f} px")
        
        # 添加建議
        if analysis.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in analysis.recommendations[:3]:  # 只顯示前 3 條建議
                lines.append(f"• {rec}")
    
    # 繪製背景
    line_height = 25
    total_height = len(lines) * line_height + 10
    cv2.rectangle(image, 
                 (position[0] - 5, position[1] - 25),
                 (position[0] + 350, position[1] + total_height),
                 (0, 0, 0), -1)
    
    # 繪製文字
    for i, line in enumerate(lines):
        y_pos = position[1] + i * line_height
        if line.startswith("•"):
            # 建議項目使用不同顏色
            cv2.putText(image, line, (position[0] + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        elif line == "改善建議:":
            # 標題使用不同顏色
            cv2.putText(image, line, (position[0], y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        else:
            # 一般資訊
            cv2.putText(image, line, (position[0], y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return image
