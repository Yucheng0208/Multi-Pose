#!/usr/bin/env python3
"""
骨架視覺化模組
提供增強的骨架繪製、身體偵測和姿態分析功能
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

# COCO-17 關鍵點定義
KEYPOINTS_C17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 骨架連線定義 (COCO-17 格式)
SKELETON_CONNECTIONS = [
    # 頭部
    (0, 1), (0, 2), (1, 3), (2, 4),  # 鼻子到眼睛、耳朵
    # 軀幹
    (5, 6), (5, 7), (6, 8), (7, 9),  # 肩膀到手臂
    (5, 11), (6, 12), (11, 12),      # 肩膀到髖部，髖部連線
    # 手臂
    (7, 9), (8, 10),                 # 手肘到手腕
    # 腿部
    (11, 13), (12, 14), (13, 15), (14, 16)  # 髖部到膝蓋到腳踝
]

# 身體部位顏色定義
BODY_PART_COLORS = {
    "head": (255, 255, 0),      # 黃色 - 頭部
    "torso": (0, 255, 255),     # 青色 - 軀幹
    "arms": (255, 0, 255),      # 洋紅色 - 手臂
    "legs": (0, 255, 0),        # 綠色 - 腿部
    "hands": (255, 165, 0),     # 橙色 - 手部
    "feet": (128, 0, 128)       # 紫色 - 腳部
}

@dataclass
class Keypoint:
    """關鍵點資料結構"""
    x: float
    y: float
    confidence: float
    name: str
    
    def is_valid(self) -> bool:
        """檢查關鍵點是否有效"""
        return not (np.isnan(self.x) or np.isnan(self.y)) and self.confidence > 0.0

@dataclass
class PosePerson:
    """單一人物姿態資料結構"""
    id: int
    keypoints: List[Keypoint]
    bbox: Optional[Tuple[float, float, float, float]] = None
    
    def get_keypoint_by_name(self, name: str) -> Optional[Keypoint]:
        """根據名稱獲取關鍵點"""
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None
    
    def get_visible_keypoints(self) -> List[Keypoint]:
        """獲取可見的關鍵點"""
        return [kp for kp in self.keypoints if kp.is_valid()]

class PoseVisualizer:
    """骨架視覺化器"""
    
    def __init__(
        self,
        line_thickness: int = 2,
        keypoint_radius: int = 4,
        show_confidence: bool = True,
        show_keypoint_names: bool = False,
        skeleton_alpha: float = 0.8,
        body_contour_alpha: float = 0.3
    ):
        self.line_thickness = line_thickness
        self.keypoint_radius = keypoint_radius
        self.show_confidence = show_confidence
        self.show_keypoint_names = show_keypoint_names
        self.skeleton_alpha = skeleton_alpha
        self.body_contour_alpha = body_contour_alpha
        
        # 初始化顏色映射
        self._init_color_mapping()
    
    def _init_color_mapping(self):
        """初始化顏色映射"""
        self.keypoint_colors = {}
        self.connection_colors = {}
        
        # 為每個關鍵點分配顏色
        for i, kp_name in enumerate(KEYPOINTS_C17):
            if "head" in kp_name or "eye" in kp_name or "ear" in kp_name or "nose" in kp_name:
                self.keypoint_colors[kp_name] = BODY_PART_COLORS["head"]
            elif "shoulder" in kp_name or "hip" in kp_name:
                self.keypoint_colors[kp_name] = BODY_PART_COLORS["torso"]
            elif "elbow" in kp_name or "wrist" in kp_name:
                self.keypoint_colors[kp_name] = BODY_PART_COLORS["arms"]
            elif "knee" in kp_name or "ankle" in kp_name:
                self.keypoint_colors[kp_name] = BODY_PART_COLORS["legs"]
            else:
                self.keypoint_colors[kp_name] = (255, 255, 255)  # 白色
        
        # 為連線分配顏色
        for i, (start_idx, end_idx) in enumerate(SKELETON_CONNECTIONS):
            start_name = KEYPOINTS_C17[start_idx]
            end_name = KEYPOINTS_C17[end_idx]
            
            # 根據連線的關鍵點類型決定顏色
            if any(part in start_name for part in ["head", "eye", "ear", "nose"]) or \
               any(part in end_name for part in ["head", "eye", "ear", "nose"]):
                self.connection_colors[(start_idx, end_idx)] = BODY_PART_COLORS["head"]
            elif any(part in start_name for part in ["shoulder", "hip"]) or \
                 any(part in end_name for part in ["shoulder", "hip"]):
                self.connection_colors[(start_idx, end_idx)] = BODY_PART_COLORS["torso"]
            elif any(part in start_name for part in ["elbow", "wrist"]) or \
                 any(part in end_name for part in ["elbow", "wrist"]):
                self.connection_colors[(start_idx, end_idx)] = BODY_PART_COLORS["arms"]
            elif any(part in start_name for part in ["knee", "ankle"]) or \
                 any(part in end_name for part in ["knee", "ankle"]):
                self.connection_colors[(start_idx, end_idx)] = BODY_PART_COLORS["legs"]
            else:
                self.connection_colors[(start_idx, end_idx)] = (128, 128, 128)  # 灰色
    
    def draw_skeleton(
        self, 
        image: np.ndarray, 
        pose_person: PosePerson,
        draw_connections: bool = True,
        draw_keypoints: bool = True,
        draw_bbox: bool = True
    ) -> np.ndarray:
        """在影像上繪製骨架"""
        if not pose_person.keypoints:
            return image
        
        # 創建副本以避免修改原圖
        result_image = image.copy()
        
        # 繪製骨架連線
        if draw_connections:
            result_image = self._draw_skeleton_connections(result_image, pose_person)
        
        # 繪製關鍵點
        if draw_keypoints:
            result_image = self._draw_keypoints(result_image, pose_person)
        
        # 繪製邊界框
        if draw_bbox and pose_person.bbox:
            result_image = self._draw_bbox(result_image, pose_person.bbox, pose_person.id)
        
        return result_image
    
    def _draw_skeleton_connections(self, image: np.ndarray, pose_person: PosePerson) -> np.ndarray:
        """繪製骨架連線"""
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx >= len(pose_person.keypoints) or end_idx >= len(pose_person.keypoints):
                continue
            
            start_kp = pose_person.keypoints[start_idx]
            end_kp = pose_person.keypoints[end_idx]
            
            # 檢查兩個關鍵點是否都有效
            if not (start_kp.is_valid() and end_kp.is_valid()):
                continue
            
            # 獲取連線顏色
            color = self.connection_colors.get((start_idx, end_idx), (128, 128, 128))
            
            # 繪製連線
            start_point = (int(start_kp.x), int(start_kp.y))
            end_point = (int(end_kp.x), int(end_kp.y))
            
            cv2.line(image, start_point, end_point, color, self.line_thickness)
        
        return image
    
    def _draw_keypoints(self, image: np.ndarray, pose_person: PosePerson) -> np.ndarray:
        """繪製關鍵點"""
        for kp in pose_person.keypoints:
            if not kp.is_valid():
                continue
            
            # 獲取關鍵點顏色
            color = self.keypoint_colors.get(kp.name, (255, 255, 255))
            
            # 繪製關鍵點圓圈
            center = (int(kp.x), int(kp.y))
            cv2.circle(image, center, self.keypoint_radius, color, -1)
            
            # 繪製邊框
            cv2.circle(image, center, self.keypoint_radius, (0, 0, 0), 1)
            
            # 顯示信心度（可選）
            if self.show_confidence:
                confidence_text = f"{kp.confidence:.2f}"
                text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                text_x = center[0] - text_size[0] // 2
                text_y = center[1] - self.keypoint_radius - 5
                
                # 繪製背景矩形
                cv2.rectangle(image, 
                            (text_x - 2, text_y - text_size[1] - 2),
                            (text_x + text_size[0] + 2, text_y + 2),
                            (0, 0, 0), -1)
                
                # 繪製文字
                cv2.putText(image, confidence_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 顯示關鍵點名稱（可選）
            if self.show_keypoint_names:
                name_text = kp.name.replace("_", " ")
                text_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                text_x = center[0] - text_size[0] // 2
                text_y = center[1] + self.keypoint_radius + text_size[1] + 5
                
                # 繪製背景矩形
                cv2.rectangle(image, 
                            (text_x - 2, text_y - text_size[1] - 2),
                            (text_x + text_size[0] + 2, text_y + 2),
                            (0, 0, 0), -1)
                
                # 繪製文字
                cv2.putText(image, name_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return image
    
    def _draw_bbox(self, image: np.ndarray, bbox: Tuple[float, float, float, float], person_id: int) -> np.ndarray:
        """繪製邊界框"""
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 繪製邊界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 繪製ID標籤
        label = f"ID: {person_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # 標籤背景
        cv2.rectangle(image, 
                     (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0] + 10, y1),
                     (0, 255, 0), -1)
        
        # 標籤文字
        cv2.putText(image, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image
    
    def draw_body_contour(self, image: np.ndarray, pose_person: PosePerson) -> np.ndarray:
        """繪製身體輪廓（基於關鍵點）"""
        if not pose_person.keypoints:
            return image
        
        # 獲取可見的關鍵點
        visible_kps = pose_person.get_visible_keypoints()
        if len(visible_kps) < 3:
            return image
        
        # 創建輪廓點
        contour_points = []
        
        # 頭部輪廓
        head_kps = [kp for kp in visible_kps if any(part in kp.name for part in ["head", "eye", "ear", "nose"])]
        if len(head_kps) >= 3:
            head_contour = np.array([[int(kp.x), int(kp.y)] for kp in head_kps], dtype=np.int32)
            if len(head_contour) >= 3:
                contour_points.append(head_contour)
        
        # 軀幹輪廓
        torso_kps = [kp for kp in visible_kps if any(part in kp.name for part in ["shoulder", "hip"])]
        if len(torso_kps) >= 3:
            torso_contour = np.array([[int(kp.x), int(kp.y)] for kp in torso_kps], dtype=np.int32)
            if len(torso_contour) >= 3:
                contour_points.append(torso_contour)
        
        # 繪製輪廓
        for contour in contour_points:
            # 使用半透明填充
            overlay = image.copy()
            cv2.fillPoly(overlay, [contour], BODY_PART_COLORS["torso"])
            cv2.addWeighted(overlay, self.body_contour_alpha, image, 1 - self.body_contour_alpha, 0, image)
            
            # 繪製輪廓邊界
            cv2.polylines(image, [contour], True, BODY_PART_COLORS["torso"], 1)
        
        return image
    
    def analyze_pose(self, pose_person: PosePerson) -> Dict[str, Any]:
        """分析姿態並返回分析結果"""
        if not pose_person.keypoints:
            return {}
        
        analysis = {
            "total_keypoints": len(pose_person.keypoints),
            "visible_keypoints": len([kp for kp in pose_person.keypoints if kp.is_valid()]),
            "average_confidence": 0.0,
            "body_parts": {},
            "pose_quality": "unknown"
        }
        
        # 計算平均信心度
        valid_kps = [kp for kp in pose_person.keypoints if kp.is_valid()]
        if valid_kps:
            analysis["average_confidence"] = sum(kp.confidence for kp in valid_kps) / len(valid_kps)
        
        # 分析身體部位
        for part_name, part_keywords in {
            "head": ["nose", "eye", "ear"],
            "torso": ["shoulder", "hip"],
            "arms": ["elbow", "wrist"],
            "legs": ["knee", "ankle"]
        }.items():
            part_kps = [kp for kp in valid_kps if any(keyword in kp.name for keyword in part_keywords)]
            analysis["body_parts"][part_name] = {
                "count": len(part_kps),
                "average_confidence": sum(kp.confidence for kp in part_kps) / len(part_kps) if part_kps else 0.0
            }
        
        # 評估姿態品質
        visible_ratio = analysis["visible_keypoints"] / analysis["total_keypoints"]
        if visible_ratio >= 0.8 and analysis["average_confidence"] >= 0.7:
            analysis["pose_quality"] = "excellent"
        elif visible_ratio >= 0.6 and analysis["average_confidence"] >= 0.5:
            analysis["pose_quality"] = "good"
        elif visible_ratio >= 0.4 and analysis["average_confidence"] >= 0.3:
            analysis["pose_quality"] = "fair"
        else:
            analysis["pose_quality"] = "poor"
        
        return analysis
    
    def draw_pose_analysis(self, image: np.ndarray, analysis: Dict[str, Any], position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """在影像上繪製姿態分析結果"""
        if not analysis:
            return image
        
        # 創建分析文字
        lines = [
            f"Pose Quality: {analysis.get('pose_quality', 'unknown')}",
            f"Visible Keypoints: {analysis.get('visible_keypoints', 0)}/{analysis.get('total_keypoints', 0)}",
            f"Avg Confidence: {analysis.get('average_confidence', 0.0):.2f}"
        ]
        
        # 添加身體部位資訊
        body_parts = analysis.get("body_parts", {})
        for part_name, part_info in body_parts.items():
            lines.append(f"{part_name}: {part_info['count']} pts, conf: {part_info['average_confidence']:.2f}")
        
        # 繪製背景
        line_height = 25
        total_height = len(lines) * line_height + 10
        cv2.rectangle(image, 
                     (position[0] - 5, position[1] - 25),
                     (position[0] + 300, position[1] + total_height),
                     (0, 0, 0), -1)
        
        # 繪製文字
        for i, line in enumerate(lines):
            y_pos = position[1] + i * line_height
            cv2.putText(image, line, (position[0], y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return image

def create_pose_person_from_result(result: Any, person_id: int) -> PosePerson:
    """從 YOLO 結果創建 PosePerson 物件"""
    keypoints = []
    
    # 獲取關鍵點資料
    kps = getattr(result, "keypoints", None)
    if kps is not None and kps.data is not None:
        kp_arr = kps.data.cpu().numpy()  # (N, 17, 3)
        
        for k_idx, k_name in enumerate(KEYPOINTS_C17):
            if k_idx < kp_arr.shape[1]:
                x, y, score = kp_arr[0, k_idx, :]
                keypoints.append(Keypoint(
                    x=float(x), y=float(y), 
                    confidence=float(score), 
                    name=k_name
                ))
    
    # 獲取邊界框
    bbox = None
    boxes = getattr(result, "boxes", None)
    if boxes is not None and hasattr(boxes, "xyxy") and boxes.xyxy is not None:
        bbox_data = boxes.xyxy.cpu().numpy()
        if len(bbox_data) > 0:
            bbox = tuple(bbox_data[0].tolist())  # (x1, y1, x2, y2)
    
    return PosePerson(id=person_id, keypoints=keypoints, bbox=bbox)
