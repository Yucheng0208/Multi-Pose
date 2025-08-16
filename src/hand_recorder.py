# hand_recorder.py
# MediaPipe Hand Landmarker 手部節點檢測與追蹤記錄器
# 支援直接執行與模組 import 使用

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any, Tuple, Generator
from enum import Enum

import numpy as np
import pandas as pd
import cv2

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError("MediaPipe is required. Please `pip install mediapipe`.") from e


# ---- MediaPipe Hand Landmarker 定義 ----
# MediaPipe 提供 21 個手部關鍵點，這裡定義完整的關鍵點名稱
HAND_LANDMARKS: List[str] = [
    # 手腕 (0)
    "wrist",
    
    # 拇指 (1-4)
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    
    # 食指 (5-8)
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    
    # 中指 (9-12)
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    
    # 無名指 (13-16)
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    
    # 小指 (17-20)
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

# 手部關鍵點索引映射（使用 MediaPipe 的標準索引）
HAND_LANDMARK_INDICES = {
    "wrist": 0,
    "thumb_cmc": 1, "thumb_mcp": 2, "thumb_ip": 3, "thumb_tip": 4,
    "index_finger_mcp": 5, "index_finger_pip": 6, "index_finger_dip": 7, "index_finger_tip": 8,
    "middle_finger_mcp": 9, "middle_finger_pip": 10, "middle_finger_dip": 11, "middle_finger_tip": 12,
    "ring_finger_mcp": 13, "ring_finger_pip": 14, "ring_finger_dip": 15, "ring_finger_tip": 16,
    "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19, "pinky_tip": 20
}

# 手部動作分類
class HandGesture(Enum):
    """手部動作類型"""
    FIST = "fist"
    OPEN_PALM = "open_palm"
    THUMB_UP = "thumb_up"
    THUMB_DOWN = "thumb_down"
    POINTING = "pointing"
    PEACE = "peace"
    OK = "ok"
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"
    UNKNOWN = "unknown"


class HandRecorder:
    """
    記錄每幀、每隻手、每個關鍵點的資料列：
    <frame_id><hand_id><landmark><x><y><z><confidence><gesture>
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        pixel_to_cm: Optional[float] = None,
        model_label: Optional[str] = None,
    ) -> None:
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.pixel_to_cm = pixel_to_cm
        self.model_label = model_label or f"hand_landmarker_{model_complexity}"

        # 初始化 MediaPipe Hand Landmarker
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # 初始化繪圖工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        logging.info("已載入 MediaPipe Hand Landmarker 模型: %s", self.model_label)

    def _detect_hands(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """檢測手部關鍵點"""
        # 轉換為 RGB 格式（MediaPipe 需要 RGB）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 進行手部檢測
        results = self.hands.process(rgb_image)
        
        hands_data = []
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 獲取手部類型（左手或右手）
                handedness = results.multi_handedness[hand_idx].classification[0].label
                confidence = results.multi_handedness[hand_idx].classification[0].score
                
                # 提取關鍵點座標
                landmarks = []
                for landmark_idx, landmark in enumerate(hand_landmarks.landmark):
                    # 轉換為像素座標
                    h, w, _ = image.shape
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    z = landmark.z
                    
                    landmarks.append({
                        'name': HAND_LANDMARKS[landmark_idx],
                        'x': x,
                        'y': y,
                        'z': z,
                        'confidence': landmark.visibility
                    })
                
                # 識別手部動作
                gesture = self._classify_gesture(landmarks)
                
                hands_data.append({
                    'hand_id': hand_idx,
                    'handedness': handedness,
                    'confidence': confidence,
                    'landmarks': landmarks,
                    'gesture': gesture
                })
        
        return hands_data

    def _classify_gesture(self, landmarks: List[Dict[str, Any]]) -> HandGesture:
        """分類手部動作"""
        if not landmarks:
            return HandGesture.UNKNOWN
        
        # 獲取關鍵點座標（直接使用索引，因為 landmarks 是按索引順序存儲的）
        wrist = landmarks[0]  # 手腕
        thumb_tip = landmarks[4]  # 拇指尖
        index_tip = landmarks[8]  # 食指尖
        middle_tip = landmarks[12]  # 中指尖
        ring_tip = landmarks[16]  # 無名指尖
        pinky_tip = landmarks[20]  # 小指尖
        
        # 計算手指是否伸展
        def is_finger_extended(finger_tip, wrist):
            return finger_tip['y'] < wrist['y'] - 20  # 手指尖在手腕上方
        
        thumb_extended = is_finger_extended(thumb_tip, wrist)
        index_extended = is_finger_extended(index_tip, wrist)
        middle_extended = is_finger_extended(middle_tip, wrist)
        ring_extended = is_finger_extended(ring_tip, wrist)
        pinky_extended = is_finger_extended(pinky_tip, wrist)
        
        # 動作分類邏輯
        if all([index_extended, middle_extended, ring_extended, pinky_extended]):
            if thumb_extended:
                return HandGesture.OPEN_PALM
            else:
                return HandGesture.FIST
        elif index_extended and not any([middle_extended, ring_extended, pinky_extended]):
            return HandGesture.POINTING
        elif index_extended and middle_extended and not any([ring_extended, pinky_extended]):
            return HandGesture.PEACE
        elif thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
            return HandGesture.THUMB_UP
        elif all([index_extended, middle_extended, ring_extended, pinky_extended, thumb_extended]):
            return HandGesture.OPEN_PALM
        
        return HandGesture.UNKNOWN

    def _record_hand_data(self, frame_id: int, hands_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """記錄手部資料為統一格式"""
        records = []
        
        for hand_data in hands_data:
            hand_id = hand_data['hand_id']
            landmarks = {}
            
            for landmark in hand_data['landmarks']:
                landmarks[landmark['name']] = {
                    'x': landmark['x'],
                    'y': landmark['y'],
                    'confidence': landmark['confidence']
                }
            
            if landmarks:  # 只添加有有效關鍵點的結果
                records.append({
                    'id': hand_id,
                    'landmarks': landmarks,
                    'model': self.model_label
                })
        
        return records

    def process_video(
        self,
        source: Any,
        output_csv: Optional[Path] = None,
        output_video: Optional[Path] = None,
        show_video: bool = False,
        save_frames: bool = False,
        frames_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        處理影片並記錄手部資料
        
        Args:
            source: 影片來源（檔案路徑、攝影機索引等）
            output_csv: 輸出 CSV 檔案路徑
            output_video: 輸出影片檔案路徑
            show_video: 是否顯示影片
            save_frames: 是否儲存幀
            frames_dir: 幀儲存目錄
        
        Returns:
            包含所有手部資料的 DataFrame
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片來源: {source}")
        
        # 獲取影片資訊
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info("影片資訊: %dx%d, %d FPS, %d 幀", width, height, fps, total_frames)
        
        # 設定輸出影片
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
        # 設定幀儲存
        if save_frames and frames_dir:
            frames_dir.mkdir(parents=True, exist_ok=True)
        
        all_records = []
        frame_id = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 檢測手部
                hands_data = self._detect_hands(frame)
                
                # 記錄資料
                frame_records = self._record_hand_data(frame_id, hands_data)
                all_records.extend(frame_records)
                
                # 繪製手部關鍵點
                annotated_frame = self._draw_hands(frame, hands_data)
                
                # 顯示處理進度
                if frame_id % 30 == 0:  # 每 30 幀顯示一次進度
                    progress = (frame_id / total_frames) * 100 if total_frames > 0 else frame_id
                    logging.info("處理進度: %.1f%% (%d/%d)", progress, frame_id, total_frames)
                
                # 儲存幀
                if save_frames and frames_dir:
                    frame_path = frames_dir / f"frame_{frame_id:06d}.jpg"
                    cv2.imwrite(str(frame_path), annotated_frame)
                
                # 寫入輸出影片
                if output_video:
                    out.write(annotated_frame)
                
                # 顯示影片
                if show_video:
                    cv2.imshow('Hand Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_id += 1
                
        finally:
            cap.release()
            if output_video:
                out.release()
            if show_video:
                cv2.destroyAllWindows()
        
        # 創建 DataFrame
        df = pd.DataFrame(all_records)
        
        # 儲存 CSV
        if output_csv:
            df.to_csv(output_csv, index=False)
            logging.info("手部資料已儲存至: %s", output_csv)
        
        logging.info("處理完成，共處理 %d 幀，檢測到 %d 筆手部資料", frame_id, len(all_records))
        
        return df

    def _draw_hands(self, image: np.ndarray, hands_data: List[Dict[str, Any]]) -> np.ndarray:
        """在影像上繪製手部關鍵點和連線"""
        annotated_image = image.copy()
        
        for hand_data in hands_data:
            # 繪製手部關鍵點和連線（使用 OpenCV 繪製，因為我們有自己的資料格式）
            landmarks = hand_data['landmarks']
            
            # 繪製關鍵點
            for landmark in landmarks:
                x, y = landmark['x'], landmark['y']
                cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)
            
            # 繪製手部連線（簡化的連線）
            if len(landmarks) >= 21:
                # 手腕到拇指
                self._draw_line(annotated_image, landmarks[0], landmarks[1], (255, 0, 0))
                self._draw_line(annotated_image, landmarks[1], landmarks[2], (255, 0, 0))
                self._draw_line(annotated_image, landmarks[2], landmarks[3], (255, 0, 0))
                self._draw_line(annotated_image, landmarks[3], landmarks[4], (255, 0, 0))
                
                # 手腕到其他手指
                for finger_start in [5, 9, 13, 17]:  # 各手指的起始點
                    self._draw_line(annotated_image, landmarks[0], landmarks[finger_start], (0, 255, 0))
                    for i in range(3):
                        self._draw_line(annotated_image, landmarks[finger_start + i], landmarks[finger_start + i + 1], (0, 255, 0))
            
            # 添加手部資訊標籤
            hand_id = hand_data['hand_id']
            handedness = hand_data['handedness']
            gesture = hand_data['gesture'].value
            
            # 在手腕位置顯示標籤
            wrist = landmarks[0]  # 手腕索引為 0
            label = f"{handedness} {hand_id}: {gesture}"
            cv2.putText(annotated_image, label, (wrist['x'] + 10, wrist['y'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_image
    
    def _draw_line(self, image: np.ndarray, point1: Dict[str, Any], point2: Dict[str, Any], color: Tuple[int, int, int]):
        """繪製兩點之間的連線"""
        x1, y1 = point1['x'], point1['y']
        x2, y2 = point2['x'], point2['y']
        cv2.line(image, (x1, y1), (x2, y2), color, 2)

    def process_image(self, image_path: Path) -> List[Dict[str, Any]]:
        """處理單張圖片"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"無法讀取圖片: {image_path}")
        
        hands_data = self._detect_hands(image)
        return hands_data

    def get_hand_metrics(self, hands_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """計算手部測量指標"""
        if not hands_data:
            return {}
        
        metrics = {}
        
        for hand_data in hands_data:
            hand_id = hand_data['hand_id']
            landmarks = hand_data['landmarks']
            
            # 計算手部大小
            wrist = landmarks[0]  # 手腕索引為 0
            middle_tip = landmarks[12]  # 中指尖索引為 12
            
            # 手部長度（手腕到中指指尖）
            hand_length = np.sqrt(
                (middle_tip['x'] - wrist['x'])**2 + 
                (middle_tip['y'] - wrist['y'])**2
            )
            
            # 手部寬度（拇指到小指）
            thumb_tip = landmarks[4]  # 拇指尖索引為 4
            pinky_tip = landmarks[20]  # 小指尖索引為 20
            
            hand_width = np.sqrt(
                (pinky_tip['x'] - thumb_tip['x'])**2 + 
                (pinky_tip['y'] - thumb_tip['y'])**2
            )
            
            metrics[f"hand_{hand_id}_length"] = hand_length
            metrics[f"hand_{hand_id}_width"] = hand_width
            
            if self.pixel_to_cm:
                metrics[f"hand_{hand_id}_length_cm"] = hand_length * self.pixel_to_cm
                metrics[f"hand_{hand_id}_width_cm"] = hand_width * self.pixel_to_cm
        
        return metrics

    def close(self):
        """關閉 MediaPipe 資源"""
        if hasattr(self, 'hands'):
            self.hands.close()


def main():
    """主函數：用於直接執行"""
    parser = argparse.ArgumentParser(description="MediaPipe 手部關鍵點檢測與記錄")
    parser.add_argument("source", help="影片來源（檔案路徑或攝影機索引）")
    parser.add_argument("--output-csv", type=Path, help="輸出 CSV 檔案路徑")
    parser.add_argument("--output-video", type=Path, help="輸出影片檔案路徑")
    parser.add_argument("--show-video", action="store_true", help="顯示影片")
    parser.add_argument("--save-frames", action="store_true", help="儲存幀")
    parser.add_argument("--frames-dir", type=Path, help="幀儲存目錄")
    parser.add_argument("--max-hands", type=int, default=2, help="最大手部數量")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1], help="模型複雜度")
    parser.add_argument("--detection-confidence", type=float, default=0.5, help="檢測信心度")
    parser.add_argument("--tracking-confidence", type=float, default=0.5, help="追蹤信心度")
    parser.add_argument("--pixel-to-cm", type=float, help="像素到公分的轉換比例")
    
    args = parser.parse_args()
    
    # 設定日誌
    logging.basicConfig(level=logging.INFO)
    
    # 創建手部記錄器
    recorder = HandRecorder(
        max_num_hands=args.max_hands,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence,
        pixel_to_cm=args.pixel_to_cm
    )
    
    try:
        # 處理影片
        df = recorder.process_video(
            source=args.source,
            output_csv=args.output_csv,
            output_video=args.output_video,
            show_video=args.show_video,
            save_frames=args.save_frames,
            frames_dir=args.frames_dir
        )
        
        print(f"處理完成，共記錄 {len(df)} 筆手部資料")
        
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
