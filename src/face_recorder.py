# face_recorder.py
# MediaPipe 臉部網格模型檢測與追蹤記錄器
# 支援直接執行與模組 import 使用

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any, Tuple, Generator

import numpy as np
import pandas as pd
import cv2

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError("MediaPipe is required. Please `pip install mediapipe`.") from e


# ---- MediaPipe Face Mesh Model 定義 ----
# MediaPipe 提供 468 個臉部網格點，這裡定義完整的網格點名稱
FACE_MESH_POINTS: List[str] = [
    # 臉部輪廓 (0-16)
    "face_oval_0", "face_oval_1", "face_oval_2", "face_oval_3", "face_oval_4",
    "face_oval_5", "face_oval_6", "face_oval_7", "face_oval_8", "face_oval_9",
    "face_oval_10", "face_oval_11", "face_oval_12", "face_oval_13", "face_oval_14",
    "face_oval_15", "face_oval_16",
    
    # 右眼 (33-46)
    "right_eye_0", "right_eye_1", "right_eye_2", "right_eye_3", "right_eye_4",
    "right_eye_5", "right_eye_6", "right_eye_7", "right_eye_8", "right_eye_9",
    "right_eye_10", "right_eye_11", "right_eye_12", "right_eye_13",
    
    # 左眼 (362-375)
    "left_eye_0", "left_eye_1", "left_eye_2", "left_eye_3", "left_eye_4",
    "left_eye_5", "left_eye_6", "left_eye_7", "left_eye_8", "left_eye_9",
    "left_eye_10", "left_eye_11", "left_eye_12", "left_eye_13",
    
    # 鼻子 (1, 2, 5, 31, 35, 195-207)
    "nose_tip", "nose_bottom", "nose_left", "nose_right", "nose_bridge",
    "nose_left_wing", "nose_right_wing", "nose_left_wing_tip", "nose_right_wing_tip",
    
    # 嘴巴 (61-84, 85-108)
    "mouth_left", "mouth_right", "mouth_top", "mouth_bottom", "mouth_center",
    "upper_lip_top", "upper_lip_bottom", "lower_lip_top", "lower_lip_bottom",
    
    # 眉毛 (70-76, 336-342)
    "right_eyebrow_0", "right_eyebrow_1", "right_eyebrow_2", "right_eyebrow_3",
    "right_eyebrow_4", "right_eyebrow_5", "right_eyebrow_6",
    "left_eyebrow_0", "left_eyebrow_1", "left_eyebrow_2", "left_eyebrow_3",
    "left_eyebrow_4", "left_eyebrow_5", "left_eyebrow_6"
]

# 簡化版本：只使用最重要的網格點（16個）
SIMPLE_FACE_MESH_POINTS: List[str] = [
    "nose_tip", "nose_bottom", "nose_left", "nose_right",
    "right_eye_center", "left_eye_center",
    "mouth_left", "mouth_right", "mouth_top", "mouth_bottom",
    "right_eyebrow_center", "left_eyebrow_center",
    "face_oval_top", "face_oval_bottom", "face_oval_left", "face_oval_right"
]

# 完整 468 點網格點名稱（基於 MediaPipe 官方定義）
FULL_FACE_MESH_POINTS: List[str] = [
    # 臉部輪廓 (0-16)
    "face_oval_0", "face_oval_1", "face_oval_2", "face_oval_3", "face_oval_4",
    "face_oval_5", "face_oval_6", "face_oval_7", "face_oval_8", "face_oval_9",
    "face_oval_10", "face_oval_11", "face_oval_12", "face_oval_13", "face_oval_14",
    "face_oval_15", "face_oval_16",
    
    # 右眼 (33-46)
    "right_eye_0", "right_eye_1", "right_eye_2", "right_eye_3", "right_eye_4",
    "right_eye_5", "right_eye_6", "right_eye_7", "right_eye_8", "right_eye_9",
    "right_eye_10", "right_eye_11", "right_eye_12", "right_eye_13",
    
    # 左眼 (362-375)
    "left_eye_0", "left_eye_1", "left_eye_2", "left_eye_3", "left_eye_4",
    "left_eye_5", "left_eye_6", "left_eye_7", "left_eye_8", "left_eye_9",
    "left_eye_10", "left_eye_11", "left_eye_12", "left_eye_13",
    
    # 鼻子 (1, 2, 5, 31, 35, 195-207)
    "nose_tip", "nose_bottom", "nose_left", "nose_right", "nose_bridge",
    "nose_left_wing", "nose_right_wing", "nose_left_wing_tip", "nose_right_wing_tip",
    
    # 嘴巴 (61-84, 85-108)
    "mouth_left", "mouth_right", "mouth_top", "mouth_bottom", "mouth_center",
    "upper_lip_top", "upper_lip_bottom", "lower_lip_top", "lower_lip_bottom",
    
    # 眉毛 (70-76, 336-342)
    "right_eyebrow_0", "right_eyebrow_1", "right_eyebrow_2", "right_eyebrow_3",
    "right_eyebrow_4", "right_eyebrow_5", "right_eyebrow_6",
    "left_eyebrow_0", "left_eyebrow_1", "left_eyebrow_2", "left_eyebrow_3",
    "left_eyebrow_4", "left_eyebrow_5", "left_eyebrow_6"
]

# 網格點索引映射（MediaPipe 468 點中的索引）
MESH_POINT_INDICES = {
    "nose_tip": 1,
    "nose_bottom": 2,
    "nose_left": 5,
    "nose_right": 31,
    "right_eye_center": 159,
    "left_eye_center": 386,
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,
    "right_eyebrow_center": 70,
    "left_eyebrow_center": 336,
    "face_oval_top": 10,
    "face_oval_bottom": 152,
    "face_oval_left": 234,
    "face_oval_right": 454
}

# 完整的 468 點索引映射（這裡只列出部分，實際應該有 468 個）
# 為了簡化，我們將使用 MediaPipe 的原始索引 0-467
FULL_MESH_INDICES = {f"point_{i}": i for i in range(468)}


def _ensure_dir(p: Path) -> None:
    """確保輸出目錄存在"""
    p.parent.mkdir(parents=True, exist_ok=True)


class FaceMeshRecorder:
    """
    記錄每幀、每個臉部、每個網格點的資料列：
    <id><keypoints><model><coor_x><coor_y><cm>
    支援完整的 468 點 MediaPipe Face Mesh
    """

    def __init__(
        self,
        use_simple_mesh_points: bool = True,
        use_full_mesh: bool = False,
        device: Optional[str] = None,
        conf: float = 0.5,
        max_faces: int = 1,
        pixel_to_cm: Optional[float] = None,
        kp_score_min: float = 0.0,
        model_label: str = "mediapipe_face_mesh",
        window_width: int = 640,
        window_height: int = 480,
    ) -> None:
        self.use_simple_mesh_points = use_simple_mesh_points
        self.use_full_mesh = use_full_mesh
        self.device = device
        self.conf = conf
        self.max_faces = max_faces
        self.pixel_to_cm = pixel_to_cm
        self.kp_score_min = kp_score_min
        self.model_label = model_label
        self.window_width = window_width
        self.window_height = window_height

        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 設定臉部網格檢測參數
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_faces,
            refine_landmarks=True,
            min_detection_confidence=self.conf,
            min_tracking_confidence=self.conf
        )
        
        # 選擇使用的網格點列表
        if use_full_mesh:
            self.mesh_points = list(range(468))  # 使用所有 468 點
            self.mesh_point_names = [f"point_{i}" for i in range(468)]
        elif use_simple_mesh_points:
            self.mesh_points = SIMPLE_FACE_MESH_POINTS
            self.mesh_point_names = SIMPLE_FACE_MESH_POINTS
        else:
            self.mesh_points = FACE_MESH_POINTS
            self.mesh_point_names = FACE_MESH_POINTS
        
        logging.info("已初始化 MediaPipe Face Mesh 模型: %s", self.model_label)
        logging.info("使用 %d 個網格點", len(self.mesh_points))
        logging.info("視窗尺寸: %dx%d", self.window_width, self.window_height)

    def _detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        檢測單一影像中的臉部網格點
        """
        # 轉換為 RGB（MediaPipe 需要 RGB 格式）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 進行臉部網格檢測
        results = self.face_mesh.process(rgb_frame)
        
        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_data = {
                    'landmarks': face_landmarks,
                    'mesh_points': {}
                }
                
                # 提取網格點座標
                if self.use_full_mesh:
                    # 使用所有 468 點
                    for i in range(468):
                        if i < len(face_landmarks.landmark):
                            landmark = face_landmarks.landmark[i]
                            h, w = frame.shape[:2]
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            z = landmark.z  # 深度值（相對）
                            face_data['mesh_points'][f"point_{i}"] = (x, y, z)
                elif self.use_simple_mesh_points:
                    # 使用簡化網格點集
                    for mesh_point_name in SIMPLE_FACE_MESH_POINTS:
                        if mesh_point_name in MESH_POINT_INDICES:
                            idx = MESH_POINT_INDICES[mesh_point_name]
                            if idx < len(face_landmarks.landmark):
                                landmark = face_landmarks.landmark[idx]
                                h, w = frame.shape[:2]
                                x = int(landmark.x * w)
                                y = int(landmark.y * h)
                                z = landmark.z  # 深度值（相對）
                                face_data['mesh_points'][mesh_point_name] = (x, y, z)
                else:
                    # 使用自訂網格點集
                    for mesh_point_name in FACE_MESH_POINTS:
                        if mesh_point_name in MESH_POINT_INDICES:
                            idx = MESH_POINT_INDICES[mesh_point_name]
                            if idx < len(face_landmarks.landmark):
                                landmark = face_landmarks.landmark[idx]
                                h, w = frame.shape[:2]
                                x = int(landmark.x * w)
                                y = int(landmark.y * h)
                                z = landmark.z  # 深度值（相對）
                                face_data['mesh_points'][mesh_point_name] = (x, y, z)
                
                faces.append(face_data)
        
        return faces

    def _rows_from_frame(self, frame: np.ndarray, frame_id: int = 0) -> List[Dict[str, Any]]:
        """
        將一幀的檢測結果轉換為統一的關鍵點格式
        """
        rows: List[Dict[str, Any]] = []
        
        # 檢測臉部網格
        faces = self._detect_faces(frame)
        
        for face_idx, face_data in enumerate(faces):
            face_id = frame_id * 1000 + face_idx  # 簡單的 ID 分配
            landmarks = {}
            
            for mesh_point_name, (x, y, z) in face_data['mesh_points'].items():
                # 檢查座標有效性
                if np.isnan(x) or np.isnan(y):
                    continue
                
                # 使用 z 值作為 confidence（MediaPipe 的 z 值範圍通常在 -1 到 1 之間）
                confidence = max(0.0, min(1.0, (z + 1.0) / 2.0))  # 轉換到 0-1 範圍
                
                landmarks[mesh_point_name] = {
                    "x": float(x),
                    "y": float(y),
                    "confidence": confidence
                }
            
            if landmarks:  # 只添加有有效關鍵點的結果
                rows.append({
                    "id": face_id,
                    "landmarks": landmarks,
                    "model": self.model_label
                })
        
        return rows

    def process_source(
        self,
        source: Any,
        output_path: Optional[str] = "face_mesh_records.csv",
        video_out_path: Optional[str] = None,
        visualize: bool = False,
        save_fps: Optional[float] = None,
        window_name: Optional[str] = None,
    ) -> None:
        """
        處理影像來源並輸出臉部網格點資料
        """
        writer = None
        video_writer = None
        out_is_parquet = bool(output_path and str(output_path).lower().endswith(".parquet"))
        accum: List[Tuple[int, str, str, float, float, Any]] = []
        
        vc_fps = None
        frame_count = 0
        
        # 設定視窗名稱
        if window_name is None:
            window_name = "face_mesh_recorder"
        
        try:
            # 開啟影像來源
            if isinstance(source, (str, int)):
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    raise RuntimeError(f"無法開啟影像來源: {source}")
                
                # 設定攝影機解析度為 1920x1080（如果是網路攝影機）
                if source == 0 or source == "0":
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    
                    # 驗證解析度設定
                    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    logging.info(f"📹 攝影機解析度: {actual_width:.0f}x{actual_height:.0f}")
            else:
                cap = source
            
            # 取得 FPS
            if save_fps is None:
                vc_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            else:
                vc_fps = float(save_fps)
            
            # 設定視窗尺寸
            if visualize:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, self.window_width, self.window_height)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 處理當前幀
                rows = self._rows_from_frame(frame, frame_count)
                accum.extend(rows)
                
                # 繪製檢測結果（如果需要）
                if (visualize or video_out_path) and frame is not None:
                    # 在原始影像上繪製臉部網格點
                    drawn_frame = self._draw_face_mesh_points(frame, rows)
                    
                    if visualize:
                        cv2.imshow(window_name, drawn_frame)
                        if cv2.waitKey(1) & 0xFF == 27:  # ESC
                            break
                    
                    if video_out_path:
                        if video_writer is None:
                            h, w = drawn_frame.shape[:2]
                            _ensure_dir(Path(video_out_path))
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            video_writer = cv2.VideoWriter(str(video_out_path), fourcc, vc_fps, (w, h))
                        video_writer.write(drawn_frame)
                
                frame_count += 1
                
                # 每 100 幀輸出進度
                if frame_count % 100 == 0:
                    logging.info("已處理 %d 幀，檢測到 %d 個網格點", frame_count, len(accum))
            
            # 關閉影像來源
            if isinstance(source, (str, int)):
                cap.release()
            
            # 輸出資料
            if output_path:
                _ensure_dir(Path(output_path))
                df = pd.DataFrame(accum, columns=["id", "keypoints", "model", "coor_x", "coor_y", "cm"])
                if out_is_parquet:
                    df.to_parquet(output_path, index=False)
                else:
                    df.to_csv(output_path, index=False)
                logging.info("已儲存臉部網格點記錄到 %s (%d 列)", output_path, len(df))
        
        finally:
            if video_writer is not None:
                video_writer.release()
            if visualize:
                cv2.destroyAllWindows()

    def _draw_face_mesh_points(self, frame: np.ndarray, rows: List[Tuple[int, str, str, float, float, Any]]) -> np.ndarray:
        """
        在影像上繪製臉部網格點
        """
        drawn_frame = frame.copy()
        
        # 按臉部 ID 分組網格點
        face_mesh_points = {}
        for row in rows:
            face_id, mesh_point_name, _, x, y, _ = row
            if face_id not in face_mesh_points:
                face_mesh_points[face_id] = []
            face_mesh_points[face_id].append((mesh_point_name, int(x), int(y)))
        
        # 為每個臉部繪製網格點
        for face_id, mesh_points in face_mesh_points.items():
            # 繪製網格點
            for mesh_point_name, x, y in mesh_points:
                # 根據網格點類型使用不同顏色
                if "eye" in mesh_point_name:
                    color = (0, 255, 0)  # 綠色眼睛
                elif "nose" in mesh_point_name:
                    color = (255, 0, 0)  # 藍色鼻子
                elif "mouth" in mesh_point_name:
                    color = (0, 0, 255)  # 紅色嘴巴
                elif "eyebrow" in mesh_point_name:
                    color = (255, 255, 0)  # 青色眉毛
                elif "face_oval" in mesh_point_name:
                    color = (255, 0, 255)  # 洋紅色臉部輪廓
                else:
                    color = (128, 128, 128)  # 灰色其他點
                
                cv2.circle(drawn_frame, (x, y), 2, color, -1)
                
                # 只為重要點顯示名稱（避免過於擁擠）
                if not self.use_full_mesh or "point_" not in mesh_point_name:
                    cv2.putText(drawn_frame, mesh_point_name, (x+3, y-3), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # 繪製臉部 ID
            if mesh_points:
                x, y = mesh_points[0][1], mesh_points[0][2]
                cv2.putText(drawn_frame, f"Face {face_id}", (x, y-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return drawn_frame

    def process_image(self, image_path: str, output_path: Optional[str] = None) -> List[Tuple[int, str, str, float, float, Any]]:
        """
        處理單一影像檔案
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError(f"無法讀取影像: {image_path}")
        
        rows = self._rows_from_frame(frame, 0)
        
        if output_path:
            _ensure_dir(Path(output_path))
            df = pd.DataFrame(rows, columns=["id", "keypoints", "model", "coor_x", "coor_y", "cm"])
            if output_path.lower().endswith(".parquet"):
                df.to_parquet(output_path, index=False)
            else:
                df.to_csv(output_path, index=False)
            logging.info("已儲存臉部網格點記錄到 %s (%d 列)", output_path, len(df))
        
        return rows

    def get_mesh_statistics(self) -> Dict[str, Any]:
        """
        獲取網格點統計資訊
        """
        return {
            "total_points": len(self.mesh_points),
            "use_full_mesh": self.use_full_mesh,
            "use_simple_mesh": self.use_simple_mesh_points,
            "mesh_point_names": self.mesh_point_names,
            "model_label": self.model_label
        }
    
    def close(self):
        """
        關閉資源並清理
        """
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        logging.info("FaceMeshRecorder 已關閉")


def run_inference(
    source: Any,
    output: str = "face_mesh_records.csv",
    use_simple_mesh_points: bool = True,
    use_full_mesh: bool = False,
    device: Optional[str] = None,
    conf: float = 0.5,
    max_faces: int = 1,
    pixel_to_cm: Optional[float] = None,
    kp_score_min: float = 0.0,
    video_out: Optional[str] = None,
    visualize: bool = False,
    save_fps: Optional[float] = None,
    window_width: int = 640,
    window_height: int = 480,
    window_name: Optional[str] = None,
) -> None:
    """
    便利函式：執行臉部網格點檢測
    """
    rec = FaceMeshRecorder(
        use_simple_mesh_points=use_simple_mesh_points,
        use_full_mesh=use_full_mesh,
        device=device,
        conf=conf,
        max_faces=max_faces,
        pixel_to_cm=pixel_to_cm,
        kp_score_min=kp_score_min,
        window_width=window_width,
        window_height=window_height,
    )
    
    rec.process_source(
        source=source,
        output_path=output,
        video_out_path=video_out,
        visualize=visualize,
        save_fps=save_fps,
        window_name=window_name,
    )


def _build_parser() -> argparse.ArgumentParser:
    """建立命令列參數解析器"""
    p = argparse.ArgumentParser(description="MediaPipe 臉部網格點檢測記錄器")
    p.add_argument("--source", type=str, default="0", help="0 為網路攝影機，或影像/影片路徑")
    p.add_argument("--output", type=str, default="face_mesh_records.csv", help="輸出檔案路徑")
    p.add_argument("--use-simple-mesh-points", action="store_true", default=True, help="使用簡化網格點集（16點）")
    p.add_argument("--use-full-mesh", action="store_true", help="使用完整 468 點網格")
    p.add_argument("--device", type=str, default=None, help="運算裝置")
    p.add_argument("--conf", type=float, default=0.5, help="檢測信心閾值")
    p.add_argument("--max-faces", type=int, default=1, help="最大檢測臉部數量")
    p.add_argument("--pixel-to-cm", type=float, default=None, help="像素轉公分比例")
    p.add_argument("--kp-score-min", type=float, default=0.0, help="網格點最小分數")
    p.add_argument("--video-out", type=str, default=None, help="輸出影片路徑")
    p.add_argument("--visualize", action="store_true", help="顯示視窗")
    p.add_argument("--save-fps", type=float, default=None, help="儲存影片 FPS")
    p.add_argument("--window-width", type=int, default=1080, help="視窗寬度（預設：1080）")
    p.add_argument("--window-height", type=int, default=1920, help="視窗高度（預設：1920）")
    p.add_argument("--window-name", type=str, default="Face Mesh Detection", help="視窗名稱")
    p.add_argument("--loglevel", type=str, default="INFO", help="日誌等級")
    return p


def main() -> None:
    """主程式入口點"""
    args = _build_parser().parse_args()
    
    # 設定日誌
    logging.basicConfig(
        level=getattr(logging, str(args.loglevel).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    # 處理來源參數
    source: Any
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    # 執行檢測
    run_inference(
        source=source,
        output=args.output,
        use_simple_mesh_points=args.use_simple_mesh_points,
        use_full_mesh=args.use_full_mesh,
        device=args.device,
        conf=args.conf,
        max_faces=args.max_faces,
        pixel_to_cm=args.pixel_to_cm,
        kp_score_min=args.kp_score_min,
        video_out=args.video_out,
        visualize=args.visualize,
        save_fps=args.save_fps,
        window_width=args.window_width,
        window_height=args.window_height,
        window_name=args.window_name,
    )


if __name__ == "__main__":
    main()
