# pose_recorder.py
# YOLOv11-pose 骨架偵測與追蹤記錄器
# 支援直接執行與模組 import 使用

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any, Tuple, Generator

import numpy as np
import pandas as pd
import cv2

# 設定環境變數強制使用 lapx
import os
os.environ["ULTRALYTICS_NO_LAP_CHECK"] = "1"
os.environ["ULTRALYTICS_TRACKER"] = "bytetrack"

try:
    from ultralytics import YOLO  # YOLOv11-pose
except Exception as e:
    raise RuntimeError("Ultralytics (YOLO) is required. Please `pip install ultralytics`.") from e

# 導入新的視覺化模組
try:
    from .pose_visualizer import PoseVisualizer, PosePerson, create_pose_person_from_result
except ImportError:
    try:
        from pose_visualizer import PoseVisualizer, PosePerson, create_pose_person_from_result
    except ImportError:
        PoseVisualizer = None
        PosePerson = None
        create_pose_person_from_result = None


# ---- Keypoint spec (COCO-17) ----
KEYPOINTS_C17: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def _ensure_dir(p: Path) -> None:
    """確保輸出目錄存在"""
    p.parent.mkdir(parents=True, exist_ok=True)


class PoseRecorder:
    """
    記錄每幀、每個ID、每個關鍵點的資料列：
    <id><keypoints><model><coor_x><coor_y><cm>
    """

    def __init__(
        self,
        weights: str = "yolo11n-pose.pt",
        tracker: Optional[str] = "bytetrack.yaml",
        device: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.7,
        pixel_to_cm: Optional[float] = None,
        kp_score_min: float = 0.0,
        model_label: Optional[str] = None,
    ) -> None:
        self.weights = weights
        self.tracker = tracker
        self.device = device
        self.conf = conf
        self.iou = iou
        self.pixel_to_cm = pixel_to_cm
        self.kp_score_min = kp_score_min
        self.model_label = model_label or Path(weights).name

        # 載入模型
        self.model = YOLO(self.weights)
        logging.info("已載入模型: %s", self.model_label)

    def _iter_track_results(
        self, source: Any
    ) -> Generator[Any, None, None]:
        """
        使用 Ultralytics .track(stream=True, persist=True) 串流追蹤結果
        """
        # 注意：task 由模型推斷（pose）。persist 保持 ID 一致
        results_gen = self.model.track(
            source=source,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker,
            device=self.device,
            stream=True,
            persist=True,
            verbose=False,
        )
        for r in results_gen:
            yield r

    def _rows_from_result(self, r: Any) -> List[Dict[str, Any]]:
        """
        將一個結果轉換為統一的關鍵點格式
        """
        rows: List[Dict[str, Any]] = []

        # boxes.id: (N,) tensor-like 或 None
        ids = None
        if hasattr(r, "boxes") and hasattr(r.boxes, "id") and r.boxes.id is not None:
            try:
                ids = r.boxes.id.cpu().numpy().astype(int)
            except Exception:
                ids = np.array([-1] * len(r))

        # keypoints: 形狀 (N, 17, 3)
        kps = getattr(r, "keypoints", None)
        if kps is None or kps.data is None:
            return rows

        kp_arr = kps.data.cpu().numpy()  # (N, 17, 3)
        n = kp_arr.shape[0]
        if ids is None:
            ids = np.arange(n, dtype=int)

        for i in range(n):
            pid = int(ids[i]) if i < len(ids) else -1
            keypoints = {}
            
            for k_idx, k_name in enumerate(KEYPOINTS_C17):
                x, y, score = kp_arr[i, k_idx, :]
                if np.isnan(x) or np.isnan(y):
                    continue
                if score < self.kp_score_min:
                    continue
                
                keypoints[k_name] = {
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(score)
                }
            
            if keypoints:  # 只添加有有效關鍵點的結果
                rows.append({
                    "id": pid,
                    "keypoints": keypoints,
                    "model": self.model_label
                })
        
        return rows

    def process_source(
        self,
        source: Any,
        output_path: Optional[str] = "records.csv",
        video_out_path: Optional[str] = None,
        visualize: bool = False,
        save_fps: Optional[float] = None,
        enhanced_visualization: bool = False,
        show_analysis: bool = False,
    ) -> None:
        """
        在來源上執行追蹤+骨架偵測，並將資料列輸出到 CSV/Parquet
        """
        writer = None
        video_writer = None
        out_is_parquet = bool(output_path and str(output_path).lower().endswith(".parquet"))
        accum: List[Tuple[int, str, str, float, float, Any]] = []

        vc_fps = None
        
        # 初始化增強視覺化器
        visualizer = None
        if enhanced_visualization and PoseVisualizer is not None:
            visualizer = PoseVisualizer(
                show_confidence=True,
                show_keypoint_names=False,
                line_thickness=3,
                keypoint_radius=5
            )

        try:
            for r in self._iter_track_results(source):
                # 影片幀 (BGR) 用於繪製/視覺化
                frame = getattr(r, "orig_img", None)

                # 收集資料列
                rows = self._rows_from_result(r)
                accum.extend(rows)

                # 繪製（如需要）
                if (visualize or video_out_path) and frame is not None:
                    if enhanced_visualization and visualizer is not None:
                        # 使用增強視覺化
                        drawn_frame = frame.copy()
                        
                        # 處理每個檢測到的人物
                        if hasattr(r, "boxes") and hasattr(r.boxes, "id") and r.boxes.id is not None:
                            try:
                                ids = r.boxes.id.cpu().numpy().astype(int)
                                for i, person_id in enumerate(ids):
                                    # 創建 PosePerson 物件
                                    pose_person = create_pose_person_from_result(r, person_id)
                                    
                                    # 繪製骨架
                                    drawn_frame = visualizer.draw_skeleton(drawn_frame, pose_person)
                                    
                                    # 繪製身體輪廓
                                    drawn_frame = visualizer.draw_body_contour(drawn_frame, pose_person)
                                    
                                    # 繪製姿態分析（如果需要）
                                    if show_analysis:
                                        analysis = visualizer.analyze_pose(pose_person)
                                        drawn_frame = visualizer.draw_pose_analysis(drawn_frame, analysis)
                                
                                plotted = drawn_frame
                            except Exception as e:
                                logging.warning("增強視覺化失敗，使用預設繪製: %s", e)
                                plotted = r.plot()
                        else:
                            plotted = r.plot()
                    else:
                        # 使用預設的 Ultralytics 繪製
                        plotted = r.plot()  # ndarray BGR
                    
                    if visualize:
                        window_name = "Enhanced Pose Detection" if enhanced_visualization else "pose_recorder"
                        cv2.imshow(window_name, plotted)
                        if cv2.waitKey(1) & 0xFF == 27:  # ESC
                            break

                    if video_out_path:
                        if video_writer is None:
                            h, w = plotted.shape[:2]
                            if save_fps is None:
                                # 嘗試從來源恢復 fps
                                try:
                                    cap = cv2.VideoCapture(source if isinstance(source, (str, int)) else 0)
                                    vc_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                                    cap.release()
                                except Exception:
                                    vc_fps = 30.0
                            else:
                                vc_fps = float(save_fps)

                            _ensure_dir(Path(video_out_path))
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            video_writer = cv2.VideoWriter(str(video_out_path), fourcc, vc_fps, (w, h))
                        video_writer.write(plotted)

            # 輸出
            if output_path:
                _ensure_dir(Path(output_path))
                df = pd.DataFrame(accum, columns=["id", "keypoints", "model", "coor_x", "coor_y", "cm"])
                if out_is_parquet:
                    df.to_parquet(output_path, index=False)
                else:
                    df.to_csv(output_path, index=False)
                logging.info("已儲存記錄到 %s (%d 列)", output_path, len(df))

        finally:
            if video_writer is not None:
                video_writer.release()
            if visualize:
                cv2.destroyAllWindows()


def run_inference(
    source: Any,
    weights: str = "yolo11n-pose.pt",
    output: str = "records.csv",
    tracker: Optional[str] = "bytetrack.yaml",
    device: Optional[str] = None,
    conf: float = 0.25,
    iou: float = 0.7,
    pixel_to_cm: Optional[float] = None,
    kp_score_min: float = 0.0,
    video_out: Optional[str] = None,
    visualize: bool = False,
    save_fps: Optional[float] = None,
    enhanced_visualization: bool = False,
    show_analysis: bool = False,
) -> None:
    """便利函式：建立 PoseRecorder 並執行推論"""
    rec = PoseRecorder(
        weights=weights, tracker=tracker, device=device,
        conf=conf, iou=iou, pixel_to_cm=pixel_to_cm, kp_score_min=kp_score_min
    )
    rec.process_source(
        source=source,
        output_path=output,
        video_out_path=video_out,
        visualize=visualize,
        save_fps=save_fps,
        enhanced_visualization=enhanced_visualization,
        show_analysis=show_analysis,
    )


def _build_parser() -> argparse.ArgumentParser:
    """建立命令列參數解析器"""
    p = argparse.ArgumentParser(description="YOLOv11-pose 記錄器")
    p.add_argument("--source", type=str, default="0", help="0 為網路攝影機，或路徑/URL")
    p.add_argument("--weights", type=str, default="yolo11n-pose.pt")
    p.add_argument("--device", type=str, default=None, help="cpu / cuda:0 / mps")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--tracker", type=str, default="bytetrack.yaml")
    p.add_argument("--output", type=str, default="records.csv")
    p.add_argument("--video-out", type=str, default=None)
    p.add_argument("--visualize", type=str, default="false")
    p.add_argument("--save-fps", type=float, default=None)
    p.add_argument("--pixel-to-cm", type=float, default=None)
    p.add_argument("--kp-score-min", type=float, default=0.0)
    p.add_argument("--enhanced-viz", action="store_true", help="啟用增強視覺化")
    p.add_argument("--show-analysis", action="store_true", help="顯示姿態分析")
    p.add_argument("--loglevel", type=str, default="INFO")
    return p


def _to_bool(s: str) -> bool:
    """將字串轉換為布林值"""
    return str(s).lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    """主程式入口點"""
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.loglevel).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    source: Any
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    run_inference(
        source=source,
        weights=args.weights,
        output=args.output,
        tracker=args.tracker,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        pixel_to_cm=args.pixel_to_cm,
        kp_score_min=args.kp_score_min,
        video_out=args.video_out,
        visualize=_to_bool(args.visualize),
        save_fps=args.save_fps,
        enhanced_visualization=args.enhanced_viz,
        show_analysis=args.show_analysis,
    )


if __name__ == "__main__":
    main()