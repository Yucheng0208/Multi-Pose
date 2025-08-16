#!/usr/bin/env python3
"""
YOLOv11-pose 骨架偵測與 MediaPipe 臉部網格點檢測整合系統
主程式入口點 - 支援多模態整合

使用方式:
    # 骨架檢測
    python main.py --mode pose --source 0 --visualize
    python main.py --mode pose --source video.mp4 --output pose_results.csv
    
    # 臉部網格檢測
    python main.py --mode face --source 0 --visualize
    python main.py --mode face --source video.mp4 --output face_mesh_results.csv
    
    # 手部檢測
    python main.py --mode hand --source 0 --visualize
    python main.py --mode hand --source video.mp4 --output hand_results.csv
    
    # 多模態整合檢測
    python main.py --mode multimodal --source 0 --visualize
    python main.py --mode multimodal --source video.mp4 --output multimodal_results.csv
    
    # 同時檢測（實驗性功能）
    python main.py --mode both --source 0 --visualize
"""

import sys
import logging
import os
import argparse
import cv2
from pathlib import Path
from typing import Optional, Any
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated",
    category=UserWarning,
    module=r"google\.protobuf\.symbol_database"
)

# 設定環境變數強制使用 lapx
os.environ["ULTRALYTICS_NO_LAP_CHECK"] = "1"
os.environ["ULTRALYTICS_TRACKER"] = "bytetrack"

# 添加 src 目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from pose_recorder import PoseRecorder, run_inference as run_pose_inference
    from face_recorder import FaceMeshRecorder, run_inference as run_face_inference
    POSE_AVAILABLE = True
    FACE_AVAILABLE = True
except ImportError as e:
    if "pose_recorder" in str(e):
        POSE_AVAILABLE = False
        logging.warning("骨架檢測模組無法載入: %s", e)
    elif "face_recorder" in str(e):
        FACE_AVAILABLE = False
        logging.warning("臉部網格檢測模組無法載入: %s", e)
    else:
        raise

# 嘗試導入新的多模態模組
try:
    from src import HandRecorder, MultimodalProcessor, MultimodalDataIntegrator, ModalityType, FaceMeshRecorder
    HAND_AVAILABLE = True
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    if "hand_recorder" in str(e):
        HAND_AVAILABLE = False
        logging.warning("手部檢測模組無法載入: %s", e)
    elif "multimodal_processor" in str(e):
        MULTIMODAL_AVAILABLE = False
        logging.warning("多模態處理器無法載入: %s", e)
    else:
        HAND_AVAILABLE = False
        MULTIMODAL_AVAILABLE = False
        logging.warning("多模態模組無法載入: %s", e)


def _build_parser() -> argparse.ArgumentParser:
    """建立統一的命令列參數解析器"""
    parser = argparse.ArgumentParser(
        description="YOLOv11-pose 骨架偵測與 MediaPipe 臉部網格點檢測整合系統 - 支援多模態整合"
    )
    
    # 基本參數
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["pose", "face", "hand", "multimodal", "both"], 
        default="pose",
        help="檢測模式：pose(骨架), face(臉部網格), hand(手部), multimodal(多模態整合), both(兩者)"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default="0",
        help="影像來源：0(網路攝影機) 或檔案路徑"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results.csv",
        help="輸出檔案路徑"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="顯示視窗"
    )
    
    # 骨架檢測參數
    pose_group = parser.add_argument_group("骨架檢測參數")
    pose_group.add_argument(
        "--pose-weights", 
        type=str, 
        default="yolo11n-pose.pt",
        help="YOLO 權重檔案路徑"
    )
    pose_group.add_argument(
        "--pose-conf", 
        type=float, 
        default=0.25,
        help="骨架檢測信心閾值"
    )
    pose_group.add_argument(
        "--pose-iou", 
        type=float, 
        default=0.7,
        help="骨架檢測 IoU 閾值"
    )
    pose_group.add_argument(
        "--pose-tracker", 
        type=str, 
        default="bytetrack.yaml",
        help="追蹤器設定檔案"
    )
    pose_group.add_argument(
        "--enhanced-viz", 
        action="store_true",
        help="啟用增強骨架視覺化（彩色骨架、身體輪廓）"
    )
    pose_group.add_argument(
        "--show-analysis", 
        action="store_true",
        help="顯示姿態分析資訊（品質評估、身體部位統計）"
    )
    
    # 臉部網格檢測參數
    face_group = parser.add_argument_group("臉部網格檢測參數")
    face_group.add_argument(
        "--face-conf", 
        type=float, 
        default=0.5,
        help="臉部網格檢測信心閾值"
    )
    face_group.add_argument(
        "--face-max-faces", 
        type=int, 
        default=1,
        help="最大檢測臉部數量"
    )
    face_group.add_argument(
        "--face-use-simple-mesh-points", 
        action="store_true", 
        default=True,
        help="使用簡化網格點集（16點）"
    )
    face_group.add_argument(
        "--face-use-full-mesh", 
        action="store_true",
        help="使用完整 468 點網格"
    )
    face_group.add_argument(
        "--face-window-width", 
        type=int, 
        default=1920,
        help="臉部檢測視窗寬度（預設：1080）"
    )
    face_group.add_argument(
        "--face-window-height", 
        type=int, 
        default=1080,
        help="臉部檢測視窗高度（預設：1920）"
    )
    face_group.add_argument(
        "--face-window-name", 
        type=str, 
        default="Face Mesh Detection",
        help="臉部檢測視窗名稱"
    )
    
    # 手部檢測參數
    hand_group = parser.add_argument_group("手部檢測參數")
    hand_group.add_argument(
        "--hand-max-hands", 
        type=int, 
        default=2,
        help="最大檢測手部數量"
    )
    hand_group.add_argument(
        "--hand-model-complexity", 
        type=int, 
        default=1,
        choices=[0, 1],
        help="手部檢測模型複雜度 (0: 快速, 1: 準確)"
    )
    hand_group.add_argument(
        "--hand-detection-conf", 
        type=float, 
        default=0.5,
        help="手部檢測信心閾值"
    )
    hand_group.add_argument(
        "--hand-tracking-conf", 
        type=float, 
        default=0.5,
        help="手部追蹤信心閾值"
    )
    
    # 多模態整合參數
    multimodal_group = parser.add_argument_group("多模態整合參數")
    multimodal_group.add_argument(
        "--enable-face", 
        action="store_true", 
        default=True,
        help="啟用臉部檢測"
    )
    multimodal_group.add_argument(
        "--enable-pose", 
        action="store_true", 
        default=True,
        help="啟用姿態檢測"
    )
    multimodal_group.add_argument(
        "--enable-hand", 
        action="store_true", 
        default=True,
        help="啟用手部檢測"
    )
    multimodal_group.add_argument(
        "--max-workers", 
        type=int, 
        default=3,
        help="並行處理的最大執行緒數"
    )
    multimodal_group.add_argument(
        "--session-id", 
        type=str, 
        default="multimodal_session",
        help="多模態會話 ID"
    )
    multimodal_group.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="多模態輸出目錄"
    )
    multimodal_group.add_argument(
        "--save-video", 
        action="store_true",
        help="儲存處理後的影片"
    )
    multimodal_group.add_argument(
        "--save-csv", 
        action="store_true", 
        default=True,
        help="儲存 CSV 資料"
    )
    
    # 通用參數
    common_group = parser.add_argument_group("通用參數")
    common_group.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="運算裝置 (cpu/cuda:0/mps)"
    )
    common_group.add_argument(
        "--pixel-to-cm", 
        type=float, 
        default=None,
        help="像素轉公分比例"
    )
    common_group.add_argument(
        "--video-out", 
        type=str, 
        default=None,
        help="輸出影片路徑"
    )
    common_group.add_argument(
        "--save-fps", 
        type=float, 
        default=None,
        help="儲存影片 FPS"
    )
    common_group.add_argument(
        "--loglevel", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日誌等級"
    )
    
    return parser


def run_pose_detection(args: argparse.Namespace) -> None:
    """執行骨架檢測"""
    if not POSE_AVAILABLE:
        raise RuntimeError("骨架檢測模組無法使用")
    
    logging.info("啟動骨架檢測...")
    
    # 處理來源參數
    source: Any
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    # 執行骨架檢測
    run_pose_inference(
        source=source,
        weights=args.pose_weights,
        output=args.output,
        tracker=args.pose_tracker,
        device=args.device,
        conf=args.pose_conf,
        iou=args.pose_iou,
        pixel_to_cm=args.pixel_to_cm,
        video_out=args.video_out,
        visualize=args.visualize,
        save_fps=args.save_fps,
        enhanced_visualization=args.enhanced_viz,
        show_analysis=args.show_analysis,
    )


def run_face_mesh_detection(args: argparse.Namespace) -> None:
    """執行臉部網格檢測"""
    if not FACE_AVAILABLE:
        raise RuntimeError("臉部網格檢測模組無法使用")
    
    logging.info("啟動臉部網格檢測...")
    
    # 處理來源參數
    source: Any
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    # 執行臉部網格檢測
    run_face_inference(
        source=source,
        output=args.output,
        use_simple_mesh_points=args.face_use_simple_mesh_points,
        use_full_mesh=args.face_use_full_mesh,
        device=args.device,
        conf=args.face_conf,
        max_faces=args.face_max_faces,
        pixel_to_cm=args.pixel_to_cm,
        video_out=args.video_out,
        visualize=args.visualize,
        save_fps=args.save_fps,
        window_width=args.face_window_width,
        window_height=args.face_window_height,
        window_name=args.face_window_name,
    )


def run_hand_detection(args: argparse.Namespace) -> None:
    """執行手部檢測"""
    if not HAND_AVAILABLE:
        raise RuntimeError("手部檢測模組無法使用")
    
    logging.info("啟動手部檢測...")
    
    # 處理來源參數
    source: Any
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    # 創建手部檢測器
    hand_recorder = HandRecorder(
        max_num_hands=args.hand_max_hands,
        model_complexity=args.hand_model_complexity,
        min_detection_confidence=args.hand_detection_conf,
        min_tracking_confidence=args.hand_tracking_conf,
        pixel_to_cm=args.pixel_to_cm
    )
    
    try:
        # 處理影片
        df = hand_recorder.process_video(
            source=source,
            output_csv=Path(args.output) if args.output else None,
            output_video=Path(args.video_out) if args.video_out else None,
            show_video=args.visualize,
            save_frames=False
        )
        
        logging.info("手部檢測完成，共記錄 %d 筆資料", len(df))
        
        # 儲存結果
        if args.output:
            df.to_csv(args.output, index=False)
            logging.info("手部資料已儲存至: %s", args.output)
    
    finally:
        hand_recorder.close()


def run_multimodal_detection(args: argparse.Namespace) -> None:
    """執行多模態整合檢測"""
    if not MULTIMODAL_AVAILABLE:
        raise RuntimeError("多模態處理器無法使用")
    
    logging.info("啟動多模態整合檢測...")
    
    # 處理來源參數
    source: Any
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    # 創建多模態處理器
    processor = MultimodalProcessor(
        # 臉部檢測參數
        face_model="face_landmarker.task",
        face_num_faces=args.face_max_faces,
        face_min_detection_confidence=args.face_conf,
        face_min_tracking_confidence=args.face_conf,
        
        # 姿態檢測參數
        pose_weights=args.pose_weights,
        pose_tracker=args.pose_tracker,
        pose_conf=args.pose_conf,
        pose_iou=args.pose_iou,
        
        # 手部檢測參數
        hand_max_num_hands=args.hand_max_hands,
        hand_model_complexity=args.hand_model_complexity,
        hand_min_detection_confidence=args.hand_detection_conf,
        hand_min_tracking_confidence=args.hand_tracking_conf,
        
        # 通用參數
        pixel_to_cm=args.pixel_to_cm,
        enable_face=args.enable_face,
        enable_pose=args.enable_pose,
        enable_hand=args.enable_hand,
        max_workers=args.max_workers,
        output_dir=args.output_dir
    )
    
    try:
        # 設定輸出目錄
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 處理影片
        result = processor.process_video(
            source=source,
            session_id=args.session_id,
            output_dir=output_dir,
            show_video=args.visualize,
            save_video=args.save_video,
            save_csv=args.save_csv
        )
        
        logging.info("多模態檢測完成: %s", result)
        
        # 顯示統計資訊（在關閉處理器之前）
        if args.save_csv:
            try:
                stats = processor.get_session_statistics(args.session_id)
                logging.info("會話統計: %s", stats)
            except Exception as e:
                logging.warning("無法獲取會話統計: %s", e)
    
    finally:
        processor.close()


def run_both_detections(args: argparse.Namespace) -> None:
    """同時執行骨架和臉部網格檢測（實驗性功能）"""
    if not POSE_AVAILABLE or not FACE_AVAILABLE:
        raise RuntimeError("骨架或臉部網格檢測模組無法使用")
    
    logging.info("啟動雙模組檢測...")
    logging.warning("雙模組檢測功能為實驗性功能，可能不穩定")
    
    # 處理來源參數
    source: Any
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    # 建立檢測器實例
    pose_recorder = PoseRecorder(
        weights=args.pose_weights,
        tracker=args.pose_tracker,
        device=args.device,
        conf=args.pose_conf,
        iou=args.pose_iou,
        pixel_to_cm=args.pixel_to_cm,
    )
    
    face_recorder = FaceMeshRecorder(
        use_simple_mesh_points=args.face_use_simple_mesh_points,
        use_full_mesh=args.face_use_full_mesh,
        device=args.device,
        conf=args.face_conf,
        max_faces=args.face_max_faces,
        pixel_to_cm=args.pixel_to_cm,
        window_width=args.face_window_width,
        window_height=args.face_window_height,
    )
    
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
    
    try:
        frame_count = 0
        pose_results = []
        face_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 骨架檢測（簡化處理）
            try:
                # 使用 pose_recorder 的內部方法進行檢測
                pose_rows = pose_recorder._rows_from_result(
                    next(pose_recorder._iter_track_results([frame]))
                )
                pose_results.extend(pose_rows)
            except Exception as e:
                logging.warning("骨架檢測失敗: %s", e)
                pose_rows = []
            
            # 臉部網格檢測
            try:
                face_rows = face_recorder._rows_from_frame(frame, frame_count)
                face_results.extend(face_rows)
            except Exception as e:
                logging.warning("臉部網格檢測失敗: %s", e)
                face_rows = []
            
            # 視覺化（如果需要）
            if args.visualize:
                drawn_frame = frame.copy()
                
                # 繪製骨架檢測結果（簡化）
                for row in pose_rows:
                    _, _, _, x, y, _ = row
                    cv2.circle(drawn_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                
                # 繪製臉部網格檢測結果
                drawn_frame = face_recorder._draw_face_mesh_points(drawn_frame, face_rows)
                
                cv2.imshow("雙模組檢測", drawn_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
            
            frame_count += 1
            
            # 每 100 幀輸出進度
            if frame_count % 100 == 0:
                logging.info("已處理 %d 幀，骨架關鍵點: %d, 臉部網格點: %d", 
                           frame_count, len(pose_results), len(face_results))
        
        # 合併結果並輸出
        all_results = pose_results + face_results
        
        if args.output:
            # 根據副檔名決定輸出格式
            if args.output.lower().endswith('.parquet'):
                import pandas as pd
                df = pd.DataFrame(all_results, 
                                columns=["id", "keypoints", "model", "coor_x", "coor_y", "cm"])
                df.to_parquet(args.output, index=False)
            else:
                import pandas as pd
                df = pd.DataFrame(all_results, 
                                columns=["id", "keypoints", "model", "coor_x", "coor_y", "cm"])
                df.to_csv(args.output, index=False)
            
            logging.info("已儲存整合結果到 %s (%d 列)", args.output, len(df))
    
    finally:
        if isinstance(source, (str, int)):
            cap.release()
        if args.visualize:
            cv2.destroyAllWindows()


def main() -> None:
    """主程式入口點"""
    # 解析命令列參數
    parser = _build_parser()
    args = parser.parse_args()
    
    # 設定日誌
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # 檢查模組可用性
    if args.mode == "pose" and not POSE_AVAILABLE:
        logging.error("骨架檢測模組無法使用")
        sys.exit(1)
    
    if args.mode == "face" and not FACE_AVAILABLE:
        logging.error("臉部網格檢測模組無法使用")
        sys.exit(1)
    
    if args.mode == "hand" and not HAND_AVAILABLE:
        logging.error("手部檢測模組無法使用")
        sys.exit(1)
    
    if args.mode == "multimodal" and not MULTIMODAL_AVAILABLE:
        logging.error("多模態處理器無法使用")
        sys.exit(1)
    
    if args.mode == "both" and (not POSE_AVAILABLE or not FACE_AVAILABLE):
        logging.error("雙模組檢測需要兩個模組都可用")
        sys.exit(1)
    
    try:
        # 根據模式執行相應的檢測
        if args.mode == "pose":
            run_pose_detection(args)
        elif args.mode == "face":
            run_face_mesh_detection(args)
        elif args.mode == "hand":
            run_hand_detection(args)
        elif args.mode == "multimodal":
            run_multimodal_detection(args)
        elif args.mode == "both":
            run_both_detections(args)
        else:
            logging.error("未知的檢測模式: %s", args.mode)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n程式已由使用者中斷")
        sys.exit(0)
    except ImportError as e:
        if "lap" in str(e):
            print("\n❌ 依賴套件問題：lap 套件未正確載入")
            print("解決方案：")
            print("1. 執行 reset_env.bat 重置環境")
            print("2. 或手動重新啟動 conda 環境")
            print("3. 或執行：pip install --force-reinstall lap>=0.5.12")
        else:
            print(f"\n❌ 模組匯入錯誤: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error("程式執行錯誤: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()