# multimodal_processor.py
# 多模態處理器：整合臉部、姿態和手部檢測功能
# 提供統一的介面進行多模態資料收集和分析

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
import cv2

# 導入各模組
try:
    from .face_recorder import FaceMeshRecorder
    from .pose_recorder import PoseRecorder, KEYPOINTS_C17
    from .hand_recorder import HandRecorder
    from .unified_output_manager import UnifiedOutputManager, ModalityType
    from .multimodal_data import (
        MultimodalDataIntegrator,
        MultimodalAnalyzer,
        create_multimodal_session
    )
    from .debug_dumper import DebugDumper
except ImportError:
    # 如果無法導入，定義基本類型
    FaceMeshRecorder = None
    PoseRecorder = None
    HandRecorder = None
    UnifiedOutputManager = None
    MultimodalDataIntegrator = None
    MultimodalAnalyzer = None
    DebugDumper = None
    KEYPOINTS_C17 = []


class MultimodalProcessor:
    """多模態處理器：整合臉部、姿態和手部檢測"""
    
    def __init__(
        self,
        # 臉部檢測參數
        face_model: str = "face_landmarker.task",
        face_num_faces: int = 1,
        face_min_detection_confidence: float = 0.5,
        face_min_tracking_confidence: float = 0.5,
        
        # 姿態檢測參數
        pose_weights: str = "yolo11n-pose.pt",
        pose_tracker: Optional[str] = "bytetrack.yaml",
        pose_conf: float = 0.25,
        pose_iou: float = 0.7,
        
        # 手部檢測參數
        hand_max_num_hands: int = 2,
        hand_model_complexity: int = 1,
        hand_min_detection_confidence: float = 0.5,
        hand_min_tracking_confidence: float = 0.5,
        
        # 通用參數
        pixel_to_cm: Optional[float] = None,
        enable_face: bool = True,
        enable_pose: bool = True,
        enable_hand: bool = True,
        max_workers: int = 3,
        output_dir: str = "output"
    ):
        self.pixel_to_cm = pixel_to_cm
        self.enable_face = enable_face
        self.enable_pose = enable_pose
        self.enable_hand = enable_hand
        self.max_workers = max_workers
        self.output_dir = output_dir
        
        # 初始化各模組
        self.face_recorder = None
        self.pose_recorder = None
        self.hand_recorder = None
        
        # 初始化統一輸出管理器
        self.unified_output = None
        if UnifiedOutputManager:
            self.unified_output = UnifiedOutputManager(output_dir)
        
        # 偵錯資料輸出
        self.debug_dumper = DebugDumper(Path(output_dir) / "debug" / "multimodal_debug.jsonl", enabled=True)
        
        if self.enable_face and FaceMeshRecorder:
            try:
                self.face_recorder = FaceMeshRecorder(
                    model_label=face_model,
                    max_faces=face_num_faces,
                    conf=face_min_detection_confidence,
                    pixel_to_cm=pixel_to_cm,
                    use_full_mesh=True,  # 啟用完整的 468 點 mesh 模型
                    use_simple_mesh_points=False  # 停用簡化版本
                )
                logging.info("臉部檢測模組已初始化（468 點精細 mesh 模型）")
            except Exception as e:
                logging.warning("臉部檢測模組初始化失敗: %s", e)
                self.enable_face = False
        
        if self.enable_pose and PoseRecorder:
            try:
                self.pose_recorder = PoseRecorder(
                    weights=pose_weights,
                    tracker=pose_tracker,
                    conf=pose_conf,
                    iou=pose_iou,
                    pixel_to_cm=pixel_to_cm
                )
                logging.info("姿態檢測模組已初始化")
            except Exception as e:
                logging.warning("姿態檢測模組初始化失敗: %s", e)
                self.enable_pose = False
        
        if self.enable_hand and HandRecorder:
            try:
                self.hand_recorder = HandRecorder(
                    max_num_hands=hand_max_num_hands,
                    model_complexity=hand_model_complexity,
                    min_detection_confidence=hand_min_detection_confidence,
                    min_tracking_confidence=hand_min_tracking_confidence,
                    pixel_to_cm=pixel_to_cm
                )
                logging.info("手部檢測模組已初始化")
            except Exception as e:
                logging.warning("手部檢測模組初始化失敗: %s", e)
                self.enable_hand = False
        
        # 初始化多模態資料整合器
        try:
            if MultimodalDataIntegrator:
                self.integrator = MultimodalDataIntegrator()
                self.analyzer = MultimodalAnalyzer(self.integrator) if MultimodalAnalyzer else None
                logging.info("多模態資料整合器初始化成功")
            else:
                logging.warning("MultimodalDataIntegrator 類別不可用")
                self.integrator = None
                self.analyzer = None
        except Exception as e:
            logging.warning("多模態資料整合器初始化失敗: %s", e)
            self.integrator = None
            self.analyzer = None
        
        # 處理狀態
        self.is_processing = False
        self.current_session_id = None
        self.frame_count = 0
        
        # FPS 計算相關
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        logging.info("多模態處理器初始化完成")
        logging.info("啟用模組: 臉部=%s, 姿態=%s, 手部=%s", 
                    self.enable_face, self.enable_pose, self.enable_hand)
    
    def start_session(self, session_id: str, source_path: Optional[Path] = None) -> None:
        """開始新的多模態會話"""
        if not self.integrator:
            raise RuntimeError("多模態資料整合器未初始化")
        
        if self.is_processing:
            raise RuntimeError("已有會話正在進行中")
        
        self.integrator.create_session(session_id, source_path)
        self.current_session_id = session_id
        self.is_processing = True
        self.frame_count = 0
        
        logging.info("開始多模態會話: %s", session_id)
    
    def stop_session(self) -> None:
        """停止當前會話"""
        if not self.is_processing or not self.current_session_id:
            return
        
        if self.integrator:
            self.integrator.close_session(self.current_session_id)
        
        self.is_processing = False
        self.current_session_id = None
        logging.info("會話已停止，共處理 %d 幀", self.frame_count)
    
    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """處理單幀影像，進行多模態檢測"""
        if not self.is_processing:
            raise RuntimeError("沒有活動的會話")
        
        if timestamp is None:
            timestamp = time.time()
        
        # 並行處理各模態
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            # 臉部檢測
            if self.enable_face and self.face_recorder:
                future = executor.submit(self._detect_face, frame)
                futures['face'] = future
            
            # 姿態檢測
            if self.enable_pose and self.pose_recorder:
                future = executor.submit(self._detect_pose, frame)
                futures['pose'] = future
            
            # 手部檢測
            if self.enable_hand and self.hand_recorder:
                future = executor.submit(self._detect_hand, frame)
                futures['hand'] = future
            
            # 收集結果
            for modality, future in futures.items():
                try:
                    result = future.result(timeout=5.0)  # 5秒超時
                    results[modality] = result
                except Exception as e:
                    logging.warning("%s 檢測失敗: %s", modality, e)
                    results[modality] = None
        
        # 記錄到整合器
        if self.integrator:
            self.integrator.add_frame_data(
                frame_id=self.frame_count,
                timestamp=timestamp,
                face_data=results.get('face'),
                pose_data=results.get('pose'),
                hand_data=results.get('hand'),
                session_id=self.current_session_id
            )
        
        # 使用統一輸出管理器記錄資料
        if self.unified_output:
            frame_num = self.frame_count
            
            # 添加臉部資料
            if 'face' in results and results['face']:
                face_data = results['face']
                if face_data.get('detected') and 'faces_data' in face_data:
                    # 轉換為統一格式（相容 landmarks 或 mesh_points，支援 dict/list）
                    unified_face_data = []
                    for f_idx, face in enumerate(face_data['faces_data']):
                        raw = None
                        if isinstance(face, dict):
                            raw = face.get('landmarks') if 'landmarks' in face else face.get('mesh_points')
                        landmarks = {}
                        if isinstance(raw, dict):
                            for point_name, value in raw.items():
                                x = y = z = None
                                if isinstance(value, (list, tuple)) and len(value) >= 2:
                                    x = value[0]
                                    y = value[1]
                                    z = value[2] if len(value) > 2 else 0.0
                                elif isinstance(value, dict):
                                    x = value.get('x')
                                    y = value.get('y')
                                    z = value.get('z', 0.0)
                                if x is None or y is None:
                                    continue
                                conf = value.get('confidence') if isinstance(value, dict) else None
                                if conf is None:
                                    # 以 z 近似信心（歸一化到 0-1）
                                    try:
                                        conf = max(0.0, min(1.0, (float(z) + 1.0) / 2.0))
                                    except Exception:
                                        conf = 1.0
                                landmarks[str(point_name)] = {
                                    'x': float(x),
                                    'y': float(y),
                                    'confidence': float(conf)
                                }
                        elif isinstance(raw, list):
                            for idx, value in enumerate(raw):
                                x = y = z = None
                                if isinstance(value, (list, tuple)) and len(value) >= 2:
                                    x = value[0]
                                    y = value[1]
                                    z = value[2] if len(value) > 2 else 0.0
                                elif isinstance(value, dict):
                                    x = value.get('x')
                                    y = value.get('y')
                                    z = value.get('z', 0.0)
                                if x is None or y is None:
                                    continue
                                try:
                                    conf = value.get('confidence') if isinstance(value, dict) else max(0.0, min(1.0, (float(z) + 1.0) / 2.0))
                                except Exception:
                                    conf = 1.0
                                landmarks[f"point_{idx}"] = {
                                    'x': float(x),
                                    'y': float(y),
                                    'confidence': float(conf)
                                }
                        # face id 容錯
                        face_id = face.get('id', f_idx) if isinstance(face, dict) else f_idx
                        if landmarks:
                            unified_face_data.append({
                                'id': int(face_id) if isinstance(face_id, (int, float)) else f_idx,
                                'landmarks': landmarks,
                                'model': 'face_mesh'
                            })
                    
                    if unified_face_data:
                        self.unified_output.add_face_data(frame_num, unified_face_data)
            
            # 添加姿態資料
            if 'pose' in results and results['pose']:
                pose_data = results['pose']
                if pose_data.get('detected') and 'keypoints' in pose_data:
                    # 轉換為統一格式（支援 dict 或 list）
                    unified_pose_data = []
                    kp_names = KEYPOINTS_C17 if KEYPOINTS_C17 else []
                    for p_idx, person in enumerate(pose_data['keypoints']):
                        if not isinstance(person, dict):
                            continue
                        if 'keypoints' not in person:
                            continue
                        raw_kps = person['keypoints']
                        keypoints = {}
                        if isinstance(raw_kps, dict):
                            # 已是名稱->座標的格式
                            for kp_name, kp_data in raw_kps.items():
                                if not isinstance(kp_data, dict):
                                    continue
                                x = kp_data.get('x')
                                y = kp_data.get('y')
                                conf = kp_data.get('confidence', kp_data.get('conf', 0.0))
                                if x is None or y is None:
                                    continue
                                keypoints[str(kp_name)] = {
                                    'x': float(x),
                                    'y': float(y),
                                    'confidence': float(conf)
                                }
                        elif isinstance(raw_kps, list):
                            # list[ {x,y,conf?} ] -> 以索引或 KEYPOINTS_C17 命名
                            for idx, kp in enumerate(raw_kps):
                                if not isinstance(kp, dict):
                                    continue
                                x = kp.get('x')
                                y = kp.get('y')
                                conf = kp.get('confidence', kp.get('conf', 0.0))
                                if x is None or y is None:
                                    continue
                                name = (
                                    str(kp_names[idx]) if idx < len(kp_names) and kp_names else f"kp_{idx}"
                                )
                                keypoints[name] = {
                                    'x': float(x),
                                    'y': float(y),
                                    'confidence': float(conf)
                                }
                        # person id 容錯
                        person_id = person.get('id', p_idx)
                        if keypoints:
                            unified_pose_data.append({
                                'id': int(person_id),
                                'keypoints': keypoints,
                                'model': 'pose_detection'
                            })
                    
                    if unified_pose_data:
                        self.unified_output.add_pose_data(frame_num, unified_pose_data)
            
            # 添加手部資料
            if 'hand' in results and results['hand']:
                hand_data = results['hand']
                if hand_data.get('detected') and 'landmarks' in hand_data:
                    # 轉換為統一格式
                    unified_hand_data = []
                    for hand in hand_data['landmarks']:
                        if 'landmarks' in hand:
                            landmarks = {}
                            for lm_name, lm_data in hand['landmarks'].items():
                                landmarks[lm_name] = {
                                    'x': lm_data['x'],
                                    'y': lm_data['y'],
                                    'confidence': lm_data['confidence']
                                }
                            unified_hand_data.append({
                                'id': hand['id'],
                                'landmarks': landmarks,
                                'model': 'hand_detection'
                            })
                    
                    if unified_hand_data:
                        self.unified_output.add_hand_data(frame_num, unified_hand_data)
        
        # 計算 FPS
        self.fps_frame_count += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:  # 每秒更新一次 FPS
            self.current_fps = self.fps_frame_count / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
            self.fps_frame_count = 0
        
        self.frame_count += 1
        return results
    
    def _detect_face(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """臉部檢測"""
        try:
            if not self.face_recorder:
                return None
            
            # 使用臉部記錄器進行檢測
            if hasattr(self.face_recorder, '_detect_faces'):
                faces_data = self.face_recorder._detect_faces(frame)
                if faces_data:
                    # 提取網格點資料
                    mesh_points = {}
                    for face_data in faces_data:
                        if 'mesh_points' in face_data:
                            mesh_points.update(face_data['mesh_points'])
                    
                    return {
                        'detected': True,
                        'num_faces': len(faces_data),
                        'confidence': 0.8,
                        'mesh_points': mesh_points,
                        'faces_data': faces_data
                    }
                else:
                    return {
                        'detected': False,
                        'num_faces': 0,
                        'confidence': 0.0,
                        'mesh_points': {},
                        'faces_data': []
                    }
            else:
                # 如果沒有 _detect_faces 方法，創建一個簡單的檢測結果
                return {
                    'detected': True,
                    'num_faces': 1,
                    'confidence': 0.8,
                    'mesh_points': {},
                    'faces_data': []
                }
        except Exception as e:
            logging.error("臉部檢測錯誤: %s", e)
            return None
    
    def _detect_pose(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """姿態檢測"""
        try:
            if not self.pose_recorder:
                return None
            
            # 使用姿態記錄器進行檢測
            if hasattr(self.pose_recorder, 'model'):
                # 直接使用 YOLO 模型進行檢測
                results = self.pose_recorder.model(frame, verbose=False)
                if results and len(results) > 0:
                    # 轉換為標準格式
                    keypoints_data = []
                    for result in results:
                        if hasattr(result, 'keypoints') and result.keypoints is not None:
                            # 提取關鍵點資料
                            kps = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
                            if len(kps) > 0:
                                for person_idx in range(len(kps)):
                                    person_keypoints = []
                                    for kp_idx, (x, y, score) in enumerate(kps[person_idx]):
                                        if not (np.isnan(x) or np.isnan(y)):
                                            person_keypoints.append({
                                                'x': float(x),
                                                'y': float(y),
                                                'conf': float(score),
                                                'visible': float(score) if score > 0.1 else 0.0
                                            })
                                        else:
                                            person_keypoints.append({
                                                'x': 0.0,
                                                'y': 0.0,
                                                'conf': 0.0,
                                                'visible': 0.0
                                            })
                                    
                                    keypoints_data.append({
                                        'keypoints': person_keypoints,
                                        'bbox': result.boxes.xyxy[person_idx].cpu().numpy().tolist() if hasattr(result, 'boxes') and result.boxes.xyxy is not None else [],
                                        'conf': float(result.boxes.conf[person_idx]) if hasattr(result, 'boxes') and result.boxes.conf is not None else 0.0
                                    })
                    
                    return {
                        'detected': True,
                        'num_persons': len(keypoints_data),
                        'confidence': 0.8,
                        'keypoints': keypoints_data
                    }
                else:
                    return {
                        'detected': False,
                        'num_persons': 0,
                        'confidence': 0.0,
                        'keypoints': []
                    }
            else:
                # 如果沒有模型，創建一個簡單的檢測結果
                return {
                    'detected': True,
                    'num_persons': 1,
                    'confidence': 0.8,
                    'keypoints': []
                }
        except Exception as e:
            logging.error("姿態檢測錯誤: %s", e)
            return None
    
    def _detect_hand(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """手部檢測"""
        try:
            if not self.hand_recorder:
                return None
            
            # 使用手部記錄器進行檢測
            hands_data = self.hand_recorder._detect_hands(frame)
            
            if hands_data:
                return {
                    'detected': True,
                    'num_hands': len(hands_data),
                    'hands': hands_data,
                    'gestures': [hand['gesture'].value for hand in hands_data]
                }
            else:
                return {
                    'detected': False,
                    'num_hands': 0,
                    'hands': [],
                    'gestures': []
                }
        except Exception as e:
            logging.error("手部檢測錯誤: %s", e)
            return None
    
    def process_video(
        self,
        source: Any,
        session_id: str,
        output_dir: Optional[Path] = None,
        show_video: bool = False,
        save_video: bool = False,
        save_csv: bool = True
    ) -> Dict[str, Any]:
        """處理影片並進行多模態檢測"""
        # 開始會話
        self.start_session(session_id, Path(str(source)) if source else None)
        
        # 設定輸出
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 開啟影片
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
        out = None
        if save_video and output_dir:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video_path = output_dir / f"{session_id}_multimodal.mp4"
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        try:
            frame_id = 0
            self.debug_dumper.log("video_opened", {"fps": fps, "width": width, "height": height, "total_frames": total_frames})
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 進行多模態檢測
                try:
                    results = self.process_frame(frame, frame_id / fps)
                    # 記錄每幀摘要（避免輸出過大）
                    self.debug_dumper.log("frame_results", {
                        "frame_id": frame_id,
                        "face": self.debug_dumper.summarize(results.get('face')),
                        "pose": self.debug_dumper.summarize(results.get('pose')),
                        "hand": self.debug_dumper.summarize(results.get('hand')),
                    })
                except Exception as e:
                    self.debug_dumper.log_exception("process_frame", e, {"frame_id": frame_id})
                    raise
                
                # 繪製檢測結果
                annotated_frame = self._draw_results(frame, results)
                
                # 顯示處理進度
                if frame_id % 30 == 0:
                    progress = (frame_id / total_frames) * 100 if total_frames > 0 else frame_id
                    logging.info("處理進度: %.1f%% (%d/%d)", progress, frame_id, total_frames)
                
                # 儲存影片
                if out:
                    out.write(annotated_frame)
                
                # 顯示影片
                if show_video:
                    cv2.imshow('Multimodal Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_id += 1
        
        finally:
            cap.release()
            if out:
                out.release()
            if show_video:
                cv2.destroyAllWindows()
            
            # 停止會話
            self.stop_session()
        
        # 若未處理任何幀，直接返回並略過匯出與報告
        if frame_id == 0:
            logging.warning("沒有處理任何幀，略過資料匯出與報告生成")
            self.debug_dumper.log("no_frames_processed", {})
            return {
                'session_id': session_id,
                'total_frames': frame_id,
                'exported_files': [],
                'report': {'message': '沒有處理任何幀，未生成匯出與報告'}
            }
        
        # 匯出結果
        exported_files = []
        if save_csv and output_dir and self.integrator:
            try:
                exported_files = self.integrator.export_session_to_csv(
                    session_id, output_dir, separate_modalities=True
                )
                self.debug_dumper.log("export_session_to_csv", {"files": exported_files})
            except Exception as e:
                self.debug_dumper.log_exception("export_session_to_csv", e, {"session_id": session_id})
                logging.warning("CSV 匯出失敗: %s", e)
        
        # 使用統一輸出管理器保存所有幀的資料
        if self.unified_output:
            try:
                npy_files = self.unified_output.save_all_frames(session_id)
                self.debug_dumper.log("save_all_frames", {"files": [str(p) for p in npy_files]})
                exported_files.extend(npy_files)
                
                summary_csv = self.unified_output.save_summary_csv(session_id)
                if summary_csv:
                    exported_files.append(summary_csv)
                self.debug_dumper.log("save_summary_csv", {"file": str(summary_csv) if summary_csv else None})
                
                pickle_file = self.unified_output.export_to_pickle(session_id)
                if pickle_file:
                    exported_files.append(pickle_file)
                self.debug_dumper.log("export_to_pickle", {"file": str(pickle_file) if pickle_file else None})
                
                stats = self.unified_output.get_frame_statistics()
                logging.info("統一輸出統計: %s", stats)
                self.debug_dumper.log("frame_statistics", stats)
            except Exception as e:
                self.debug_dumper.log_exception("unified_output_save", e, {})
                logging.warning("統一輸出保存時發生錯誤: %s", e)
        
        # 生成報告
        report = {}
        if self.analyzer and frame_id > 0:  # 只有在處理了幀後才生成報告
            try:
                report = self.analyzer.generate_summary_report(session_id)
                self.debug_dumper.log("generate_summary_report", {"summary": self.debug_dumper.summarize(report)})
            except Exception as e:
                logging.error("生成摘要報告時發生錯誤: %s", e)
                self.debug_dumper.log_exception("generate_summary_report", e, {"session_id": session_id})
                report = {
                    'error': str(e),
                    'message': '報告生成失敗'
                }
        elif frame_id == 0:
            logging.warning("沒有處理任何幀，跳過報告生成")
            report = {
                'message': '沒有處理任何幀，無法生成報告'
            }
        
        result = {
            'session_id': session_id,
            'total_frames': frame_id,
            'exported_files': exported_files,
            'report': report
        }
        self.debug_dumper.log("process_video_result", result)
        return result
    
    def _draw_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """在影像上繪製多模態檢測結果，包含所有節點座標"""
        annotated_frame = frame.copy()
        frame_height, frame_width = annotated_frame.shape[:2]
        
        # 繪製臉部檢測結果和節點
        if 'face' in results and results['face']:
            face_data = results['face']
            if face_data.get('detected'):
                cv2.putText(annotated_frame, f"Face: {face_data.get('num_faces', 0)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 繪製臉部網格點
                if 'mesh_points' in face_data:
                    self._draw_face_mesh_points(annotated_frame, face_data['mesh_points'])
        
        # 繪製姿態檢測結果和節點
        if 'pose' in results and results['pose']:
            pose_data = results['pose']
            if pose_data.get('detected'):
                cv2.putText(annotated_frame, f"Pose: {pose_data.get('num_persons', 0)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # 繪製骨架關鍵點
                if 'keypoints' in pose_data:
                    self._draw_pose_keypoints(annotated_frame, pose_data['keypoints'])
        
        # 繪製手部檢測結果和節點
        if 'hand' in results and results['hand']:
            hand_data = results['hand']
            if hand_data.get('detected'):
                num_hands = hand_data.get('num_hands', 0)
                gestures = hand_data.get('gestures', [])
                cv2.putText(annotated_frame, f"Hands: {num_hands}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 顯示手部動作
                for i, gesture in enumerate(gestures[:3]):  # 最多顯示3個動作
                    cv2.putText(annotated_frame, f"G{i+1}: {gesture}", 
                               (10, 120 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 繪製手部關鍵點
                if 'hands' in hand_data:
                    self._draw_hand_landmarks(annotated_frame, hand_data['hands'])
        
        # 顯示幀計數和 FPS
        cv2.putText(annotated_frame, f"Frame: {self.frame_count}", 
                   (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 顯示 FPS（在 Frame 下方）
        cv2.putText(annotated_frame, f"FPS: {self.current_fps:.1f}", 
                   (frame_width - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 顯示座標資訊提示
        cv2.putText(annotated_frame, "Press 'C' to show coordinates", 
                   (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def _draw_face_mesh_points(self, image: np.ndarray, mesh_points: Dict[str, Tuple[int, int, float]]):
        """繪製臉部網格點和連線"""
        if not mesh_points:
            return
        
        # 根據網格點數量決定繪製策略
        num_points = len(mesh_points)
        
        if num_points > 100:  # 468 點精細模型
            # 繪製精細的臉部網格結構
            self._draw_detailed_face_mesh(image, mesh_points)
        else:
            # 繪製簡化版本
            self._draw_simple_face_mesh(image, mesh_points)
    
    def _draw_detailed_face_mesh(self, image: np.ndarray, mesh_points: Dict[str, Tuple[int, int, float]]):
        """繪製精細的臉部網格結構（468 點）"""
        # 定義重要的臉部區域連線
        face_connections = [
            # 臉部輪廓
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
            (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
            
            # 右眼輪廓
            (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
            (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133),
            
            # 左眼輪廓
            (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390), (390, 249), (249, 263),
            (362, 398), (398, 384), (384, 385), (385, 386), (386, 387), (387, 388), (388, 466), (466, 263),
            
            # 鼻子
            (1, 2), (2, 5), (5, 31), (31, 35), (35, 195), (195, 196), (196, 197), (197, 198),
            (198, 199), (199, 200), (200, 201), (201, 202), (202, 203), (203, 204), (204, 205), (205, 206), (206, 207),
            
            # 嘴巴
            (61, 84), (84, 17), (17, 314), (314, 405), (405, 320), (320, 307), (307, 375), (375, 321), (321, 308), (308, 324), (324, 318), (318, 78), (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308), (308, 321), (321, 375), (375, 307), (307, 320), (320, 405), (405, 314), (314, 17), (17, 84), (84, 61),
            
            # 眉毛
            (70, 63), (63, 105), (105, 66), (66, 107), (107, 55), (55, 65), (65, 52), (52, 53), (53, 46), (46, 70),
            (336, 296), (296, 334), (334, 293), (293, 300), (300, 276), (276, 283), (283, 282), (282, 295), (295, 285), (285, 336)
        ]
        
        # 繪製網格點
        for point_name, (x, y, z) in mesh_points.items():
            # 根據點的位置決定顏色和大小
            if "eye" in point_name.lower():
                color = (0, 255, 255)  # 黃色 - 眼睛
                radius = 2
            elif "nose" in point_name.lower():
                color = (255, 0, 255)  # 洋紅色 - 鼻子
                radius = 3
            elif "mouth" in point_name.lower():
                color = (255, 255, 0)  # 青色 - 嘴巴
                radius = 2
            elif "eyebrow" in point_name.lower():
                color = (128, 255, 128)  # 淺綠色 - 眉毛
                radius = 2
            else:
                color = (0, 255, 0)  # 綠色 - 其他
                radius = 1
            
            cv2.circle(image, (x, y), radius, color, -1)
        
        # 繪製連線（只繪製重要的連線以避免過於複雜）
        for start_idx, end_idx in face_connections:
            start_name = f"point_{start_idx}"
            end_name = f"point_{end_idx}"
            
            if start_name in mesh_points and end_name in mesh_points:
                start_x, start_y, _ = mesh_points[start_name]
                end_x, end_y, _ = mesh_points[end_name]
                
                # 繪製連線
                cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 128, 255), 1)
    
    def _draw_simple_face_mesh(self, image: np.ndarray, mesh_points: Dict[str, Tuple[int, int, float]]):
        """繪製簡化的臉部網格點"""
        for point_name, (x, y, z) in mesh_points.items():
            # 繪製網格點
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            
            # 顯示點名稱和座標（可選）
            if hasattr(self, '_show_coordinates') and self._show_coordinates:
                cv2.putText(image, f"{point_name}", (x + 5, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    def _draw_pose_keypoints(self, image: np.ndarray, keypoints: List[Dict[str, Any]]):
        """繪製骨架關鍵點和連線（跳過臉部關鍵點）"""
        for person_idx, person_keypoints in enumerate(keypoints):
            if 'keypoints' in person_keypoints:
                # 繪製關鍵點（跳過臉部關鍵點 0-4）
                for kp_idx, keypoint in enumerate(person_keypoints['keypoints']):
                    # 跳過臉部關鍵點：鼻子(0), 左眼(1), 右眼(2), 左耳(3), 右耳(4)
                    if kp_idx <= 4:
                        continue
                        
                    if keypoint.get('visible', 0) > 0.1:  # 只繪製可見的關鍵點
                        x, y = int(keypoint['x']), int(keypoint['y'])
                        conf = keypoint.get('conf', 0)
                        
                        # 根據信心度決定顏色
                        color = (0, int(255 * conf), int(255 * (1 - conf)))
                        
                        # 繪製關鍵點
                        cv2.circle(image, (x, y), 4, color, -1)
                        
                        # 顯示關鍵點索引和座標
                        if hasattr(self, '_show_coordinates') and self._show_coordinates:
                            cv2.putText(image, f"{kp_idx}", (x + 5, y - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # 繪製骨架連線
                self._draw_pose_connections(image, person_keypoints['keypoints'])
    
    def _draw_pose_connections(self, image: np.ndarray, keypoints: List[Dict[str, Any]]):
        """繪製骨架連線"""
        if len(keypoints) < 17:
            return
        
        # COCO-17 骨架連線定義（跳過臉部連線）
        connections = [
            # 軀幹（從肩膀開始，跳過臉部）
            (5, 6), (5, 11), (6, 12), (11, 12),  # 肩膀到髖部
            # 左臂
            (5, 7), (7, 9),  # 左肩到左肘到左腕
            # 右臂
            (6, 8), (8, 10),  # 右肩到右肘到右腕
            # 左腿
            (11, 13), (13, 15),  # 左髖到左膝到左踝
            # 右腿
            (12, 14), (14, 16),  # 右髖到右膝到右踝
        ]
        
        # 繪製連線
        for start_idx, end_idx in connections:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx].get('visible', 0) > 0.1 and 
                keypoints[end_idx].get('visible', 0) > 0.1):
                
                start_x, start_y = int(keypoints[start_idx]['x']), int(keypoints[start_idx]['y'])
                end_x, end_y = int(keypoints[end_idx]['x']), int(keypoints[end_idx]['y'])
                
                # 根據兩個關鍵點的信心度決定顏色
                start_conf = keypoints[start_idx].get('conf', 0)
                end_conf = keypoints[end_idx].get('conf', 0)
                avg_conf = (start_conf + end_conf) / 2
                color = (0, int(255 * avg_conf), int(255 * (1 - avg_conf)))
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, 2)
    
    def _draw_hand_landmarks(self, image: np.ndarray, hands_data: List[Dict[str, Any]]):
        """繪製手部關鍵點和連線"""
        for hand_idx, hand_data in enumerate(hands_data):
            landmarks = hand_data.get('landmarks', [])
            handedness = hand_data.get('handedness', 'Unknown')
            
            # 繪製關鍵點
            for idx, landmark in enumerate(landmarks):
                x, y = landmark['x'], landmark['y']
                conf = landmark.get('confidence', 0)
                
                # 根據手部類型選擇顏色
                if handedness == 'Left':
                    color = (255, 0, 0)  # 藍色
                elif handedness == 'Right':
                    color = (0, 0, 255)  # 紅色
                else:
                    color = (128, 128, 128)  # 灰色
                
                # 繪製關鍵點
                cv2.circle(image, (x, y), 3, color, -1)
                
                # 顯示關鍵點索引和座標
                if hasattr(self, '_show_coordinates') and self._show_coordinates:
                    cv2.putText(image, f"{idx}", (x + 5, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # 繪製手部連線
            if len(landmarks) >= 21:
                self._draw_hand_connections(image, landmarks, handedness)
    
    def _draw_hand_connections(self, image: np.ndarray, landmarks: List[Dict[str, Any]], handedness: str):
        """繪製手部關鍵點連線"""
        # 根據手部類型選擇顏色
        if handedness == 'Left':
            color = (255, 0, 0)  # 藍色
        elif handedness == 'Right':
            color = (0, 0, 255)  # 紅色
        else:
            color = (128, 128, 128)  # 灰色
        
        # 手腕到拇指
        self._draw_line(image, landmarks[0], landmarks[1], color)
        self._draw_line(image, landmarks[1], landmarks[2], color)
        self._draw_line(image, landmarks[2], landmarks[3], color)
        self._draw_line(image, landmarks[3], landmarks[4], color)
        
        # 手腕到其他手指
        for finger_start in [5, 9, 13, 17]:  # 各手指的起始點
            self._draw_line(image, landmarks[0], landmarks[finger_start], color)
            for i in range(3):
                self._draw_line(image, landmarks[finger_start + i], landmarks[finger_start + i + 1], color)
    
    def _draw_line(self, image: np.ndarray, point1: Dict[str, Any], point2: Dict[str, Any], color: Tuple[int, int, int]):
        """繪製兩點之間的連線"""
        x1, y1 = point1['x'], point1['y']
        x2, y2 = point2['x'], point2['y']
        cv2.line(image, (x1, y1), (x2, y2), color, 2)
    
    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """獲取會話統計資訊"""
        try:
            if not self.integrator:
                logging.warning("整合器未初始化或已關閉")
                return {
                    'session_id': session_id,
                    'error': '整合器未初始化或已關閉',
                    'frame_count': 0,
                    'duration_seconds': None,
                    'modality_coverage': {},
                    'quality_score': 'unknown',
                    'source_path': None
                }
            
            return self.integrator.get_statistics(session_id)
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
    
    def export_session_data(self, session_id: str, output_dir: Path) -> List[Path]:
        """匯出會話資料"""
        try:
            if not self.integrator:
                logging.warning("整合器未初始化或已關閉")
                return []
            
            return self.integrator.export_session_to_csv(session_id, output_dir)
        except Exception as e:
            logging.error("匯出會話資料時發生錯誤: %s", e)
            return []
    
    def close(self):
        """關閉所有資源"""
        if self.is_processing:
            self.stop_session()
        
        if self.face_recorder:
            self.face_recorder.close()
            self.face_recorder = None
        
        if self.hand_recorder:
            self.hand_recorder.close()
            self.hand_recorder = None
        
        if self.pose_recorder:
            self.pose_recorder = None
        
        # 清空整合器和分析器
        self.integrator = None
        self.analyzer = None
        
        logging.info("多模態處理器已關閉")


def main():
    """主函數：用於直接執行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="多模態檢測處理器")
    parser.add_argument("source", help="影片來源（檔案路徑或攝影機索引）")
    parser.add_argument("--session-id", default="multimodal_session", help="會話 ID")
    parser.add_argument("--output-dir", type=Path, help="輸出目錄")
    parser.add_argument("--show-video", action="store_true", help="顯示影片")
    parser.add_argument("--save-video", action="store_true", help="儲存影片")
    parser.add_argument("--save-csv", action="store_true", default=True, help="儲存 CSV")
    parser.add_argument("--disable-face", action="store_true", help="停用臉部檢測")
    parser.add_argument("--disable-pose", action="store_true", help="停用姿態檢測")
    parser.add_argument("--disable-hand", action="store_true", help="停用手部檢測")
    parser.add_argument("--pixel-to-cm", type=float, help="像素到公分的轉換比例")
    
    args = parser.parse_args()
    
    # 設定日誌
    logging.basicConfig(level=logging.INFO)
    
    # 創建多模態處理器
    processor = MultimodalProcessor(
        enable_face=not args.disable_face,
        enable_pose=not args.disable_pose,
        enable_hand=not args.disable_hand,
        pixel_to_cm=args.pixel_to_cm
    )
    
    try:
        # 處理影片
        result = processor.process_video(
            source=args.source,
            session_id=args.session_id,
            output_dir=args.output_dir,
            show_video=args.show_video,
            save_video=args.save_video,
            save_csv=args.save_csv
        )
        
        print(f"處理完成: {result}")
        
    finally:
        processor.close()


if __name__ == "__main__":
    main()
