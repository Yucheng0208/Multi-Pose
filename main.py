#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe 整合偵測器（進階版，支援每秒切片/分模態/流水夾/row-wise CSV）
- 抑制雜訊：warnings、OpenCV、glog/absl
- 命令列旗標：--save-json/--save-npy/--save-csv/--save-row-csv/--out-video/--skeleton-only/--silent
- 每秒切片 (--chunk-per-sec) 與分模態資料夾 (--split-modalities)
- 切片資料夾採固定寬度流水號 (--folder-width, --start-index, --chunk-prefix)
- 單人流程：num_faces=1、num_poses=1、num_hands=2
- 全域 keypoint 重新編號：Pose(33)→0-32, Left(21)→33-53, Right(21)→54-74, Face(478)→75-552，共 553 點
- Row-wise CSV 欄位：person_id, model, keypoints, coor_x, coor_y, confidence, frame_idx, timestamp_ms, fps
"""

import os
# 先壓低底層日誌
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

try:
    import absl.logging
    absl.logging.set_verbosity("error")
except Exception:
    pass

import mediapipe as mp
import numpy as np
from typing import Optional, List, Dict, Any
import time
import json
import argparse
from pathlib import Path


# -------------------- 自動偵測 models --------------------
def auto_detect_models(models_dir: str = "models"):
    model_paths = {'face': None, 'pose': None, 'hand': None}
    if not os.path.exists(models_dir):
        print(f"⚠️  模型資料夾不存在: {models_dir}")
        return model_paths

    face_names = ['face_landmarker.task', 'face.task', 'face_landmark.task']
    pose_names = ['pose_landmarker_full.task', 'pose.task', 'pose_landmark_full.task']
    hand_names = ['hand_landmarker.task', 'hand.task', 'hand_landmark.task']

    for name in face_names:
        p = os.path.join(models_dir, name)
        if os.path.exists(p):
            model_paths['face'] = p
            print(f"✓ 找到臉部模型: {p}")
            break
    for name in pose_names:
        p = os.path.join(models_dir, name)
        if os.path.exists(p):
            model_paths['pose'] = p
            print(f"✓ 找到姿勢模型: {p}")
            break
    for name in hand_names:
        p = os.path.join(models_dir, name)
        if os.path.exists(p):
            model_paths['hand'] = p
            print(f"✓ 找到手部模型: {p}")
            break

    for k, v in model_paths.items():
        if v is None:
            print(f"❌ 未找到 {k} 模型，將使用內建模型")
    return model_paths


# -------------------- 全域索引定義 --------------------
GLOBAL_INDEX_OFFSET = {
    "pose": 0,       # 0..32 (33)
    "left_hand": 33, # 33..53 (21)
    "right_hand": 54,# 54..74 (21)
    "face": 75       # 75..552 (478)
}
TOTAL_KEYPOINTS = 553  # 33 + 21 + 21 + 478


def split_modal_arrays_from_global(global_kps: np.ndarray) -> dict:
    """
    將 [553,4] 的全域 keypoints 拆為三個模態：
      - pose: [33,4]  (0..32)
      - hand: [42,4]  (左21 + 右21 → 33..53 + 54..74)
      - face: [478,4] (75..552)
    回傳: {"pose": np.ndarray, "hand": np.ndarray, "face": np.ndarray}
    """
    pose = global_kps[0:33, :].copy()
    left = global_kps[33:54, :].copy()
    right = global_kps[54:75, :].copy()
    hand = np.vstack([left, right])  # [42,4]
    face = global_kps[75:553, :].copy()
    return {"pose": pose, "hand": hand, "face": face}


# -------------------- 寫入單一切片（每秒） --------------------
def _write_chunk(out_dir: Path,
                 seq_folder_name: str,
                 arrays_per_frame: List[Dict[str, Any]],
                 split_modalities: bool,
                 save_json: bool,
                 save_csv: bool,
                 save_npy: bool,
                 chunk_prefix: str = "s",
                 save_row_csv: bool = False,
                 person_id: int = 0):
    """
    將某一『秒』的所有影格輸出到 out_dir/seq_folder_name/
    arrays_per_frame[t] 結構：
      {
        "frame_idx": int,
        "timestamp_ms": int,
        "run_fps": float,
        "global": np.ndarray (553,4),
        "modal": {"pose":(33,4), "hand":(42,4), "face":(478,4)}
      }
    """
    if not arrays_per_frame:
        return

    sec_dir = out_dir / seq_folder_name
    sec_dir.mkdir(parents=True, exist_ok=True)

    # 疊 T×N×4
    g_stack = np.stack([it["global"] for it in arrays_per_frame], axis=0)   # [T,553,4]
    m_pose  = np.stack([it["modal"]["pose"] for it in arrays_per_frame], 0) # [T,33,4]
    m_hand  = np.stack([it["modal"]["hand"] for it in arrays_per_frame], 0) # [T,42,4]
    m_face  = np.stack([it["modal"]["face"] for it in arrays_per_frame], 0) # [T,478,4]

    # --- 寬表（原本的 NPY/CSV/JSON；split 或不拆） ---
    def _save_wide(base_dir: Path, base_name: str, stack: np.ndarray, N: int):
        npy_file  = base_dir / f"{base_name}.npy"
        csv_file  = base_dir / f"{base_name}.csv"
        json_file = base_dir / f"{base_name}.json"

        if save_npy:
            np.save(npy_file, stack)

        if save_csv:
            with open(csv_file, "w", encoding="utf-8") as f:
                header = ["frame_idx", "timestamp_ms", "fps"]
                for i in range(N):
                    header += [f"k{i}_x", f"k{i}_y", f"k{i}_z", f"k{i}_vis"]
                f.write(",".join(header) + "\n")
                for t in range(stack.shape[0]):
                    meta = arrays_per_frame[t]
                    row = [str(meta["frame_idx"]), str(meta["timestamp_ms"]), f"{meta['run_fps']:.6f}"]
                    flat = stack[t].reshape(-1).tolist()
                    row.extend([f"{v:.7f}" for v in flat])
                    f.write(",".join(row) + "\n")

        if save_json:
            frames = []
            for t in range(stack.shape[0]):
                meta = arrays_per_frame[t]
                frames.append({
                    "frame_idx": meta["frame_idx"],
                    "timestamp_ms": meta["timestamp_ms"],
                    "fps": meta["run_fps"],
                    "keypoints": stack[t].tolist()
                })
            payload = {"schema": f"per-point=[x,y,z,vis], count={N}", "frames": frames}
            with open(json_file, "w", encoding="utf-8") as jf:
                json.dump(payload, jf, ensure_ascii=False)

    if split_modalities:
        d_face = sec_dir / "face"
        d_pose = sec_dir / "pose"
        d_hand = sec_dir / "hand"
        d_face.mkdir(exist_ok=True)
        d_pose.mkdir(exist_ok=True)
        d_hand.mkdir(exist_ok=True)

        _save_wide(d_pose, f"{chunk_prefix}_pose", m_pose, 33)
        _save_wide(d_hand, f"{chunk_prefix}_hand", m_hand, 42)
        _save_wide(d_face, f"{chunk_prefix}_face", m_face, 478)
    else:
        _save_wide(sec_dir, f"{chunk_prefix}_global", g_stack, 553)

    # --- Row-wise 長表 CSV（符合你指定欄位） ---
    if save_row_csv:
        rows_path = sec_dir / f"{chunk_prefix}_rows.csv"
        with open(rows_path, "w", encoding="utf-8") as f:
            f.write("person_id,model,keypoints,coor_x,coor_y,confidence,frame_idx,timestamp_ms,fps\n")
            T = len(arrays_per_frame)

            # Pose（model='pose'，0..32；confidence=visibility）
            for t in range(T):
                meta = arrays_per_frame[t]
                for k in range(33):
                    x, y, z, vis = m_pose[t, k, :]
                    f.write(f"{person_id},pose,{k},{x:.7f},{y:.7f},{vis:.7f},{meta['frame_idx']},{meta['timestamp_ms']},{meta['run_fps']:.6f}\n")

            # Hand：0..20 → left_hand、21..41 → right_hand；confidence 取 vis>0 否則 1.0
            for t in range(T):
                meta = arrays_per_frame[t]
                for k in range(42):
                    model = "left_hand" if k < 21 else "right_hand"
                    kk = k if k < 21 else (k - 21)
                    x, y, z, vis = m_hand[t, k, :]
                    conf = vis if vis > 0 else 1.0
                    f.write(f"{person_id},{model},{kk},{x:.7f},{y:.7f},{conf:.7f},{meta['frame_idx']},{meta['timestamp_ms']},{meta['run_fps']:.6f}\n")

            # Face（0..477；confidence 取 vis>0 否則 1.0）
            for t in range(T):
                meta = arrays_per_frame[t]
                for k in range(478):
                    x, y, z, vis = m_face[t, k, :]
                    conf = vis if vis > 0 else 1.0
                    f.write(f"{person_id},face,{k},{x:.7f},{y:.7f},{conf:.7f},{meta['frame_idx']},{meta['timestamp_ms']},{meta['run_fps']:.6f}\n")


# -------------------- 主類別 --------------------
class MediaPipeIntegratedDetector:
    """整合 MediaPipe 臉部、姿勢、手部偵測的類別"""

    def __init__(self,
                 face_model_path: Optional[str] = None,
                 pose_model_path: Optional[str] = None,
                 hand_model_path: Optional[str] = None,
                 running_mode: str = "IMAGE",
                 silent: bool = True):
        self.running_mode = running_mode
        self.silent = silent

        self._validate_model_paths(face_model_path, pose_model_path, hand_model_path)

        self.BaseOptions = mp.tasks.BaseOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        if running_mode == "IMAGE":
            self.mode = self.VisionRunningMode.IMAGE
        elif running_mode == "VIDEO":
            self.mode = self.VisionRunningMode.VIDEO
        elif running_mode == "LIVE_STREAM":
            self.mode = self.VisionRunningMode.LIVE_STREAM
        else:
            raise ValueError("running_mode 必須是 'IMAGE', 'VIDEO', 或 'LIVE_STREAM'")

        self._init_face_detector(face_model_path)
        self._init_pose_detector(pose_model_path)
        self._init_hand_detector(hand_model_path)

        self.latest_results = {'face': None, 'pose': None, 'hand': None}

    def _log(self, msg: str):
        if not self.silent:
            print(msg)

    def _validate_model_paths(self, face_path: Optional[str],
                              pose_path: Optional[str],
                              hand_path: Optional[str]):
        paths = {'face_model': face_path, 'pose_model': pose_path, 'hand_model': hand_path}
        for name, path in paths.items():
            if path is not None:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{name} 檔案不存在: {path}")
                if not path.endswith('.task'):
                    self._log(f"⚠️ 警告: {name} 檔案可能不是 .task：{path}")

    def _init_face_detector(self, model_path: Optional[str]):
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        base_options = self.BaseOptions(model_asset_path=model_path) if model_path else self.BaseOptions()
        kwargs = dict(
            base_options=base_options,
            running_mode=self.mode,
            num_faces=1,  # 單人
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        if self.mode == self.VisionRunningMode.LIVE_STREAM:
            kwargs["result_callback"] = self._face_result_callback
        self.face_detector = FaceLandmarker.create_from_options(FaceLandmarkerOptions(**kwargs))

    def _init_pose_detector(self, model_path: Optional[str]):
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        base_options = self.BaseOptions(model_asset_path=model_path) if model_path else self.BaseOptions()
        kwargs = dict(
            base_options=base_options,
            running_mode=self.mode,
            num_poses=1,  # 單人
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        if self.mode == self.VisionRunningMode.LIVE_STREAM:
            kwargs["result_callback"] = self._pose_result_callback
        self.pose_detector = PoseLandmarker.create_from_options(PoseLandmarkerOptions(**kwargs))

    def _init_hand_detector(self, model_path: Optional[str]):
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        base_options = self.BaseOptions(model_asset_path=model_path) if model_path else self.BaseOptions()
        kwargs = dict(
            base_options=base_options,
            running_mode=self.mode,
            num_hands=2,  # 左右手
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        if self.mode == self.VisionRunningMode.LIVE_STREAM:
            kwargs["result_callback"] = self._hand_result_callback
        self.hand_detector = HandLandmarker.create_from_options(HandLandmarkerOptions(**kwargs))

    def _face_result_callback(self, result, output_image: mp.Image, timestamp_ms: int):
        self.latest_results['face'] = result

    def _pose_result_callback(self, result, output_image: mp.Image, timestamp_ms: int):
        self.latest_results['pose'] = result

    def _hand_result_callback(self, result, output_image: mp.Image, timestamp_ms: int):
        self.latest_results['hand'] = result

    # ------------ 偵測介面 ------------
    def detect_image(self, image_path: str) -> dict:
        if self.mode != self.VisionRunningMode.IMAGE:
            raise ValueError("此方法僅適用於 IMAGE 模式")
        mp_image = mp.Image.create_from_file(image_path)
        face_result = self.face_detector.detect(mp_image)
        pose_result = self.pose_detector.detect(mp_image)
        hand_result = self.hand_detector.detect(mp_image)
        return {'face': face_result, 'pose': pose_result, 'hand': hand_result, 'image': mp_image}

    def detect_frame(self, frame_rgb: np.ndarray, timestamp_ms: int) -> dict:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self.mode == self.VisionRunningMode.VIDEO:
            return {
                'face': self.face_detector.detect_for_video(mp_image, timestamp_ms),
                'pose': self.pose_detector.detect_for_video(mp_image, timestamp_ms),
                'hand': self.hand_detector.detect_for_video(mp_image, timestamp_ms),
                'image': mp_image
            }
        elif self.mode == self.VisionRunningMode.LIVE_STREAM:
            self.face_detector.detect_async(mp_image, timestamp_ms)
            self.pose_detector.detect_async(mp_image, timestamp_ms)
            self.hand_detector.detect_async(mp_image, timestamp_ms)
            return {
                'face': self.latest_results['face'],
                'pose': self.latest_results['pose'],
                'hand': self.latest_results['hand'],
                'image': mp_image
            }

    # ------------ 視覺化 ------------
    def visualize_results(self, image_bgr: np.ndarray, results: dict, skeleton_only: bool = False) -> np.ndarray:
        annotated = np.zeros_like(image_bgr) if skeleton_only else image_bgr.copy()
        h, w = image_bgr.shape[:2]

        # Face
        if results['face'] and results['face'].face_landmarks:
            for face in results['face'].face_landmarks:
                for lm in face:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (x, y), 1, (0, 255, 0), -1)

        # Pose
        if results['pose'] and results['pose'].pose_landmarks:
            for pose in results['pose'].pose_landmarks:
                for lm in pose:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (x, y), 3, (255, 0, 0), -1)
                self._draw_pose_connections(annotated, pose, w, h)

        # Hands
        if results['hand'] and results['hand'].hand_landmarks:
            for hand in results['hand'].hand_landmarks:
                for lm in hand:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (x, y), 2, (0, 0, 255), -1)
                self._draw_hand_connections(annotated, hand, w, h)

        return annotated

    def _draw_pose_connections(self, image: np.ndarray, landmarks, width: int, height: int):
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28),
        ]
        for a, b in connections:
            if a < len(landmarks) and b < len(landmarks):
                p1 = (int(landmarks[a].x * width), int(landmarks[a].y * height))
                p2 = (int(landmarks[b].x * width), int(landmarks[b].y * height))
                cv2.line(image, p1, p2, (255, 0, 0), 2)

    def _draw_hand_connections(self, image: np.ndarray, landmarks, width: int, height: int):
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9,10), (10,11), (11,12),
            (0,13), (13,14), (14,15), (15,16),
            (0,17), (17,18), (18,19), (19,20),
        ]
        for a, b in connections:
            if a < len(landmarks) and b < len(landmarks):
                p1 = (int(landmarks[a].x * width), int(landmarks[a].y * height))
                p2 = (int(landmarks[b].x * width), int(landmarks[b].y * height))
                cv2.line(image, p1, p2, (0, 0, 255), 2)

    # ------------ 封包成全域 keypoints (553×4) ------------
    @staticmethod
    def _bbox_center_x(landmarks) -> float:
        xs = [lm.x for lm in landmarks]
        return float(sum(xs) / len(xs)) if xs else 0.5

    def pack_global_keypoints(self, results: dict) -> np.ndarray:
        out = np.zeros((TOTAL_KEYPOINTS, 4), dtype=np.float32)

        # Pose → 0..32
        if results['pose'] and results['pose'].pose_landmarks:
            pose = results['pose'].pose_landmarks[0]
            for i, lm in enumerate(pose[:33]):
                vis = getattr(lm, "visibility", 1.0) if hasattr(lm, "visibility") else 1.0
                out[i, :] = [lm.x, lm.y, lm.z, vis]

        # Hands：盡量用 handedness，否則以 x 中心判斷
        lh, rh = None, None
        if results['hand']:
            hands = results['hand']
            hand_lms = hands.hand_landmarks or []
            handedness = getattr(hands, "handedness", [[] for _ in hand_lms])
            for idx, lms in enumerate(hand_lms):
                label = None
                try:
                    if handedness and handedness[idx]:
                        label = handedness[idx][0].category_name  # "Left"/"Right"
                except Exception:
                    label = None
                if label == "Left":
                    lh = lms
                elif label == "Right":
                    rh = lms
            if lh is None and rh is None and len(hand_lms) == 2:
                c0 = self._bbox_center_x(hand_lms[0])
                c1 = self._bbox_center_x(hand_lms[1])
                lh, rh = (hand_lms[0], hand_lms[1]) if c0 <= c1 else (hand_lms[1], hand_lms[0])
            elif lh is None and len(hand_lms) == 1:
                rh = hand_lms[0]

        if lh is not None:
            base = GLOBAL_INDEX_OFFSET["left_hand"]
            for i, lm in enumerate(lh[:21]):
                out[base + i, :] = [lm.x, lm.y, lm.z, 1.0]
        if rh is not None:
            base = GLOBAL_INDEX_OFFSET["right_hand"]
            for i, lm in enumerate(rh[:21]):
                out[base + i, :] = [lm.x, lm.y, lm.z, 1.0]

        # Face → 75..552
        if results['face'] and results['face'].face_landmarks:
            face = results['face'].face_landmarks[0]
            base = GLOBAL_INDEX_OFFSET["face"]
            for i, lm in enumerate(face[:478]):
                out[base + i, :] = [lm.x, lm.y, lm.z, 1.0]
        return out

    # ------------ 影片處理（含每秒切片輸出） ------------
    def process_video(self,
                      video_path: str,
                      output_video_path: Optional[str] = None,
                      save_json: bool = False,
                      save_csv: bool = False,
                      save_npy: bool = False,
                      save_row_csv: bool = False,
                      person_id: int = 0,
                      skeleton_only: bool = False,
                      out_dir: str = "outputs",
                      chunk_per_sec: bool = True,
                      split_modalities: bool = True,
                      folder_width: int = 19,
                      start_index: int = 1,
                      chunk_prefix: str = "s"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片：{video_path}")

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_video_path, fourcc, fps_src, (width, height))

        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        prev_t = time.time()
        run_fps = 0.0

        # 每秒切片累積器
        current_sec = None
        bucket: List[Dict[str, Any]] = []
        seq = start_index  # 流水號
        zero = lambda x: str(x).zfill(folder_width)

        def flush_bucket():
            nonlocal bucket, seq
            if not bucket:
                return
            seq_folder = zero(seq)
            _write_chunk(out_root, seq_folder, bucket,
                         split_modalities=split_modalities,
                         save_json=save_json, save_csv=save_csv, save_npy=save_npy,
                         chunk_prefix=chunk_prefix,
                         save_row_csv=save_row_csv,
                         person_id=person_id)
            bucket = []
            seq += 1

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx * 1000.0 / fps_src)
            sec_mark = timestamp_ms // 1000

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.detect_frame(frame_rgb, timestamp_ms)

            # 全域 keypoints + 分模態
            kps = self.pack_global_keypoints(results)
            modal = split_modal_arrays_from_global(kps)

            # 視覺化
            annotated = self.visualize_results(frame_bgr, results, skeleton_only=skeleton_only)

            # 即時計算執行 FPS（非來源 FPS）
            now = time.time()
            delta = now - prev_t
            run_fps = (1.0 / delta) if delta > 0 else run_fps
            prev_t = now

            # 畫 FPS
            text = f"FPS: {run_fps:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(annotated, text, (annotated.shape[1] - tw - 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if out_writer is not None:
                out_writer.write(annotated)

            # 每秒切片：換秒就 flush
            if chunk_per_sec:
                if current_sec is None:
                    current_sec = sec_mark
                if sec_mark != current_sec:
                    flush_bucket()
                    current_sec = sec_mark
                bucket.append({
                    "frame_idx": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "run_fps": run_fps,
                    "global": kps,
                    "modal": modal
                })
            else:
                # 不切片→整段影片結束時才寫一次（以一桶處理）
                if current_sec is None:
                    current_sec = 0
                bucket.append({
                    "frame_idx": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "run_fps": run_fps,
                    "global": kps,
                    "modal": modal
                })

            frame_idx += 1

        # 收尾
        cap.release()
        if out_writer is not None:
            out_writer.release()
        cv2.destroyAllWindows()

        # flush 最後一桶
        if bucket:
            flush_bucket()

        self._log("影片處理完成")
        return {"frames": frame_idx, "fps_src": fps_src, "out_dir": str(out_root)}

    # ------------ 即時攝影機 ------------
    def process_webcam(self, skeleton_only: bool = False):
        cap = cv2.VideoCapture(0)
        prev_time = time.time()
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break
            timestamp_ms = int(time.time() * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.detect_frame(frame_rgb, timestamp_ms)
            annotated = self.visualize_results(frame_bgr, results, skeleton_only=skeleton_only)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            text = f"FPS: {fps:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(annotated, text, (annotated.shape[1] - tw - 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(annotated, 'Press Q to quit', (10, 30 + th + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('MediaPipe 整合偵測 - 即時', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def close(self):
        self.face_detector.close()
        self.pose_detector.close()
        self.hand_detector.close()


# -------------------- CLI --------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="MediaPipe 整合偵測器（進階版，支援每秒切片/分模態/流水夾/row-wise CSV）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--mode", type=str, default="VIDEO", choices=["IMAGE", "VIDEO", "LIVE_STREAM"],
                   help="執行模式")
    p.add_argument("--image", type=str, default=None, help="圖片路徑（IMAGE 模式）")
    p.add_argument("--video", type=str, default=None, help="影片路徑（VIDEO 模式）")
    p.add_argument("--out-video", type=str, default=None, help="輸出影片路徑（mp4）")

    p.add_argument("--models-dir", type=str, default="models", help="模型資料夾")
    p.add_argument("--out-dir", type=str, default="outputs", help="資料輸出資料夾")
    p.add_argument("--skeleton-only", type=lambda s: s.lower() in ("1","true","yes","y"), default=False,
                   help="輸出純骨架可視化（背景黑）")
    p.add_argument("--silent", type=lambda s: s.lower() in ("1","true","yes","y"), default=True,
                   help="靜音模式（抑制多數訊息）")

    # 存檔選項
    p.add_argument("--save-json", type=lambda s: s.lower() in ("1","true","yes","y"), default=False,
                   help="輸出 JSON")
    p.add_argument("--save-csv", type=lambda s: s.lower() in ("1","true","yes","y"), default=False,
                   help="輸出 CSV（寬表）")
    p.add_argument("--save-npy", type=lambda s: s.lower() in ("1","true","yes","y"), default=False,
                   help="輸出 NPY")
    p.add_argument("--save-row-csv", type=lambda s: s.lower() in ("1","true","yes","y"), default=False,
                   help="輸出 row-wise CSV（person_id, model, keypoints, coor_x, coor_y, confidence, frame_idx, timestamp_ms, fps）")
    p.add_argument("--person-id", type=int, default=0, help="固定寫入的 person_id（目前僅單人）")

    # 每秒切片/分模態/流水夾
    p.add_argument("--chunk-per-sec", type=lambda s: s.lower() in ("1","true","yes","y"), default=True,
                   help="是否按『秒』切片輸出")
    p.add_argument("--split-modalities", type=lambda s: s.lower() in ("1","true","yes","y"), default=True,
                   help="是否將每秒資料拆到 face/pose/hand 子資料夾")
    p.add_argument("--folder-width", type=int, default=19,
                   help="流水號資料夾的零填寬度（例如 19 → 0000000000000000001）")
    p.add_argument("--start-index", type=int, default=1, help="流水號起始值")
    p.add_argument("--chunk-prefix", type=str, default="s",
                   help="每秒切片檔名前綴（例如 s_pose.npy）")
    return p


def cli_main(args=None):
    parser = build_parser()
    cfg = parser.parse_args(args=args)

    detected = auto_detect_models(cfg.models_dir)
    face_model = detected['face']
    pose_model = detected['pose']
    hand_model = detected['hand']

    det = MediaPipeIntegratedDetector(
        face_model_path=face_model,
        pose_model_path=pose_model,
        hand_model_path=hand_model,
        running_mode=cfg.mode,
        silent=cfg.silent
    )

    try:
        if cfg.mode == "IMAGE":
            if not cfg.image:
                raise ValueError("IMAGE 模式需提供 --image")
            results = det.detect_image(cfg.image)
            img = cv2.imread(cfg.image)
            ann = det.visualize_results(img, results, skeleton_only=cfg.skeleton_only)
            cv2.imshow('結果', ann)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif cfg.mode == "VIDEO":
            if not cfg.video:
                raise ValueError("VIDEO 模式需提供 --video")
            det.process_video(
                cfg.video,
                output_video_path=cfg.out_video if cfg.out_video else None,
                save_json=cfg.save_json,
                save_csv=cfg.save_csv,
                save_npy=cfg.save_npy,
                save_row_csv=cfg.save_row_csv,
                person_id=cfg.person_id,
                skeleton_only=cfg.skeleton_only,
                out_dir=cfg.out_dir,
                chunk_per_sec=cfg.chunk_per_sec,
                split_modalities=cfg.split_modalities,
                folder_width=cfg.folder_width,
                start_index=cfg.start_index,
                chunk_prefix=cfg.chunk_prefix
            )

        elif cfg.mode == "LIVE_STREAM":
            det.process_webcam(skeleton_only=cfg.skeleton_only)

    finally:
        det.close()


# -------------------- 互動式 main/quick_test 仍保留 --------------------
def main():
    print("MediaPipe 整合偵測器")
    print("=" * 40)
    print("正在搜尋本地模型...")
    detected_models = auto_detect_models("models")
    face_model = detected_models['face']
    pose_model = detected_models['pose']
    hand_model = detected_models['hand']
    print(f"\n模型載入狀況:")
    print(f"臉部模型: {'✓ 本地模型' if face_model else '❌ 使用內建模型'}")
    print(f"姿勢模型: {'✓ 本地模型' if pose_model else '❌ 使用內建模型'}")
    print(f"手部模型: {'✓ 本地模型' if hand_model else '❌ 使用內建模型'}")

    print("\n選擇執行模式：")
    print("1. 圖片偵測")
    print("2. 影片偵測")
    print("3. 即時攝影機偵測")
    choice = input("請選擇模式 (1/2/3): ")

    if choice == '1':
        detector = MediaPipeIntegratedDetector(face_model, pose_model, hand_model, running_mode="IMAGE")
        image_path = input("請輸入圖片路徑: ")
        try:
            results = detector.detect_image(image_path)
            image = cv2.imread(image_path)
            annotated_image = detector.visualize_results(image, results)
            cv2.imshow('結果', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("偵測完成！")
        except Exception as e:
            print(f"錯誤: {e}")
        detector.close()

    elif choice == '2':
        detector = MediaPipeIntegratedDetector(face_model, pose_model, hand_model, running_mode="VIDEO")
        video_path = input("請輸入影片路徑: ")
        output_path = input("請輸入輸出影片路徑 (可選，Enter 略過): ").strip() or None
        try:
            detector.process_video(
                video_path,
                output_video_path=output_path,
                save_json=True, save_csv=True, save_npy=True, save_row_csv=True,
                chunk_per_sec=True, split_modalities=True
            )
            print("影片處理完成！")
        except Exception as e:
            print(f"錯誤: {e}")
        detector.close()

    elif choice == '3':
        detector = MediaPipeIntegratedDetector(face_model, pose_model, hand_model, running_mode="LIVE_STREAM")
        print("啟動網路攝影機，按 Q 退出...")
        try:
            detector.process_webcam()
        except Exception as e:
            print(f"錯誤: {e}")
        detector.close()
    else:
        print("無效選擇")


def quick_test():
    print("快速測試模式 - 自動載入本地模型")
    detected_models = auto_detect_models("models")
    detector = MediaPipeIntegratedDetector(
        face_model_path=detected_models['face'],
        pose_model_path=detected_models['pose'],
        hand_model_path=detected_models['hand'],
        running_mode="LIVE_STREAM"
    )
    print("啟動網路攝影機，按 Q 退出...")
    try:
        detector.process_webcam()
    except Exception as e:
        print(f"錯誤: {e}")
    detector.close()
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")


if __name__ == "__main__":
    # 想用命令列：python this.py --mode VIDEO --video in.mp4 ...
    # 或用互動式主選單
    print("請選擇:")
    print("1. 完整模式 (互動式)")
    print("2. 快速測試 (直接開攝影機)")
    print("3. 命令列模式 (直接使用 --mode/--video 等參數)")
    mode = input("輸入選擇 (1/2/3): ").strip()
    if mode == "1":
        main()
    elif mode == "2":
        quick_test()
    else:
        # 命令列模式：把控制權交給 argparse
        cli_main()
