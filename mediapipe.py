#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三合一 MediaPipe 偵測器（Face + Pose + Hand）
- 模式：圖片 / 影片 / 攝影機
- 單人：num_faces=1、num_poses=1、num_hands=2
- 新增開關：--save-info、--save-keypoints
  * save-info      : 是否疊上資訊（FPS）
  * save-keypoints : 是否繪製 keypoints/骨架
- 即時影像也可同步錄影（--out-video 對 VIDEO 與 CAM 都有效）
- 已去除「人數」計算與列印
"""

import os
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings
warnings.filterwarnings("ignore")

import sys
import contextlib
import cv2
import time
import argparse
from pathlib import Path
import numpy as np
import mediapipe as mp

@contextlib.contextmanager
def suppress_stderr():
    devnull = open(os.devnull, 'w')
    old = sys.stderr
    try:
        sys.stderr = devnull
        yield
    finally:
        sys.stderr = old
        devnull.close()

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass
try:
    import absl.logging
    absl.logging.set_verbosity("error")
except Exception:
    pass

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


def build_parser():
    p = argparse.ArgumentParser(
        description="三合一 MediaPipe 偵測器（Face + Pose + Hand）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--mode", choices=["IMAGE", "VIDEO", "CAM"], default="VIDEO",
                   help="IMAGE 單張圖片；VIDEO 影片檔；CAM 攝影機")
    p.add_argument("--input", type=str, default=None,
                   help="輸入路徑（IMAGE/VIDEO 模式必填）")
    p.add_argument("--out-video", type=str, default=None,
                   help="輸出視覺化影片路徑（VIDEO/CAM 模式皆可）")
    p.add_argument("--models-dir", type=str, default="models",
                   help=".task 模型資料夾（可省略）")
    p.add_argument("--cam-id", type=int, default=0,
                   help="攝影機裝置 ID（CAM 模式）")
    p.add_argument("--silent", type=lambda s: s.lower() in ("1","true","yes","y"),
                   default=True, help="靜音模式（減少印出）")

    # 新增：控制是否顯示/錄進 overlay
    p.add_argument("--save-info", type=lambda s: s.lower() in ("1","true","yes","y"),
                   default=True, help="是否疊上資訊（FPS）")
    p.add_argument("--save-keypoints", type=lambda s: s.lower() in ("1","true","yes","y"),
                   default=True, help="是否繪製 keypoints 與骨架")
    return p


def auto_detect_models(models_dir: str):
    paths = {'face': None, 'pose': None, 'hand': None}
    d = Path(models_dir)
    if not d.exists():
        return paths
    cands = {
        'face': ['face_landmarker.task', 'face.task', 'face_landmark.task'],
        'pose': ['pose_landmarker_full.task', 'pose.task', 'pose_landmark_full.task'],
        'hand': ['hand_landmarker.task', 'hand.task', 'hand_landmark.task'],
    }
    for k, names in cands.items():
        for n in names:
            p = d / n
            if p.exists():
                paths[k] = str(p)
                break
    return paths


class ThreeInOneDetector:
    def __init__(self, mode: str, models: dict, silent: bool = True):
        self.silent = silent
        if mode == "IMAGE":
            self.running_mode = VisionRunningMode.IMAGE
        elif mode == "VIDEO":
            self.running_mode = VisionRunningMode.VIDEO
        elif mode == "CAM":
            self.running_mode = VisionRunningMode.LIVE_STREAM
        else:
            raise ValueError("mode 必須是 IMAGE/VIDEO/CAM")

        face_base = BaseOptions(model_asset_path=models['face']) if models['face'] else BaseOptions()
        face_kwargs = dict(
            base_options=face_base,
            running_mode=self.running_mode,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        if self.running_mode == VisionRunningMode.LIVE_STREAM:
            face_kwargs["result_callback"] = self._cb_face
        with suppress_stderr():
            self.face = FaceLandmarker.create_from_options(FaceLandmarkerOptions(**face_kwargs))

        pose_base = BaseOptions(model_asset_path=models['pose']) if models['pose'] else BaseOptions()
        pose_kwargs = dict(
            base_options=pose_base,
            running_mode=self.running_mode,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        if self.running_mode == VisionRunningMode.LIVE_STREAM:
            pose_kwargs["result_callback"] = self._cb_pose
        with suppress_stderr():
            self.pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(**pose_kwargs))

        hand_base = BaseOptions(model_asset_path=models['hand']) if models['hand'] else BaseOptions()
        hand_kwargs = dict(
            base_options=hand_base,
            running_mode=self.running_mode,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        if self.running_mode == VisionRunningMode.LIVE_STREAM:
            hand_kwargs["result_callback"] = self._cb_hand
        with suppress_stderr():
            self.hand = HandLandmarker.create_from_options(HandLandmarkerOptions(**hand_kwargs))

        self._latest = {'face': None, 'pose': None, 'hand': None}

    # callbacks
    def _cb_face(self, result, out_img: mp.Image, ts_ms: int): self._latest['face'] = result
    def _cb_pose(self, result, out_img: mp.Image, ts_ms: int): self._latest['pose'] = result
    def _cb_hand(self, result, out_img: mp.Image, ts_ms: int): self._latest['hand'] = result

    def detect_image(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with suppress_stderr():
            f = self.face.detect(mp_img)
            p = self.pose.detect(mp_img)
            h = self.hand.detect(mp_img)
        return {'face': f, 'pose': p, 'hand': h}

    def detect_video_frame(self, bgr, ts_ms: int):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with suppress_stderr():
            f = self.face.detect_for_video(mp_img, ts_ms)
            p = self.pose.detect_for_video(mp_img, ts_ms)
            h = self.hand.detect_for_video(mp_img, ts_ms)
        return {'face': f, 'pose': p, 'hand': h}

    def detect_stream_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)
        with suppress_stderr():
            self.face.detect_async(mp_img, ts)
            self.pose.detect_async(mp_img, ts)
            self.hand.detect_async(mp_img, ts)
        return self._latest

    @staticmethod
    def draw(results, bgr, draw_keypoints: bool, show_info: bool, fps_value: float | None):
        """根據旗標決定是否繪製 keypoints / 顯示 FPS。"""
        img = bgr.copy()
        h, w = img.shape[:2]

        if draw_keypoints:
            # Face dots
            if results['face'] and results['face'].face_landmarks:
                for face in results['face'].face_landmarks:
                    for lm in face:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

            # Pose points + bones
            if results['pose'] and results['pose'].pose_landmarks:
                for pose in results['pose'].pose_landmarks:
                    for lm in pose:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
                    for a, b in [(11,12),(11,13),(13,15),(12,14),(14,16),
                                 (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]:
                        if a < len(pose) and b < len(pose):
                            p1 = (int(pose[a].x*w), int(pose[a].y*h))
                            p2 = (int(pose[b].x*w), int(pose[b].y*h))
                            cv2.line(img, p1, p2, (255, 0, 0), 2)

            # Hands points + bones
            if results['hand'] and results['hand'].hand_landmarks:
                for hand in results['hand'].hand_landmarks:
                    for lm in hand:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
                    for a, b in [(0,1),(1,2),(2,3),(3,4),
                                 (0,5),(5,6),(6,7),(7,8),
                                 (0,9),(9,10),(10,11),(11,12),
                                 (0,13),(13,14),(14,15),(15,16),
                                 (0,17),(17,18),(18,19),(19,20)]:
                        if a < len(hand) and b < len(hand):
                            p1 = (int(hand[a].x*w), int(hand[a].y*h))
                            p2 = (int(hand[b].x*w), int(hand[b].y*h))
                            cv2.line(img, p1, p2, (0, 0, 255), 2)

        if show_info and fps_value is not None:
            txt = f"FPS: {fps_value:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(img, txt, (img.shape[1]-tw-10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        return img

    def close(self):
        self.face.close()
        self.pose.close()
        self.hand.close()


def run_image(det: ThreeInOneDetector, path: str, save_info: bool, save_keypoints: bool):
    if not Path(path).exists():
        raise FileNotFoundError(f"找不到圖片：{path}")
    bgr = cv2.imread(path)
    res = det.detect_image(bgr)
    vis = det.draw(res, bgr, draw_keypoints=save_keypoints, show_info=save_info, fps_value=None)
    cv2.imshow("Result - IMAGE", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(det: ThreeInOneDetector, path: str, out_path: str | None, save_info: bool, save_keypoints: bool):
    if not Path(path).exists():
        raise FileNotFoundError(f"找不到影片：{path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps_src, (W, H))

    t_prev = time.time()
    i = 0
    while True:
        ret, bgr = cap.read()
        if not ret: break
        ts_ms = int(i * 1000 / fps_src)
        res = det.detect_video_frame(bgr, ts_ms)

        now = time.time()
        runfps = 1.0 / (now - t_prev) if now > t_prev else 0.0
        t_prev = now

        vis = det.draw(res, bgr, draw_keypoints=save_keypoints, show_info=save_info, fps_value=runfps)

        if writer: writer.write(vis)
        cv2.imshow("Result - VIDEO", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        i += 1

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()


def run_cam(det: ThreeInOneDetector, cam_id: int, out_path: str | None, save_info: bool, save_keypoints: bool):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟攝影機 id={cam_id}")

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    if not fps_src or fps_src <= 0: fps_src = 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps_src, (W, H))

    t_prev = time.time()
    while True:
        ret, bgr = cap.read()
        if not ret: break

        res = det.detect_stream_frame(bgr)

        now = time.time()
        runfps = 1.0 / (now - t_prev) if now > t_prev else 0.0
        t_prev = now

        vis = det.draw(res, bgr, draw_keypoints=save_keypoints, show_info=save_info, fps_value=runfps)

        if writer: writer.write(vis)
        cv2.imshow("Result - CAM", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()


def main():
    args = build_parser().parse_args()

    models = auto_detect_models(args.models_dir)
    det = ThreeInOneDetector(mode=args.mode, models=models, silent=args.silent)

    try:
        if args.mode == "IMAGE":
            if not args.input:
                raise ValueError("IMAGE 模式需提供 --input 圖片路徑")
            run_image(det, args.input, save_info=args.save_info, save_keypoints=args.save_keypoints)

        elif args.mode == "VIDEO":
            if not args.input:
                raise ValueError("VIDEO 模式需提供 --input 影片路徑")
            run_video(det, args.input, args.out_video, save_info=args.save_info, save_keypoints=args.save_keypoints)

        elif args.mode == "CAM":
            run_cam(det, cam_id=args.cam_id, out_path=args.out_video, save_info=args.save_info, save_keypoints=args.save_keypoints)

    finally:
        det.close()


if __name__ == "__main__":
    main()
