# -*- coding: utf-8 -*-
import mediapipe as mp
import numpy as np
import cv2
import config
from typing import List, Tuple

class MediapipeProcessor:
    def __init__(self, face_task, pose_task, hand_task, use_face=True, use_pose=True, use_hands=True):
        # 壓 absl 噪音
        try:
            from absl import logging as absl_logging
            absl_logging.set_verbosity(absl_logging.ERROR)
        except Exception:
            pass

        self.use_face = use_face
        self.use_pose = use_pose
        self.use_hands = use_hands

        self.BaseOptions = mp.tasks.BaseOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        # 以 IMAGE 模式運行
        mode = self.VisionRunningMode.IMAGE

        # Face
        if self.use_face:
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            face_opts = FaceLandmarkerOptions(
                base_options=self.BaseOptions(model_asset_path=str(face_task)),
                running_mode=mode,
                num_faces=2,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
            )
            self.face_detector = FaceLandmarker.create_from_options(face_opts)
        else:
            self.face_detector = None

        # Pose
        if self.use_pose:
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            pose_opts = PoseLandmarkerOptions(
                base_options=self.BaseOptions(model_asset_path=str(pose_task)),
                running_mode=mode,
                num_poses=2,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            self.pose_detector = PoseLandmarker.create_from_options(pose_opts)
        else:
            self.pose_detector = None

        # Hands
        if self.use_hands:
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            hand_opts = HandLandmarkerOptions(
                base_options=self.BaseOptions(model_asset_path=str(hand_task)),
                running_mode=mode,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.hand_detector = HandLandmarker.create_from_options(hand_opts)
        else:
            self.hand_detector = None

    def close(self):
        if self.face_detector: self.face_detector.close()
        if self.pose_detector: self.pose_detector.close()
        if self.hand_detector: self.hand_detector.close()

    @staticmethod
    def _get_conf(lm):
        # 優先用 visibility/presence，缺就 1.0
        for k in ("visibility", "presence"):
            if hasattr(lm, k):
                v = getattr(lm, k)
                if v is not None:
                    return float(v)
        return 1.0

    @staticmethod
    def _front_by_interocular(landmarks) -> bool:
        """
        以 FaceMesh 468 索引系統：33(左眼外側)、263(右眼外側)，
        用兩眼距 / 臉寬作為閾值，粗判是否正面。
        """
        try:
            li = landmarks[33]
            ri = landmarks[263]
            # 眼距（歸一化）
            d = np.hypot(li.x - ri.x, li.y - ri.y)
            # 臉寬近似（以兩眼到鼻翼比例，保守閾值）
            return d > 0.12  # 視素材可調，越小越寬鬆
        except Exception:
            return False

    def analyze_crop(self, crop_rgb: np.ndarray, timestamp_ms: int,
                     kp_face_count=1, kp_pose_count=1, need_front_pose=True):
        """
        輸入：ROI RGB 圖
        輸出：is_front(bool), kp_all(List[dict]), debug(dict)
        kp_all: {"kid":全域重新編號, "x":0~1(ROI內), "y":0~1(ROI內), "conf":float}
        """

        mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

        face_res = self.face_detector.detect(mpimg) if self.face_detector else None
        pose_res = self.pose_detector.detect(mpimg) if self.pose_detector else None
        hand_res = self.hand_detector.detect(mpimg) if self.hand_detector else None

        # 是否正面
        is_front = True
        face_lms = []
        if face_res and face_res.face_landmarks:
            face_lms = face_res.face_landmarks[0]
            is_front = self._front_by_interocular(face_lms)
        else:
            is_front = False

        if need_front_pose and not is_front:
            return False, [], {}

        # 重新編號：0..542（468 face + 33 pose + 21 LH + 21 RH）
        kp_all = []
        kbase = 0

        # Face
        if self.use_face and face_lms:
            face_points = face_lms
            if config.FACE_USE_468 and len(face_points) >= 468:
                face_points = face_points[:468]  # 截到 468
            for i, lm in enumerate(face_points):
                kp_all.append({"kid": kbase + i, "x": float(lm.x), "y": float(lm.y), "conf": self._get_conf(lm)})
            kbase += len(face_points)

        # Pose
        if self.use_pose and pose_res and pose_res.pose_landmarks:
            pose_points = pose_res.pose_landmarks[0]
            for i, lm in enumerate(pose_points):
                kp_all.append({"kid": kbase + i, "x": float(lm.x), "y": float(lm.y), "conf": self._get_conf(lm)})
            kbase += len(pose_points)

        # Hands（最多兩手，順序：左、右（以 x 中心較小者視為左））
        hand_points_all = []
        if self.use_hands and hand_res and hand_res.handedness and hand_res.hand_landmarks:
            # MediaPipe 提供 handedness，但保守起見用 x 排序再貼標籤
            for h_lms in hand_res.hand_landmarks:
                xs = [lm.x for lm in h_lms]
                cx = float(np.mean(xs))
                hand_points_all.append((cx, h_lms))
            hand_points_all.sort(key=lambda t: t[0])  # 左->右

            for _, h_lms in hand_points_all[:2]:
                for i, lm in enumerate(h_lms):
                    kp_all.append({"kid": kbase + i, "x": float(lm.x), "y": float(lm.y), "conf": self._get_conf(lm)})
                kbase += len(h_lms)

        return is_front, kp_all, {}

    # 視覺化：把 ROI 內 0~1 座標畫回全圖
    @staticmethod
    def draw_keypoints_on(frame_bgr: np.ndarray, kp_all: List[dict], roi_xyxy: Tuple[int,int,int,int]):
        x1, y1, x2, y2 = map(int, roi_xyxy)
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        for kp in kp_all:
            px = int(x1 + kp["x"] * w)
            py = int(y1 + kp["y"] * h)
            cv2.circle(frame_bgr, (px, py), 2, (0, 255, 255), -1, cv2.LINE_AA)
