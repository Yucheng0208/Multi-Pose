import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import argparse
import time
import json
import os
from typing import Union, Optional, Tuple, Dict, List
# 必需導入，用於資料格式轉換
from mediapipe.framework.formats import landmark_pb2

# JSON編碼器，用於處理Numpy數據類型
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class SignLanguageDetector:
    def __init__(self, 
                 yolo_model_path: str = "models/yolo11n-pose.pt",
                 hand_model_path: str = "models/hand_landmarker.task",
                 face_model_path: str = "models/face_landmarker.task",
                 device: str = "cuda",
                 confidence: float = 0.5,
                 base_output_dir: str = "outputs"):
        """
        初始化手語識別系統
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.device = device
        self.confidence = confidence
        
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.use_task_models = True
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            hand_base_options = python.BaseOptions(model_asset_path=hand_model_path)
            hand_options = vision.HandLandmarkerOptions(base_options=hand_base_options, num_hands=2, min_hand_detection_confidence=0.5)
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            
            face_base_options = python.BaseOptions(model_asset_path=face_model_path)
            face_options = vision.FaceLandmarkerOptions(base_options=face_base_options, output_face_blendshapes=True, num_faces=1, min_face_detection_confidence=0.7)
            self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
            print("✅ 使用新版 .task 模型")
        except Exception as e:
            print(f"⚠️  新版模型載入失敗，使用舊版API: {e}")
            # ... (舊版API初始化)
        
        self.current_roi_coords = {'face': None}
        self.frame_counter = 0
        self.base_output_dir = base_output_dir
        self.output_json_dir = None # 將在每次執行時動態設定
        
    def extract_face_roi_from_pose(self, image: np.ndarray, pose_keypoints: np.ndarray, person_id: int) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        self.current_roi_coords['face'] = None
        keypoints = pose_keypoints.reshape(-1, 3)
        face_points = keypoints[0:5]
        valid_face_points = face_points[face_points[:, 2] > 0.3]
        
        if len(valid_face_points) > 0:
            x_min = int(max(0, valid_face_points[:, 0].min() - 80))
            y_min = int(max(0, valid_face_points[:, 1].min() - 80))
            x_max = int(min(w, valid_face_points[:, 0].max() + 80))
            y_max = int(min(h, valid_face_points[:, 1].max() + 120))
            
            if x_max > x_min and y_max > y_min:
                self.current_roi_coords['face'] = (x_min, y_min, x_max, y_max)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                cv2.putText(image, f"Face_ROI (P{person_id + 1})", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                return image[y_min:y_max, x_min:x_max]
        return None

    def _build_and_save_json(self, num_persons, keypoints_data, frame_shape):
        self.frame_counter += 1
        frame_json_data = {'frame_id': self.frame_counter, 'num_persons': num_persons, 'persons': []}
        h, w = frame_shape

        if keypoints_data['pose'] is not None:
            person_data = {'person_id': 0, 'keypoints': {}}
            person_data['keypoints']['pose'] = [{'id': idx, 'x': kp[0], 'y': kp[1], 'confidence': kp[2]} for idx, kp in enumerate(keypoints_data['pose'])]
            
            # ... (手部和臉部關鍵點處理)
            left_hand_kps, right_hand_kps = [], []
            for hand_label in ['left', 'right']:
                hand_data = keypoints_data['hands'][hand_label]
                if hand_data:
                    landmarks, confidence = hand_data
                    target_list = left_hand_kps if hand_label == 'left' else right_hand_kps
                    for idx, lm in enumerate(landmarks):
                        target_list.append({'id': idx, 'x': lm.x * w, 'y': lm.y * h, 'confidence': confidence})
            person_data['keypoints']['left_hand'] = left_hand_kps
            person_data['keypoints']['right_hand'] = right_hand_kps

            face_kps = []
            if keypoints_data['face']:
                landmarks, confidence = keypoints_data['face']
                roi_coords = self.current_roi_coords.get('face')
                if roi_coords:
                    x_min, y_min, x_max, y_max = roi_coords
                    roi_w, roi_h = x_max - x_min, y_max - y_min
                    for idx, lm in enumerate(landmarks):
                        face_kps.append({'id': idx, 'x': x_min + lm.x * roi_w, 'y': y_min + lm.y * roi_h, 'confidence': confidence})
            person_data['keypoints']['face'] = face_kps
            
            frame_json_data['persons'].append(person_data)

        filename = f"{self.frame_counter:012d}.json"
        # 確保 self.output_json_dir 已經被設定
        if self.output_json_dir:
            filepath = os.path.join(self.output_json_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(frame_json_data, f, indent=4, cls=NpEncoder)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict, int]:
        h, w, _ = frame.shape
        keypoints_data = {'pose': None, 'hands': {'left': None, 'right': None}, 'face': None}
        
        yolo_results = self.yolo_model(frame, conf=self.confidence, device=self.device)
        num_persons = len(yolo_results[0]) if yolo_results[0] else 0
        
        pose_keypoints = None
        if num_persons > 0 and yolo_results[0].keypoints:
            pose_keypoints = yolo_results[0].keypoints.data[0].cpu().numpy()
            keypoints_data['pose'] = pose_keypoints
            self.draw_pose_keypoints(frame, pose_keypoints)

        # ... (手部和臉部偵測)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        hand_results = self.hand_landmarker.detect(mp_image)
        if hand_results.hand_landmarks:
            for i, handedness in enumerate(hand_results.handedness):
                hand_label = handedness[0].category_name.lower()
                confidence = handedness[0].score
                landmarks = hand_results.hand_landmarks[i]
                keypoints_data['hands'][hand_label] = (landmarks, confidence)
            self.draw_hand_landmarks(frame, hand_results)

        if pose_keypoints is not None:
            face_roi = self.extract_face_roi_from_pose(frame, pose_keypoints, person_id=0)
            if face_roi is not None and face_roi.size > 0:
                mp_face_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                face_results = self.face_landmarker.detect(mp_face_image)
                if face_results.face_landmarks:
                    keypoints_data['face'] = (face_results.face_landmarks[0], 1.0)
                    self.draw_face_landmarks(frame, face_roi, face_results)

        self._build_and_save_json(num_persons, keypoints_data, (h, w))
        
        return frame, keypoints_data, num_persons

    def draw_pose_keypoints(self, image: np.ndarray, keypoints: np.ndarray):
        # ... (繪圖邏輯)
        keypoints = keypoints.reshape(-1, 3)
        connections = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        for _, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3: cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)
        for start, end in connections:
            if start < len(keypoints) and end < len(keypoints):
                sp, ep = keypoints[start], keypoints[end]
                if sp[2] > 0.3 and ep[2] > 0.3:
                    cv2.line(image, (int(sp[0]), int(sp[1])), (int(ep[0]), int(ep[1])), (255, 0, 0), 2)
    
    def draw_hand_landmarks(self, frame: np.ndarray, hand_results):
        # ... (繪圖邏輯)
        if hand_results.hand_landmarks:
            for hand_landmarks_list in hand_results.hand_landmarks:
                proto = landmark_pb2.NormalizedLandmarkList()
                proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks_list])
                self.mp_drawing.draw_landmarks(
                    frame, proto, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

    def draw_face_landmarks(self, frame: np.ndarray, face_roi: np.ndarray, face_results):
        # ... (繪圖邏輯)
        roi_coords = self.current_roi_coords.get('face')
        if not roi_coords: return
        x_min, y_min, _, _ = roi_coords
        target_image = frame[y_min:y_min+face_roi.shape[0], x_min:x_min+face_roi.shape[1]]

        if face_results.face_landmarks:
            for face_landmarks_list in face_results.face_landmarks:
                proto = landmark_pb2.NormalizedLandmarkList()
                proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in face_landmarks_list])
                self.mp_drawing.draw_landmarks(
                    image=target_image, landmark_list=proto,
                    connections=self.mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=-1, circle_radius=1),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())

    # --- MODIFIED FUNCTION ---
    def process_realtime(self, camera_id: int = 0):
        """即時攝像頭處理，並自動儲存影片和JSON"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"錯誤: 無法開啟攝像頭 ID {camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # --- 自動化輸出設定 ---
        # 1. 產生本次運行的時間戳
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        # 2. 設定並建立影片輸出路徑
        video_dir = os.path.join(self.base_output_dir, "media")
        os.makedirs(video_dir, exist_ok=True)
        video_filepath = os.path.join(video_dir, f"{timestamp}.mp4")
        
        # 3. 設定並建立JSON輸出路徑
        self.output_json_dir = os.path.join(self.base_output_dir, "json", timestamp)
        os.makedirs(self.output_json_dir, exist_ok=True)
        
        # 4. 設定影片寫入器
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filepath, fourcc, fps, (width, height))
        
        print(f"✅ 將自動錄製影片至: {video_filepath}")
        print(f"✅ JSON檔案將儲存於: {self.output_json_dir}")
        print("即時手語識別已啟動，按 'esc' 退出")
        
        prev_time = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                processed_frame, keypoints, num_persons = self.process_frame(frame)
                
                # ... (繪製FPS, 人數, 狀態等)
                current_time = time.time()
                real_fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                prev_time = current_time
                cv2.putText(processed_frame, f"Real FPS: {real_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                h, w, _ = processed_frame.shape
                cv2.putText(processed_frame, f"Persons: {num_persons}", (w - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                status_y = 70
                status_info = [("Pose", keypoints['pose'] is not None),
                               ("L_Hand", keypoints['hands']['left'] is not None),
                               ("R_Hand", keypoints['hands']['right'] is not None),
                               ("Face", keypoints['face'] is not None)]
                for name, detected in status_info:
                    text, color = (f"{name}: OK", (0, 255, 0)) if detected else (f"{name}: X", (0, 0, 255))
                    cv2.putText(processed_frame, text, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    status_y += 30

                cv2.imshow('Sign Language Detection', processed_frame)
                
                # 將處理後的畫面寫入影片檔
                out.write(processed_frame)

                if cv2.waitKey(1) & 0xFF == 27: break

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"\n影片已成功儲存至: {video_filepath}")
            print(f"JSON檔案已儲存於: {self.output_json_dir}")

def main():
    parser = argparse.ArgumentParser(description='手語識別系統')
    parser.add_argument('--mode', choices=['image', 'video', 'realtime'], default='realtime')
    parser.add_argument('--input', type=str, help='輸入檔案路徑 (用於 image/video 模式)')
    parser.add_argument('--output', type=str, help='輸出檔案路徑 (用於 image/video 模式)')
    parser.add_argument('--camera', type=int, default=0, help='攝像頭ID')
    parser.add_argument('--model', type=str, default='models/yolo11n-pose.pt')
    parser.add_argument('--hand_model', type=str, default='models/hand_landmarker.task')
    parser.add_argument('--face_model', type=str, default='models/face_landmarker.task')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--confidence', type=float, default=0.5)
    
    args = parser.parse_args()
    
    device = args.device
    if 'cuda' in device and not torch.cuda.is_available():
        print(f"警告: CUDA ({device}) 不可用, 將切換至 CPU.")
        device = 'cpu'

    detector = SignLanguageDetector(
        yolo_model_path=args.model,
        hand_model_path=args.hand_model,
        face_model_path=args.face_model,
        device=device,
        confidence=args.confidence,
        base_output_dir="outputs" # 指定基礎輸出資料夾
    )
    
    if args.mode == 'realtime':
        detector.process_realtime(args.camera)

if __name__ == "__main__":
    main()