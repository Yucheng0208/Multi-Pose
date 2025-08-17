import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import argparse
import time  # <--- 導入 time 模組
from typing import Union, Optional, Tuple, Dict, List
# 必需導入，用於資料格式轉換
from mediapipe.framework.formats import landmark_pb2

class SignLanguageDetector:
    def __init__(self, 
                 yolo_model_path: str = "models/yolo11n-pose.pt",
                 hand_model_path: str = "models/hand_landmarker.task",
                 face_model_path: str = "models/face_landmarker.task",
                 device: str = "cuda",
                 confidence: float = 0.5):
        """
        初始化手語識別系統
        """
        # 初始化YOLO模型
        self.yolo_model = YOLO(yolo_model_path)
        self.device = device
        self.confidence = confidence
        
        # 初始化MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.use_task_models = True
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            hand_base_options = python.BaseOptions(model_asset_path=hand_model_path)
            hand_options = vision.HandLandmarkerOptions(
                base_options=hand_base_options,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            
            face_base_options = python.BaseOptions(model_asset_path=face_model_path)
            face_options = vision.FaceLandmarkerOptions(
                base_options=face_base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
                min_face_detection_confidence=0.7,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
            print("✅ 使用新版 .task 模型")
        except Exception as e:
            print(f"⚠️  新版模型載入失敗，使用舊版API: {e}")
            self.use_task_models = False
            self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
            self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)
        
        self.current_roi_coords = {'face': None}
        
    def extract_face_roi_from_pose(self, image: np.ndarray, pose_keypoints: np.ndarray) -> Optional[np.ndarray]:
        """
        根據YOLO姿態關鍵點僅提取臉部ROI
        """
        h, w = image.shape[:2]
        face_roi = None
        self.current_roi_coords['face'] = None
        
        if pose_keypoints is None:
            return None
            
        keypoints = pose_keypoints.reshape(-1, 3)
        face_points = keypoints[0:5]
        valid_face_points = face_points[face_points[:, 2] > 0.3]
        
        if len(valid_face_points) > 0:
            x_min = int(max(0, valid_face_points[:, 0].min() - 80))
            y_min = int(max(0, valid_face_points[:, 1].min() - 80))
            x_max = int(min(w, valid_face_points[:, 0].max() + 80))
            y_max = int(min(h, valid_face_points[:, 1].max() + 120))
            
            if x_max > x_min and y_max > y_min:
                face_roi = image[y_min:y_max, x_min:x_max]
                self.current_roi_coords['face'] = (x_min, y_min, x_max, y_max)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                cv2.putText(image, "Face_ROI", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        return face_roi

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        處理單一幀，提取所有關鍵點
        """
        keypoints_data = {'pose': None, 'hands': {'left': None, 'right': None}, 'face': None}
        
        results = self.yolo_model(frame, conf=self.confidence, device=self.device)
        pose_keypoints = None
        if results[0].keypoints and len(results[0].keypoints.data) > 0:
            pose_keypoints = results[0].keypoints.data[0].cpu().numpy()
            keypoints_data['pose'] = pose_keypoints
            self.draw_pose_keypoints(frame, pose_keypoints)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.use_task_models:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            hand_results = self.hand_landmarker.detect(mp_image)
            if hand_results.hand_landmarks:
                for i, handedness in enumerate(hand_results.handedness):
                    hand_label = handedness[0].category_name.lower()
                    keypoints_data['hands'][hand_label] = hand_results.hand_landmarks[i]
                self.draw_hand_landmarks(frame, hand_results, is_new_api=True)
        else:
            hand_results = self.hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                for i, handedness in enumerate(hand_results.multi_handedness):
                     hand_label = handedness.classification[0].label.lower()
                     keypoints_data['hands'][hand_label] = hand_results.multi_hand_landmarks[i]
                self.draw_hand_landmarks(frame, hand_results, is_new_api=False)

        if pose_keypoints is not None:
            face_roi = self.extract_face_roi_from_pose(frame, pose_keypoints)
            if face_roi is not None and face_roi.size > 0:
                try:
                    if self.use_task_models:
                        mp_face_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                        face_results = self.face_landmarker.detect(mp_face_image)
                        if face_results.face_landmarks:
                            keypoints_data['face'] = face_results.face_landmarks[0]
                            self.draw_face_landmarks(frame, face_roi, face_results, is_new_api=True)
                    else:
                        face_results = self.face_mesh.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                        if face_results.multi_face_landmarks:
                            keypoints_data['face'] = face_results.multi_face_landmarks[0]
                            self.draw_face_landmarks(frame, face_roi, face_results, is_new_api=False)
                except Exception as e:
                    print(f"臉部檢測錯誤: {e}")
        
        return frame, keypoints_data

    def draw_pose_keypoints(self, image: np.ndarray, keypoints: np.ndarray):
        """繪製YOLO姿態關鍵點"""
        keypoints = keypoints.reshape(-1, 3)
        connections = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        for _, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:
                cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)
        for start, end in connections:
            if start < len(keypoints) and end < len(keypoints):
                start_point, end_point = keypoints[start], keypoints[end]
                if start_point[2] > 0.3 and end_point[2] > 0.3:
                    cv2.line(image, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), (255, 0, 0), 2)
    
    def draw_hand_landmarks(self, frame: np.ndarray, hand_results, is_new_api: bool = True):
        """直接在主畫面上繪製手部關鍵點"""
        if is_new_api:
            if hand_results.hand_landmarks:
                for hand_landmarks_list in hand_results.hand_landmarks:
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks_list
                    ])
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks_proto,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
        else:
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

    def draw_face_landmarks(self, frame: np.ndarray, face_roi: np.ndarray, face_results, is_new_api: bool = True):
        """繪製臉部網格和所有478個關鍵點"""
        roi_coords = self.current_roi_coords.get('face')
        if not roi_coords: return
            
        x_min, y_min, _, _ = roi_coords
        target_image = frame[y_min:y_min+face_roi.shape[0], x_min:x_min+face_roi.shape[1]]

        landmarks_to_draw = []
        if is_new_api and face_results.face_landmarks:
            landmarks_to_draw = face_results.face_landmarks
        elif not is_new_api and face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=target_image,
                    landmark_list=landmarks,
                    connections=self.mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=-1, circle_radius=1),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
            return
        
        for face_landmarks_list in landmarks_to_draw:
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks_list
            ])
            self.mp_drawing.draw_landmarks(
                image=target_image,
                landmark_list=face_landmarks_proto,
                connections=self.mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=-1, circle_radius=1),
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())

    # --- MODIFIED FUNCTION ---
    def process_realtime(self, camera_id: int = 0):
        """即時攝像頭處理，並計算真實的處理幀率"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"錯誤: 無法開啟攝像頭 ID {camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("即時手語識別已啟動，按 'q' 退出")
        
        # 用於計算真實FPS的變數
        prev_time = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # --- 開始計時 ---
                start_time = time.time()
                
                processed_frame, keypoints = self.process_frame(frame)
                
                # --- 結束計時並計算真實FPS ---
                current_time = time.time()
                # 避免除以零的錯誤
                if (current_time - prev_time) > 0:
                    real_fps = 1 / (current_time - prev_time)
                else:
                    real_fps = 0
                prev_time = current_time
                
                # 在畫面上顯示真實的FPS
                fps_text = f"Real FPS: {real_fps:.1f}"
                cv2.putText(processed_frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # 用紅色顯示以示區別
                
                # 顯示檢測狀態
                status_y = 70
                status_info = [
                    ("Pose", keypoints['pose'] is not None),
                    ("L_Hand", keypoints['hands']['left'] is not None),
                    ("R_Hand", keypoints['hands']['right'] is not None),
                    ("Face", keypoints['face'] is not None)
                ]
                for name, detected in status_info:
                    text = f"{name}: {'OK' if detected else 'X'}"
                    color = (0, 255, 0) if detected else (0, 0, 255)
                    cv2.putText(processed_frame, text, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    status_y += 30

                cv2.imshow('Sign Language Detection', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
    def process_image(self, image_path: str, output_path: Optional[str] = None):
        """處理單張圖片"""
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"無法讀取圖片: {image_path}")
        processed_image, _ = self.process_frame(image)
        if output_path:
            cv2.imwrite(output_path, processed_image)
            print(f"處理結果已保存到: {output_path}")
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_video(self, video_path: str, output_path: Optional[str] = None):
        """處理影片"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError(f"無法開啟影片: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) if output_path else None
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                processed_frame, _ = self.process_frame(frame)
                if out: out.write(processed_frame)
                frame_count += 1
                if frame_count % 30 == 0: print(f"處理進度: {100*frame_count/total_frames:.1f}%")
        finally:
            cap.release()
            if out: out.release()
            print("影片處理完成。")


def main():
    parser = argparse.ArgumentParser(description='手語識別系統')
    parser.add_argument('--mode', choices=['image', 'video', 'realtime'], default='realtime', help='處理模式')
    parser.add_argument('--input', type=str, help='輸入檔案路徑')
    parser.add_argument('--output', type=str, help='輸出檔案路徑')
    parser.add_argument('--camera', type=int, default=0, help='攝像頭ID')
    parser.add_argument('--model', type=str, default='models/yolo11n-pose.pt', help='YOLO模型路徑')
    parser.add_argument('--hand_model', type=str, default='models/hand_landmarker.task', help='MediaPipe手部模型路徑')
    parser.add_argument('--face_model', type=str, default='models/face_landmarker.task', help='MediaPipe臉部模型路徑')
    parser.add_argument('--device', type=str, default='cuda', help='運行設備')
    parser.add_argument('--confidence', type=float, default=0.5, help='檢測信心度閾值')
    
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
        confidence=args.confidence
    )
    
    if args.mode == 'image':
        if not args.input: print("錯誤：圖片模式需要 --input 參數"); return
        detector.process_image(args.input, args.output)
    elif args.mode == 'video':
        if not args.input: print("錯誤：影片模式需要 --input 參數"); return
        detector.process_video(args.input, args.output)
    elif args.mode == 'realtime':
        detector.process_realtime(args.camera)

if __name__ == "__main__":
    main()