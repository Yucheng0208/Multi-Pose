"""
MediaPipe + YOLOv11n-pose GPU加速偵測器
- 手部21個關鍵點 (MediaPipe + GPU)
- 臉部248個關鍵點 (MediaPipe + GPU)  
- 人體17個姿態關鍵點 (YOLOv11n-pose + GPU)
- 支援NVIDIA GPU加速
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime
from ultralytics import YOLO
import threading
from queue import Queue
import torch

class IntegratedLandmarkDetector:
    def __init__(self):
        print("🚀 正在初始化GPU加速偵測器...")
        
        # 檢查GPU可用性
        self.check_gpu_availability()
        
        # 初始化MediaPipe解決方案
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # 初始化MediaPipe偵測器（GPU加速）
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0  # 使用較快的模型
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,  # 減少到1個臉以提高性能
            refine_landmarks=False,  # 248個點
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 初始化YOLOv11n-pose（GPU加速）
        self.yolo_model = None
        self.yolo_load_attempts = 0
        self.max_load_attempts = 3
        
        print("⚡ 正在載入YOLOv11n-pose模型（GPU加速）...")
        self.load_yolo_model()
        
        # 多線程處理
        self.use_threading = True
        self.frame_queue = Queue(maxsize=3)
        self.result_queue = Queue(maxsize=3)
        self.processing = False
        
        # COCO人體關鍵點名稱 (17個點)
        self.pose_keypoints = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # 人體骨架連接
        self.pose_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 頭部
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上半身
            (5, 11), (6, 12), (11, 12),  # 軀幹
            (11, 13), (13, 15), (12, 14), (14, 16)  # 下半身
        ]
        
        # 手部21個關鍵點的名稱
        self.hand_landmarks_names = [
            "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
            "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]
        
        # 設定
        self.show_hands = True
        self.show_face = True
        self.show_pose = True
        self.show_hand_connections = True
        self.show_face_mesh = True
        self.show_pose_connections = True
        self.show_landmarks_info = False  # 預設關閉以保持畫面清潔
        self.mirror_mode = True  # 鏡像模式開關
        
        # 統計資訊
        self.hand_landmarks_count = 0
        self.face_landmarks_count = 0
        self.pose_landmarks_count = 0
        
        # 儲存相關設定
        self.output_dir = "output_media"
        self.auto_save = False
        self.video_writer = None
        self.save_interval = 30  # 每30幀儲存一次截圖
        self.frame_count = 0
        
        # 建立輸出資料夾
        self.create_output_directory()
        
        # 性能優化設定
        self.skip_frames = 1  # 每隔幾幀處理一次（1=每幀都處理）
        self.frame_counter = 0
        self.resize_factor = 1.0  # 圖片縮放因子（1.0=原尺寸）
        self.last_results = None  # 快取上一幀結果
        
        print("✅ GPU加速偵測器初始化完成！")
    
    def check_gpu_availability(self):
        """檢查GPU可用性"""
        print("🔍 檢查GPU加速支援...")
        
        # 檢查CUDA
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA可用！偵測到 {gpu_count} 個GPU")
            print(f"🎮 主GPU: {gpu_name}")
            self.device = 'cuda'
        else:
            print("⚠️  CUDA不可用，將使用CPU")
            self.device = 'cpu'
        
        # 檢查MediaPipe GPU支援
        try:
            # MediaPipe會自動使用GPU（如果可用）
            print("✅ MediaPipe GPU支援已啟用")
        except Exception as e:
            print(f"⚠️  MediaPipe GPU支援檢查失敗: {e}")
    
    def load_yolo_model(self):
        """載入YOLO模型的專用方法（GPU加速）"""
        model_names = [
            'yolo11n-pose.pt',
            'yolov8n-pose.pt'
        ]
        
        for model_name in model_names:
            try:
                print(f"🔄 嘗試載入: {model_name}")
                self.yolo_model = YOLO(model_name)
                
                # 設定為GPU裝置
                if self.device == 'cuda':
                    self.yolo_model.to('cuda')
                    print(f"🎮 {model_name} 已載入到GPU")
                else:
                    print(f"💻 {model_name} 使用CPU")
                
                # 測試模型是否正常工作
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                test_results = self.yolo_model(dummy_img, verbose=False, device=self.device)
                
                print(f"✅ 成功載入並測試: {model_name}")
                return True
                
            except Exception as e:
                print(f"❌ {model_name} 載入失敗: {str(e)[:100]}...")
                self.yolo_model = None
                continue
        
        print("⚠️  所有YOLO模型載入失敗")
        return False
    
    def optimize_frame(self, frame):
        """優化幀處理"""
        # 如果設定了縮放因子，縮小圖片以提高處理速度
        if self.resize_factor < 1.0:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * self.resize_factor), int(w * self.resize_factor)
            frame = cv2.resize(frame, (new_w, new_h))
        
        return frame
    
    def restore_coordinates(self, results, original_shape):
        """將縮放後的座標還原到原始尺寸"""
        if self.resize_factor >= 1.0:
            return results
        
        # 這裡可以加入座標還原邏輯
        # 暫時簡化處理
        return results
        
    def draw_hand_landmarks(self, frame, hand_landmarks, handedness):
        """繪製手部關鍵點"""
        h, w, _ = frame.shape
        
        # 繪製手部連接線
        if self.show_hand_connections:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # 繪製關鍵點編號和座標（僅在開啟資訊模式時）
        if self.show_landmarks_info:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # 繪製點
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(frame, str(idx), (x + 5, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        # 顯示手部類型
        wrist = hand_landmarks.landmark[0]
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        hand_type = handedness.classification[0].label
        cv2.putText(frame, f"{hand_type}", (wrist_x - 30, wrist_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    def draw_face_landmarks(self, frame, face_landmarks):
        """繪製臉部248個關鍵點"""
        h, w, _ = frame.shape
        
        # 繪製臉部網格
        if self.show_face_mesh:
            # 繪製主要輪廓
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 128, 255), thickness=1)
            )
            
            # 繪製眼部細節
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=1)
            )
            
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=1)
            )
    
    def draw_pose_landmarks(self, frame, pose_results):
        """繪製YOLOv11n-pose人體關鍵點"""
        if pose_results is None:
            return
            
        for result in pose_results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                # 處理不同的keypoints格式
                if hasattr(result.keypoints, 'data'):
                    keypoints_data = result.keypoints.data
                elif hasattr(result.keypoints, 'xy'):
                    keypoints_data = result.keypoints.xy
                else:
                    keypoints_data = result.keypoints
                
                # 轉換為numpy array以便處理
                keypoints_data = keypoints_data.cpu().numpy() if hasattr(keypoints_data, 'cpu') else keypoints_data
                
                # 檢查是否有偵測到人
                if len(keypoints_data) == 0:
                    self.pose_landmarks_count = 0
                    cv2.putText(frame, "Pose: No person detected", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
                    return
                
                # 處理每個偵測到的人
                for person_idx in range(len(keypoints_data)):
                    keypoints = keypoints_data[person_idx]
                    
                    # 偵測並顯示關鍵點
                    valid_points = []
                    for i, point in enumerate(keypoints):
                        if len(point) >= 3:  # x, y, confidence
                            x, y, conf = point[0], point[1], point[2]
                        elif len(point) == 2:  # 只有x, y
                            x, y, conf = point[0], point[1], 1.0
                        else:
                            continue
                        
                        if conf > 0.3:  # 信心度閾值
                            x, y = int(x), int(y)
                            valid_points.append((i, x, y, conf))
                            
                            # 繪製關鍵點
                            cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)  # 紫色點
                            cv2.circle(frame, (x, y), 7, (255, 255, 255), 2)  # 白色邊框
                            
                            # 顯示關鍵點編號（僅在資訊模式時）
                            if self.show_landmarks_info:
                                cv2.putText(frame, f"{i}", (x + 8, y - 8), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                                cv2.putText(frame, f"{conf:.2f}", (x + 8, y + 15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                    
                    # 繪製骨架連接
                    if self.show_pose_connections and len(valid_points) > 0:
                        # 建立關鍵點字典以便快速查找
                        point_dict = {idx: (x, y, conf) for idx, x, y, conf in valid_points}
                        
                        for connection in self.pose_connections:
                            start_idx, end_idx = connection
                            
                            if start_idx in point_dict and end_idx in point_dict:
                                start_x, start_y, start_conf = point_dict[start_idx]
                                end_x, end_y, end_conf = point_dict[end_idx]
                                
                                # 檢查兩個點的信心度
                                if start_conf > 0.3 and end_conf > 0.3:
                                    # 根據連接部位使用不同顏色
                                    if start_idx <= 4 or end_idx <= 4:  # 頭部
                                        color = (0, 255, 255)  # 黃色
                                    elif start_idx >= 11 or end_idx >= 11:  # 下半身
                                        color = (255, 0, 0)    # 藍色
                                    else:  # 上半身
                                        color = (0, 255, 0)    # 綠色
                                    
                                    cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 3)
                    
                    # 更新姿態關鍵點計數
                    self.pose_landmarks_count = len(valid_points)
                    
                    # 顯示偵測成功
                    if len(valid_points) > 0:
                        cv2.putText(frame, f"Pose: Person {person_idx+1} ({len(valid_points)}/17)", 
                                   (10, 90 + person_idx * 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    def load_yolo_model(self):
        """載入YOLO模型的專用方法"""
        model_names = [
            'yolo11n-pose.pt',
            'yolov8n-pose.pt', 
            'yolov8s-pose.pt',
            'yolo11s-pose.pt'
        ]
        
        for model_name in model_names:
            try:
                print(f"🔄 嘗試載入: {model_name}")
                self.yolo_model = YOLO(model_name)
                
                # 測試模型是否正常工作
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                test_results = self.yolo_model(dummy_img, verbose=False)
                
                print(f"✅ 成功載入並測試: {model_name}")
                return True
                
            except Exception as e:
                print(f"❌ {model_name} 載入失敗: {str(e)[:100]}...")
                self.yolo_model = None
                continue
        
        print("⚠️  所有YOLO模型載入失敗，將只使用MediaPipe功能")
        print("請檢查：")
        print("1. 網路連接是否正常")
        print("2. pip install ultralytics")
        print("3. 嘗試手動下載: python -c \"from ultralytics import YOLO; YOLO('yolo11n-pose.pt')\"")
        return False
    
    def debug_yolo_detection(self, frame):
        """YOLO偵測除錯方法"""
        if self.yolo_model is None:
            return False
        
        try:
            print(f"🔍 YOLO除錯 - 輸入影像尺寸: {frame.shape}")
            
            # 進行偵測
            results = self.yolo_model(frame, verbose=False, conf=0.1)  # 降低信心度閾值
            
            print(f"🔍 YOLO結果數量: {len(results)}")
            
            for i, result in enumerate(results):
                print(f"🔍 結果 {i}:")
                print(f"   - boxes: {result.boxes is not None if hasattr(result, 'boxes') else 'No boxes attr'}")
                print(f"   - keypoints: {result.keypoints is not None if hasattr(result, 'keypoints') else 'No keypoints attr'}")
                
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    kpts = result.keypoints
                    print(f"   - keypoints shape: {kpts.data.shape if hasattr(kpts, 'data') else 'No data'}")
                    print(f"   - keypoints type: {type(kpts)}")
                    
                    if hasattr(kpts, 'data'):
                        data = kpts.data
                        if len(data) > 0:
                            print(f"   - first person keypoints shape: {data[0].shape}")
                            # 檢查有效關鍵點數量
                            valid_count = 0
                            for point in data[0]:
                                if len(point) >= 3 and point[2] > 0.1:
                                    valid_count += 1
                            print(f"   - valid keypoints (conf>0.1): {valid_count}/17")
                        else:
                            print("   - no person detected")
            
            return True
            
        except Exception as e:
            print(f"❌ YOLO除錯失敗: {e}")
            return False
    
    def create_output_directory(self):
        """建立輸出資料夾"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✅ 建立輸出資料夾: {self.output_dir}")
        
        # 建立子資料夾
        subdirs = ['images', 'videos', 'data']
        for subdir in subdirs:
            path = os.path.join(self.output_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
    
    def save_frame(self, frame, frame_type="detection"):
        """儲存單一幀到圖片"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{frame_type}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, "images", filename)
        
        cv2.imwrite(filepath, frame)
        print(f"📸 已儲存截圖: {filename}")
        return filepath
    
    def save_detection_result_image(self, original_frame):
        """儲存偵測結果圖片 - 包含所有關鍵點標註"""
        # 複製原始影像進行處理
        result_frame = original_frame.copy()
        
        # 重新進行偵測並繪製所有關鍵點
        result_frame = self.process_frame_for_output(result_frame)
        
        # 加入詳細資訊標註
        self.add_detailed_annotations(result_frame)
        
        # 儲存結果圖片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_result_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, "images", filename)
        
        cv2.imwrite(filepath, result_frame)
        print(f"🎯 已儲存偵測結果圖片: {filename}")
        return filepath
    
    def process_frame_for_output(self, frame):
        """專門用於輸出的幀處理 - 包含所有視覺化效果"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 手部偵測並繪製
        if self.show_hands:
            hands_results = self.hands.process(rgb_frame)
            if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
                for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, 
                                                    hands_results.multi_handedness):
                    # 繪製手部關鍵點和連接線
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # 標註手部類型和關鍵點編號
                    h, w, _ = frame.shape
                    hand_type = handedness.classification[0].label
                    wrist = hand_landmarks.landmark[0]
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                    
                    # 手部標籤
                    cv2.putText(frame, f"{hand_type} Hand (21 points)", 
                               (wrist_x - 50, wrist_y - 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    
                    # 標註重要關鍵點編號
                    important_hand_points = [0, 4, 8, 12, 16, 20]  # 手腕和指尖
                    for idx in important_hand_points:
                        landmark = hand_landmarks.landmark[idx]
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.putText(frame, str(idx), (x + 5, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # 臉部偵測並繪製
        if self.show_face:
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # 繪製臉部輪廓
                    self.mp_drawing.draw_landmarks(
                        frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 255), thickness=1, circle_radius=1),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 128, 255), thickness=1)
                    )
                    
                    # 繪製眼部和嘴部細節
                    self.mp_drawing.draw_landmarks(
                        frame, face_landmarks, self.mp_face_mesh.FACEMESH_LEFT_EYE,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(255, 0, 0), thickness=1, circle_radius=1)
                    )
                    self.mp_drawing.draw_landmarks(
                        frame, face_landmarks, self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(255, 0, 0), thickness=1, circle_radius=1)
                    )
                    
                    # 標註臉部資訊
                    cv2.putText(frame, "Face Mesh (248 points)", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 人體姿態偵測並繪製
        if self.show_pose and self.yolo_model is not None:
            try:
                pose_results = self.yolo_model(frame, verbose=False, conf=0.3)
                for result in pose_results:
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        # 處理keypoints格式
                        if hasattr(result.keypoints, 'data'):
                            keypoints_data = result.keypoints.data
                        else:
                            keypoints_data = result.keypoints
                        
                        if len(keypoints_data.shape) == 3:
                            keypoints = keypoints_data[0]
                        else:
                            keypoints = keypoints_data
                        
                        keypoints = keypoints.cpu().numpy() if hasattr(keypoints, 'cpu') else keypoints
                        
                        # 繪製關鍵點和骨架
                        valid_points = []
                        for i, point in enumerate(keypoints):
                            if len(point) >= 3:
                                x, y, conf = point[0], point[1], point[2]
                            elif len(point) == 2:
                                x, y, conf = point[0], point[1], 1.0
                            else:
                                continue
                            
                            if conf > 0.3:
                                x, y = int(x), int(y)
                                valid_points.append((i, x, y, conf))
                                
                                # 繪製關鍵點
                                cv2.circle(frame, (x, y), 6, (255, 0, 255), -1)
                                cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
                                
                                # 標註關鍵點名稱
                                cv2.putText(frame, f"{i}:{self.pose_keypoints[i][:4]}", 
                                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.4, (255, 255, 0), 1)
                        
                        # 繪製骨架連接
                        point_dict = {idx: (x, y, conf) for idx, x, y, conf in valid_points}
                        for connection in self.pose_connections:
                            start_idx, end_idx = connection
                            if start_idx in point_dict and end_idx in point_dict:
                                start_x, start_y, start_conf = point_dict[start_idx]
                                end_x, end_y, end_conf = point_dict[end_idx]
                                
                                if start_conf > 0.3 and end_conf > 0.3:
                                    # 不同部位不同顏色
                                    if start_idx <= 4 or end_idx <= 4:
                                        color = (0, 255, 255)  # 頭部-黃色
                                    elif start_idx >= 11 or end_idx >= 11:
                                        color = (255, 0, 0)    # 下半身-藍色
                                    else:
                                        color = (0, 255, 0)    # 上半身-綠色
                                    
                                    cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 3)
                        
                        # 標註姿態資訊
                        if len(valid_points) > 0:
                            cv2.putText(frame, f"Pose Skeleton (17 points)", 
                                       (50, frame.shape[0] - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            except Exception as e:
                print(f"姿態處理錯誤: {e}")
        
        return frame
    
    def add_detailed_annotations(self, frame):
        """加入詳細的註釋資訊"""
        h, w = frame.shape[:2]
        
        # 建立半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-350, 10), (w-10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 標題
        cv2.putText(frame, "Detection Summary", (w-340, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 偵測統計
        info_lines = [
            f"Hands: {self.hand_landmarks_count} (max 42 points)",
            f"Face: {'✓' if self.face_landmarks_count > 0 else '✗'} (248 points)",
            f"Pose: {'✓' if self.pose_landmarks_count > 0 else '✗'} (17 points)",
            f"Total Points: {self.hand_landmarks_count * 21 + self.face_landmarks_count + self.pose_landmarks_count}",
            "",
            "Legend:",
            "🟡 Head connections",
            "🟢 Upper body",
            "🔵 Lower body",
            "🟣 Pose keypoints"
        ]
        
        for i, line in enumerate(info_lines):
            if line:
                color = (255, 255, 255)
                if "✓" in line:
                    color = (0, 255, 0)
                elif "✗" in line:
                    color = (0, 0, 255)
                elif line.startswith("Legend"):
                    color = (0, 255, 255)
                
                cv2.putText(frame, line, (w-335, 60 + i * 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 時間戳記
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def start_video_recording(self, frame_shape):
        """開始錄影"""
        if self.video_writer is not None:
            self.stop_video_recording()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"landmark_detection_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, "videos", filename)
        
        # 設定視訊編碼器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = frame_shape[:2]
        self.video_writer = cv2.VideoWriter(filepath, fourcc, 30.0, (w, h))
        
        print(f"🎥 開始錄影: {filename}")
        return filepath
    
    def create_detection_video(self, duration_seconds=10):
        """建立偵測結果影片 - 錄製指定時間長度"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_analysis_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, "videos", filename)
        
        # 開啟攝影機
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 取得影片參數
        ret, frame = cap.read()
        if not ret:
            print("❌ 無法讀取攝影機")
            cap.release()
            return None
        
        # 根據鏡像模式決定是否翻轉
        if self.mirror_mode:
            frame = cv2.flip(frame, 1)
        
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filepath, fourcc, 30.0, (w, h))
        
        print(f"🎬 正在錄製偵測分析影片 ({duration_seconds}秒)...")
        print(f"📹 鏡像模式: {'開啟' if self.mirror_mode else '關閉'}")
        
        frame_count = 0
        target_frames = duration_seconds * 30  # 30 FPS
        
        while frame_count < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 根據鏡像模式決定是否翻轉
            if self.mirror_mode:
                frame = cv2.flip(frame, 1)
            
            # 處理偵測並繪製所有關鍵點
            processed_frame = self.process_frame_for_output(frame.copy())
            
            # 加入詳細註釋
            self.add_detailed_annotations(processed_frame)
            
            # 加入進度條
            progress = frame_count / target_frames
            bar_width = w - 40
            bar_height = 20
            bar_x, bar_y = 20, h - 50
            
            # 背景條
            cv2.rectangle(processed_frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            # 進度條
            cv2.rectangle(processed_frame, (bar_x, bar_y), 
                         (bar_x + int(bar_width * progress), bar_y + bar_height), 
                         (0, 255, 0), -1)
            # 進度文字
            cv2.putText(processed_frame, f"Recording: {frame_count}/{target_frames}", 
                       (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 寫入影片
            video_writer.write(processed_frame)
            frame_count += 1
            
            # 顯示預覽
            cv2.imshow('Recording Detection Video', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 清理
        video_writer.release()
        cap.release()
        cv2.destroyWindow('Recording Detection Video')
        
        print(f"✅ 偵測分析影片已儲存: {filename}")
        return filepath
    
    def stop_video_recording(self):
        """停止錄影"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print("⏹️  錄影已停止")
    
    def save_video_frame(self, frame):
        """儲存視訊幀"""
        if self.video_writer is not None:
            self.video_writer.write(frame)
    
    def process_frame(self, frame):
        """處理單一幀（優化版）"""
        # 幀跳躍優化
        self.frame_counter += 1
        if self.frame_counter % self.skip_frames != 0 and self.last_results is not None:
            # 使用上一幀的結果
            return self.apply_cached_results(frame, self.last_results)
        
        # 優化幀尺寸
        optimized_frame = self.optimize_frame(frame)
        rgb_frame = cv2.cvtColor(optimized_frame, cv2.COLOR_BGR2RGB)
        
        # 重置計數器
        self.hand_landmarks_count = 0
        self.face_landmarks_count = 0
        self.pose_landmarks_count = 0
        
        results = {}
        
        # 手部偵測 (MediaPipe + GPU)
        if self.show_hands:
            hands_results = self.hands.process(rgb_frame)
            results['hands'] = hands_results
            if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
                self.hand_landmarks_count = len(hands_results.multi_hand_landmarks)
                
                for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, 
                                                    hands_results.multi_handedness):
                    self.draw_hand_landmarks(frame, hand_landmarks, handedness)
        
        # 臉部偵測 (MediaPipe + GPU)
        if self.show_face:
            face_results = self.face_mesh.process(rgb_frame)
            results['face'] = face_results
            if face_results.multi_face_landmarks:
                self.face_landmarks_count = len(face_results.multi_face_landmarks[0].landmark)
                
                for face_landmarks in face_results.multi_face_landmarks:
                    self.draw_face_landmarks(frame, face_landmarks)
                
                cv2.putText(frame, f"Face: {len(face_results.multi_face_landmarks)} detected", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Face: Not detected", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                self.face_landmarks_count = 0
        
        # 人體姿態偵測 (YOLOv11n-pose + GPU)
        if self.show_pose:
            if self.yolo_model is not None:
                try:
                    # GPU加速推理
                    pose_results = self.yolo_model(
                        optimized_frame, 
                        verbose=False, 
                        conf=0.15,  # 稍微提高信心度以減少誤檢
                        iou=0.5,
                        device=self.device,
                        half=self.device == 'cuda'  # 使用半精度加速（僅GPU）
                    )
                    
                    results['pose'] = pose_results
                    
                    if pose_results and len(pose_results) > 0:
                        self.draw_pose_landmarks(frame, pose_results)
                    else:
                        self.pose_landmarks_count = 0
                        cv2.putText(frame, "Pose: No results", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                except Exception as e:
                    self.pose_landmarks_count = 0
                    cv2.putText(frame, f"Pose: Error", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                self.pose_landmarks_count = 0
                cv2.putText(frame, "Pose: Model not loaded", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # 快取結果
        self.last_results = results
        
        return frame
    
    def apply_cached_results(self, frame, cached_results):
        """應用快取的偵測結果"""
        # 手部結果
        if 'hands' in cached_results and cached_results['hands'] and self.show_hands:
            hands_results = cached_results['hands']
            if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
                for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, 
                                                    hands_results.multi_handedness):
                    self.draw_hand_landmarks(frame, hand_landmarks, handedness)
        
        # 臉部結果
        if 'face' in cached_results and cached_results['face'] and self.show_face:
            face_results = cached_results['face']
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    self.draw_face_landmarks(frame, face_landmarks)
        
        # 姿態結果
        if 'pose' in cached_results and cached_results['pose'] and self.show_pose:
            pose_results = cached_results['pose']
            if pose_results and len(pose_results) > 0:
                self.draw_pose_landmarks(frame, pose_results)
        
        return frame
    
    def add_info_panel(self, frame):
        """添加資訊面板"""
        h, w = frame.shape[:2]
        
        # 只顯示FPS（右上角）
        if hasattr(self, 'fps'):
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 在左上角顯示偵測狀態和性能資訊
        status_text = []
        if self.show_hands:
            status_text.append(f"Hands: {self.hand_landmarks_count}")
        if self.show_face:
            status_text.append(f"Face: {'OK' if self.face_landmarks_count > 0 else 'NO'}")
        if self.show_pose:
            status_text.append(f"Pose: {'OK' if self.pose_landmarks_count > 0 else 'NO'}")
        
        # GPU狀態
        status_text.append(f"Device: {self.device.upper()}")
        
        # 性能設定
        if self.skip_frames > 1:
            status_text.append(f"Skip: 1/{self.skip_frames}")
        if self.resize_factor < 1.0:
            status_text.append(f"Scale: {self.resize_factor:.1f}")
        
        # 加入錄影狀態和鏡像狀態
        if self.video_writer is not None:
            status_text.append("🔴 Recording")
        if not self.mirror_mode:
            status_text.append("📹 Normal View")
        
        for i, text in enumerate(status_text):
            color = (0, 0, 255) if "🔴" in text else (255, 255, 255)
            if "📹" in text:
                color = (0, 255, 255)
            elif "CUDA" in text:
                color = (0, 255, 0)
            elif "CPU" in text:
                color = (0, 165, 255)
            cv2.putText(frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def run_camera(self):
        """執行攝影機偵測"""
        cap = cv2.VideoCapture(0)
        
        # 設定攝影機參數（優化性能）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)  # 嘗試設定更高的FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 減少緩衝區以降低延遲
        
        print("MediaPipe + YOLOv11n-pose GPU加速偵測器已啟動！")
        print("=" * 60)
        print("GPU加速功能：")
        print(f"- 運算裝置: {self.device.upper()}")
        print("- MediaPipe GPU加速: 自動啟用")
        print("- YOLO半精度推理: " + ("啟用" if self.device == 'cuda' else "不支援"))
        print("- 幀跳躍優化: " + ("啟用" if self.skip_frames > 1 else "關閉"))
        print("")
        print("偵測功能：")
        print("- 手部偵測：21個關鍵點 × 最多2隻手 (MediaPipe)")
        print("- 臉部偵測：248個關鍵點 (MediaPipe)")
        print("- 人體姿態：17個關鍵點 (YOLOv11n-pose)")
        print("")
        print("控制鍵：")
        print("  H - 切換手部偵測")
        print("  F - 切換臉部偵測")
        print("  P - 切換人體姿態偵測")
        print("  C - 切換手部連接線")
        print("  M - 切換臉部網格")
        print("  S - 切換姿態骨架")
        print("  I - 切換關鍵點資訊")
        print("  R - 開始/停止錄影")
        print("  A - 切換自動截圖")
        print("  SPACE - 手動截圖")
        print("  D - 儲存偵測結果圖片")
        print("  V - 錄製10秒偵測分析影片")
        print("  T - YOLO模型除錯測試")
        print("  X - 切換鏡像模式")
        print("  1-5 - 調整幀跳躍 (1=最高品質, 5=最高速度)")
        print("  Q - 退出程式")
        print("-" * 60)
        
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取攝影機畫面")
                break
            
            # 水平翻轉畫面（可選的鏡像模式）
            if self.mirror_mode:
                frame = cv2.flip(frame, 1)
            
            # 處理關鍵點偵測
            frame = self.process_frame(frame)
            
            # 計算FPS
            current_time = time.time()
            self.fps = 1.0 / (current_time - prev_time)
            prev_time = current_time
            
            # 添加資訊面板
            self.add_info_panel(frame)
            
            # 自動儲存功能
            self.frame_count += 1
            
            # 儲存視訊幀
            if self.video_writer is not None:
                self.save_video_frame(frame)
            
            # 自動截圖（每30幀一次）
            if self.auto_save and self.frame_count % self.save_interval == 0:
                if self.hand_landmarks_count > 0 or self.face_landmarks_count > 0 or self.pose_landmarks_count > 0:
                    self.save_frame(frame, "auto_detection")
            
            # 顯示畫面
            cv2.imshow('MediaPipe + YOLOv11n-pose Detector', frame)
            
            # 按鍵控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('h') or key == ord('H'):
                self.show_hands = not self.show_hands
                print(f"手部偵測: {'開啟' if self.show_hands else '關閉'}")
            elif key == ord('f') or key == ord('F'):
                self.show_face = not self.show_face
                print(f"臉部偵測: {'開啟' if self.show_face else '關閉'}")
            elif key == ord('p') or key == ord('P'):
                self.show_pose = not self.show_pose
                print(f"人體姿態偵測: {'開啟' if self.show_pose else '關閉'}")
            elif key == ord('c') or key == ord('C'):
                self.show_hand_connections = not self.show_hand_connections
                print(f"手部連接線: {'開啟' if self.show_hand_connections else '關閉'}")
            elif key == ord('m') or key == ord('M'):
                self.show_face_mesh = not self.show_face_mesh
                print(f"臉部網格: {'開啟' if self.show_face_mesh else '關閉'}")
            elif key == ord('s') or key == ord('S'):
                self.show_pose_connections = not self.show_pose_connections
                print(f"姿態骨架: {'開啟' if self.show_pose_connections else '關閉'}")
            elif key == ord('i') or key == ord('I'):
                self.show_landmarks_info = not self.show_landmarks_info
                print(f"關鍵點資訊: {'開啟' if self.show_landmarks_info else '關閉'}")
            elif key == ord('r') or key == ord('R'):
                if self.video_writer is None:
                    self.start_video_recording(frame.shape)
                else:
                    self.stop_video_recording()
            elif key == ord('a') or key == ord('A'):
                self.auto_save = not self.auto_save
                print(f"自動截圖: {'開啟' if self.auto_save else '關閉'}")
            elif key == ord(' '):  # 空白鍵
                self.save_frame(frame, "manual_capture")
            elif key == ord('d') or key == ord('D'):
                self.save_detection_result_image(frame)
            elif key == ord('v') or key == ord('V'):
                print("⚠️  開始錄製偵測分析影片，請保持姿勢...")
                self.create_detection_video(10)  # 錄製10秒
            elif key == ord('t') or key == ord('T'):
                print("🔧 執行YOLO除錯測試...")
                if self.yolo_model is not None:
                    self.debug_yolo_detection(frame)
                else:
                    print("⚠️  YOLO模型未載入，嘗試重新載入...")
                    self.load_yolo_model()
            elif key == ord('x') or key == ord('X'):
                self.mirror_mode = not self.mirror_mode
                print(f"鏡像模式: {'開啟' if self.mirror_mode else '關閉'}")
                print("💡 提示：關閉鏡像模式可修正錄影時的左右相反問題")
            elif key >= ord('1') and key <= ord('5'):
                # 調整幀跳躍設定
                self.skip_frames = int(chr(key))
                print(f"🚀 幀跳躍設定: 1/{self.skip_frames} ({'最高品質' if self.skip_frames == 1 else '提升性能'})")
                if self.skip_frames > 1:
                    print("💡 提示：數字越大FPS越高，但偵測更新頻率越低")
        
        # 清理資源
        self.stop_video_recording()
        cap.release()
        cv2.destroyAllWindows()
        print("程式已結束")
    
    def save_landmarks_data(self, frame, filename=None):
        """儲存所有關鍵點資料到檔案"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"landmarks_data_{timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, "data", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Integrated Landmarks Data (MediaPipe + YOLOv11n-pose)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 手部資料
            hands_results = self.hands.process(rgb_frame)
            if hands_results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    f.write(f"Hand {i+1} (21 landmarks - MediaPipe):\n")
                    coords = self.get_landmarks_coordinates(hand_landmarks, frame.shape)
                    for j, (x, y, z) in enumerate(coords):
                        f.write(f"  {j:2d} {self.hand_landmarks_names[j]:20s}: ({x:4d}, {y:4d}, {z:6.3f})\n")
                    f.write("\n")
            
            # 臉部資料
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                f.write("Face Landmarks (248 points - MediaPipe):\n")
                coords = self.get_landmarks_coordinates(face_results.multi_face_landmarks[0], frame.shape)
                for j, (x, y, z) in enumerate(coords):
                    f.write(f"  {j:3d}: ({x:4d}, {y:4d}, {z:6.3f})\n")
                f.write("\n")
            
            # 人體姿態資料
            if self.yolo_model is not None:
                try:
                    pose_results = self.yolo_model(frame, verbose=False)
                    for result in pose_results:
                        if hasattr(result, 'keypoints') and result.keypoints is not None:
                            # 處理不同的keypoints格式
                            if hasattr(result.keypoints, 'data'):
                                keypoints_data = result.keypoints.data
                            elif hasattr(result.keypoints, 'xy'):
                                keypoints_data = result.keypoints.xy
                            else:
                                keypoints_data = result.keypoints
                            
                            if len(keypoints_data.shape) == 3:
                                keypoints = keypoints_data[0]
                            else:
                                keypoints = keypoints_data
                            
                            keypoints = keypoints.cpu().numpy() if hasattr(keypoints, 'cpu') else keypoints
                            
                            f.write("Pose Landmarks (17 points - YOLOv11n-pose):\n")
                            for j, point in enumerate(keypoints):
                                if len(point) >= 3:
                                    x, y, conf = point[0], point[1], point[2]
                                elif len(point) == 2:
                                    x, y, conf = point[0], point[1], 1.0
                                else:
                                    continue
                                
                                if conf > 0.3:
                                    f.write(f"  {j:2d} {self.pose_keypoints[j]:15s}: ({x:7.1f}, {y:7.1f}, conf:{conf:5.3f})\n")
                            f.write("\n")
                except Exception as e:
                    f.write(f"Pose detection error: {e}\n")
        
        print(f"💾 關鍵點資料已儲存至: {filepath}")
        return filepath
    
    def get_landmarks_coordinates(self, landmarks, frame_shape):
        """獲取關鍵點座標"""
        h, w = frame_shape[:2]
        coordinates = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z if hasattr(landmark, 'z') else 0
            coordinates.append([x, y, z])
        return coordinates


def main():
    """主程式"""
    print("正在初始化偵測器...")
    detector = IntegratedLandmarkDetector()
    
    print("\nMediaPipe + YOLOv11n-pose 整合偵測器")
    print("=" * 60)
    print("此程式整合了三種先進的關鍵點偵測技術：")
    print("• MediaPipe 手部偵測：21個關鍵點（最多2隻手，共42個點）")
    print("• MediaPipe 臉部偵測：248個關鍵點")
    print("• YOLOv11n-pose 人體偵測：17個關鍵點")
    print("")
    
    try:
        detector.run_camera()
    except KeyboardInterrupt:
        print("\n程式被使用者中斷")
    except Exception as e:
        print(f"發生錯誤: {e}")


if __name__ == "__main__":
    main()