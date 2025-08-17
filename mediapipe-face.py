import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from pathlib import Path
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class FaceLandmarkDetector:
    def __init__(self, model_path=None, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        初始化人臉關鍵點檢測器
        
        Args:
            model_path: 模型檔案路徑
            min_detection_confidence: 最小檢測置信度
            min_tracking_confidence: 最小追蹤置信度
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 設定模型路徑
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # 預設模型路徑
        if model_path is None:
            model_path = self.models_dir / "face_landmarker.task"
        
        self.model_path = Path(model_path)
        
        # 檢查模型檔案是否存在
        if not self.model_path.exists():
            print(f"警告: 找不到模型檔案 {self.model_path}")
            print("請確保 face_landmarker.task 檔案已放在 models 目錄中")
            print("您可以從以下連結下載:")
            print("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
            raise FileNotFoundError(f"模型檔案不存在: {self.model_path}")
        
        print(f"使用模型檔案: {self.model_path}")
        
        # 初始化 FaceLandmarker
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # 設定檢測選項
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=5,  # 最多檢測5張臉
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            
            self.detector = vision.FaceLandmarker.create_from_options(options)
            self.use_new_api = True
            print("使用 MediaPipe Tasks API")
            
        except ImportError:
            # 回退到舊版API
            print("使用 MediaPipe Solutions API (舊版)")
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            
            self.face_mesh_static = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.use_new_api = False
        
        # 定義人臉關鍵點的連接線
        self.face_connections = [
            # 臉部輪廓
            self.mp_face_mesh.FACEMESH_FACE_OVAL,
            # 左眼
            self.mp_face_mesh.FACEMESH_LEFT_EYE,
            # 右眼  
            self.mp_face_mesh.FACEMESH_RIGHT_EYE,
            # 左眉毛
            self.mp_face_mesh.FACEMESH_LEFT_EYEBROW,
            # 右眉毛
            self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
            # 嘴唇
            self.mp_face_mesh.FACEMESH_LIPS,
            # 鼻子
            # 可以添加更多連接線
        ]
    
    def draw_landmarks(self, image, detection_result=None, results=None):
        """
        在圖像上繪製人臉關鍵點
        
        Args:
            image: 輸入圖像
            detection_result: 新API的檢測結果
            results: 舊API的檢測結果
        
        Returns:
            處理後的圖像
        """
        annotated_image = image.copy()
        
        if self.use_new_api and detection_result:
            # 使用新API繪製 - 使用自訂繪製方法避免兼容性問題
            if detection_result.face_landmarks:
                for idx, face_landmarks in enumerate(detection_result.face_landmarks):
                    self._draw_face_landmarks_custom(annotated_image, face_landmarks, idx)
        
        elif not self.use_new_api and results:
            # 使用舊API繪製
            if results.multi_face_landmarks:
                for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    # 繪製臉部輪廓
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_FACE_OVAL,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                    )
                    
                    # 繪製眼部
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style()
                    )
                    
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style()
                    )
                    
                    # 繪製嘴部
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                    )
                    
                    # 添加人臉編號
                    h, w, _ = annotated_image.shape
                    nose_tip = face_landmarks.landmark[1]  # 鼻尖
                    cx, cy = int(nose_tip.x * w), int(nose_tip.y * h)
                    cv2.putText(annotated_image, f'Face {idx+1}', (cx-30, cy-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        elif not self.use_new_api and results:
            # 使用舊API繪製
            if results.multi_face_landmarks:
                for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    # 繪製臉部輪廓
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_FACE_OVAL,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                    )
                    
                    # 繪製眼部
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style()
                    )
                    
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style()
                    )
                    
                    # 繪製嘴部
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                    )
                    
                    # 添加人臉編號
                    h, w, _ = annotated_image.shape
                    nose_tip = face_landmarks.landmark[1]  # 鼻尖
                    cx, cy = int(nose_tip.x * w), int(nose_tip.y * h)
                    cv2.putText(annotated_image, f'Face {idx+1}', (cx-30, cy-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_image
    
    def _draw_face_landmarks_custom(self, image, face_landmarks, face_id):
        """
        自訂人臉關鍵點繪製方法（顯示所有468個點）
        
        Args:
            image: 圖像
            face_landmarks: 人臉關鍵點
            face_id: 人臉ID
        """
        h, w, _ = image.shape
        
        # 定義不同區域的顏色
        colors = {
            'face_oval': (0, 255, 0),      # 臉部輪廓 - 綠色
            'left_eye': (255, 0, 0),       # 左眼 - 藍色
            'right_eye': (255, 0, 0),      # 右眼 - 藍色
            'left_eyebrow': (0, 255, 255), # 左眉 - 黃色
            'right_eyebrow': (0, 255, 255),# 右眉 - 黃色
            'nose': (255, 255, 0),         # 鼻子 - 青色
            'lips_outer': (0, 0, 255),     # 外嘴唇 - 紅色
            'lips_inner': (128, 0, 255),   # 內嘴唇 - 紫色
            'face_mesh': (255, 255, 255)   # 其他面部點 - 白色
        }
        
        # 定義各區域的關鍵點索引
        regions = {
            'face_oval': [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ],
            'left_eye': [
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
                159, 160, 161, 246, 130, 25, 110, 24, 23, 22, 26, 112,
                243, 190, 56, 28, 27, 29, 30, 247, 31, 226, 35, 31, 228,
                229, 230, 231, 232, 233, 244, 245, 122, 6, 202, 214, 234
            ],
            'right_eye': [
                362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
                386, 385, 384, 398, 362, 398, 384, 385, 386, 387, 388, 466,
                263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385,
                359, 255, 339, 254, 253, 252, 256, 341, 463, 414, 286, 258,
                257, 259, 260, 467, 261, 446, 265, 261, 448, 449, 450, 451,
                452, 453, 464, 435, 410, 454
            ],
            'left_eyebrow': [
                46, 53, 52, 51, 48, 115, 131, 134, 102, 48, 64
            ],
            'right_eyebrow': [
                276, 283, 282, 281, 278, 344, 360, 363, 331, 278, 294
            ],
            'nose': [
                1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 
                115, 131, 134, 102, 49, 220, 305, 275, 273, 287, 269, 270,
                267, 271, 272, 358, 429, 420, 456, 248, 281, 5, 4, 6, 168,
                8, 9, 10, 151, 195, 197, 196, 3, 51, 48, 115, 131
            ],
            'lips_outer': [
                61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321,
                308, 324, 318, 402, 317, 14, 87, 178, 88, 95
            ],
            'lips_inner': [
                78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 269, 270,
                267, 271, 272, 12, 15, 16, 17, 18, 200
            ]
        }
        
        # 創建已使用索引的集合
        used_indices = set()
        for region_indices in regions.values():
            used_indices.update(region_indices)
        
        # 繪製各區域的關鍵點
        for region, indices in regions.items():
            color = colors[region]
            for idx in indices:
                if idx < len(face_landmarks):
                    landmark = face_landmarks[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    # 根據區域重要性調整點的大小
                    if region in ['left_eye', 'right_eye', 'lips_outer']:
                        radius = 3
                    elif region in ['face_oval', 'nose']:
                        radius = 2
                    else:
                        radius = 2
                    cv2.circle(image, (x, y), radius, color, -1)
        
        # 繪製剩餘的面部網格點（所有其他點）
        for idx, landmark in enumerate(face_landmarks):
            if idx not in used_indices:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(image, (x, y), 1, colors['face_mesh'], -1)
        
        # 標示特殊關鍵點（更大更明顯）
        special_points = {
            1: (0, 255, 255),    # 鼻尖 - 黃色
            33: (255, 100, 0),   # 左眼內角 - 橙色
            362: (255, 100, 0),  # 右眼內角 - 橙色
            61: (255, 0, 255),   # 左嘴角 - 洋紅
            291: (255, 0, 255),  # 右嘴角 - 洋紅
            13: (0, 255, 255),   # 上嘴唇中心 - 黃色
            14: (0, 255, 255),   # 下嘴唇中心 - 黃色
        }
        
        for idx, color in special_points.items():
            if idx < len(face_landmarks):
                landmark = face_landmarks[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(image, (x, y), 4, color, -1)
                # 添加小的黑色邊框讓點更明顯
                cv2.circle(image, (x, y), 4, (0, 0, 0), 1)
        
        # 添加人臉編號和關鍵點總數
        if len(face_landmarks) > 1:
            nose_tip = face_landmarks[1]
            cx, cy = int(nose_tip.x * w), int(nose_tip.y * h)
            # 人臉編號
            cv2.putText(image, f'Face {face_id+1}', (cx-40, cy-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 關鍵點數量
            cv2.putText(image, f'{len(face_landmarks)} points', (cx-50, cy-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def get_face_info(self, detection_result=None, results=None):
        """
        獲取人臉資訊
        
        Args:
            detection_result: 新API的檢測結果
            results: 舊API的檢測結果
            
        Returns:
            人臉資訊列表
        """
        face_info = []
        
        if self.use_new_api and detection_result:
            if detection_result.face_landmarks:
                for idx, face_landmarks in enumerate(detection_result.face_landmarks):
                    landmarks = []
                    for landmark in face_landmarks:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                    
                    info = {
                        'face_id': idx + 1,
                        'landmarks_count': len(landmarks),
                        'landmarks': landmarks
                    }
                    
                    # 添加表情資訊（如果有）
                    if detection_result.face_blendshapes and idx < len(detection_result.face_blendshapes):
                        blendshapes = detection_result.face_blendshapes[idx]
                        info['expressions'] = [
                            {'category': bs.category_name, 'score': bs.score}
                            for bs in blendshapes if bs.score > 0.1  # 只顯示分數大於0.1的表情
                        ]
                    
                    face_info.append(info)
        
        elif not self.use_new_api and results:
            if results.multi_face_landmarks:
                for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                    
                    face_info.append({
                        'face_id': idx + 1,
                        'landmarks_count': len(landmarks),
                        'landmarks': landmarks
                    })
        
        return face_info
    
    def process_image(self, image_path, output_path=None):
        """
        處理單張圖片
        
        Args:
            image_path: 輸入圖片路徑
            output_path: 輸出圖片路徑（可選）
        """
        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            print(f"無法讀取圖片: {image_path}")
            return
        
        # 轉換顏色空間
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.use_new_api:
            # 使用新API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            detection_result = self.detector.detect(mp_image)
            annotated_image = self.draw_landmarks(image, detection_result=detection_result)
            face_info = self.get_face_info(detection_result=detection_result)
        else:
            # 使用舊API
            results = self.face_mesh_static.process(rgb_image)
            annotated_image = self.draw_landmarks(image, results=results)
            face_info = self.get_face_info(results=results)
        
        # 顯示結果
        cv2.imshow('Face Landmarks - Image', annotated_image)
        
        # 儲存結果
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"結果已儲存至: {output_path}")
        
        # 顯示人臉資訊
        if face_info:
            print(f"檢測到 {len(face_info)} 張人臉:")
            for info in face_info:
                print(f"  人臉 {info['face_id']}: {info['landmarks_count']} 個關鍵點")
                if 'expressions' in info:
                    print(f"    主要表情: {info['expressions'][:3]}")  # 顯示前3個表情
        else:
            print("未檢測到人臉")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def process_video(self, video_path, output_path=None):
        """
        處理影片檔案
        
        Args:
            video_path: 輸入影片路徑
            output_path: 輸出影片路徑（可選）
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法開啟影片: {video_path}")
            return
        
        # 獲取影片資訊
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 設定影片寫入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if output_path:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("按 'q' 退出影片播放")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 轉換顏色空間
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.use_new_api:
                # 使用新API
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect(mp_image)
                annotated_frame = self.draw_landmarks(frame, detection_result=detection_result)
            else:
                # 使用舊API
                results = self.face_mesh.process(rgb_frame)
                annotated_frame = self.draw_landmarks(frame, results=results)
            
            # 顯示結果
            cv2.imshow('Face Landmarks - Video', annotated_frame)
            
            # 儲存影格
            if out is not None:
                out.write(annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if out is not None:
            out.release()
            print(f"處理後的影片已儲存至: {output_path}")
        
        cv2.destroyAllWindows()
    
    def process_webcam(self, camera_id=0):
        """
        處理網路攝影機即時影像
        
        Args:
            camera_id: 攝影機ID（預設0）
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"無法開啟攝影機 {camera_id}")
            return
        
        print("按 'q' 退出即時影像，按 's' 截圖")
        
        screenshot_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("無法從攝影機讀取影格")
                break
            
            # 水平翻轉（鏡像效果）
            frame = cv2.flip(frame, 1)
            
            # 轉換顏色空間
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.use_new_api:
                # 使用新API
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect(mp_image)
                annotated_frame = self.draw_landmarks(frame, detection_result=detection_result)
                
                # 顯示人臉數量
                face_count = len(detection_result.face_landmarks) if detection_result.face_landmarks else 0
            else:
                # 使用舊API
                results = self.face_mesh.process(rgb_frame)
                annotated_frame = self.draw_landmarks(frame, results=results)
                
                # 顯示人臉數量
                face_count = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
            
            cv2.putText(annotated_frame, f'Faces detected: {face_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 顯示操作提示
            cv2.putText(annotated_frame, "Press 'q' to quit, 's' to screenshot", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 顯示結果
            cv2.imshow('Face Landmarks - Webcam', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                screenshot_name = f'face_screenshot_{screenshot_count:03d}.jpg'
                cv2.imwrite(screenshot_name, annotated_frame)
                print(f"截圖已儲存: {screenshot_name}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def get_model_info(self):
        """
        取得模型資訊
        """
        info = {
            'model_path': str(self.model_path),
            'model_exists': self.model_path.exists(),
            'api_version': 'MediaPipe Tasks API' if self.use_new_api else 'MediaPipe Solutions API',
            'models_directory': str(self.models_dir.absolute())
        }
        
        if self.model_path.exists():
            info['model_size'] = f"{self.model_path.stat().st_size / (1024*1024):.2f} MB"
        
        return info

def main():
    parser = argparse.ArgumentParser(description='MediaPipe 人臉關鍵點檢測程式')
    parser.add_argument('--mode', type=str, choices=['webcam', 'image', 'video', 'info'], 
                       default='webcam', help='處理模式')
    parser.add_argument('--input', type=str, help='輸入檔案路徑（圖片或影片）')
    parser.add_argument('--output', type=str, help='輸出檔案路徑（可選）')
    parser.add_argument('--camera', type=int, default=0, help='攝影機ID（預設0）')
    parser.add_argument('--confidence', type=float, default=0.5, help='檢測置信度閾值')
    parser.add_argument('--model', type=str, help='模型檔案路徑（預設使用 models/face_landmarker.task）')
    
    args = parser.parse_args()
    
    try:
        # 初始化檢測器
        detector = FaceLandmarkDetector(
            model_path=args.model,
            min_detection_confidence=args.confidence
        )
        
        if args.mode == 'info':
            print("模型資訊:")
            print("=" * 50)
            info = detector.get_model_info()
            print(f"模型路徑: {info['model_path']}")
            print(f"模型存在: {info['model_exists']}")
            print(f"API版本: {info['api_version']}")
            print(f"模型目錄: {info['models_directory']}")
            if 'model_size' in info:
                print(f"模型大小: {info['model_size']}")
            return
        
        elif args.mode == 'webcam':
            print("啟動網路攝影機模式...")
            detector.process_webcam(args.camera)
        
        elif args.mode == 'image':
            if not args.input:
                print("請提供圖片路徑 --input")
                return
            
            if not os.path.exists(args.input):
                print(f"檔案不存在: {args.input}")
                return
            
            print(f"處理圖片: {args.input}")
            detector.process_image(args.input, args.output)
        
        elif args.mode == 'video':
            if not args.input:
                print("請提供影片路徑 --input")
                return
            
            if not os.path.exists(args.input):
                print(f"檔案不存在: {args.input}")
                return
            
            print(f"處理影片: {args.input}")
            detector.process_video(args.input, args.output)
    
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        print("\n請下載 face_landmarker.task 模型檔案:")
        print("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        print("並將其放在 models 目錄中")

if __name__ == "__main__":
    # 直接執行範例（不使用命令列參數時）
    if len(os.sys.argv) == 1:
        print("MediaPipe 人臉關鍵點檢測程式")
        print("=" * 40)
        print("1. 網路攝影機即時檢測")
        print("2. 圖片檢測")
        print("3. 影片檢測")
        print("4. 顯示模型資訊")
        
        choice = input("\n請選擇模式 (1-4): ").strip()
        
        try:
            detector = FaceLandmarkDetector()
            
            if choice == '1':
                print("\n啟動網路攝影機模式...")
                detector.process_webcam()
            
            elif choice == '2':
                image_path = input("請輸入圖片路徑: ").strip()
                if os.path.exists(image_path):
                    output_path = input("輸出路徑（可選，直接按Enter跳過）: ").strip()
                    output_path = output_path if output_path else None
                    detector.process_image(image_path, output_path)
                else:
                    print("檔案不存在!")
            
            elif choice == '3':
                video_path = input("請輸入影片路徑: ").strip()
                if os.path.exists(video_path):
                    output_path = input("輸出路徑（可選，直接按Enter跳過）: ").strip()
                    output_path = output_path if output_path else None
                    detector.process_video(video_path, output_path)
                else:
                    print("檔案不存在!")
            
            elif choice == '4':
                print("\n模型資訊:")
                print("=" * 50)
                info = detector.get_model_info()
                print(f"模型路徑: {info['model_path']}")
                print(f"模型存在: {info['model_exists']}")
                print(f"API版本: {info['api_version']}")
                print(f"模型目錄: {info['models_directory']}")
                if 'model_size' in info:
                    print(f"模型大小: {info['model_size']}")
            
            else:
                print("無效選擇!")
        
        except FileNotFoundError as e:
            print(f"錯誤: {e}")
            print("\n請下載 face_landmarker.task 模型檔案:")
            print("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
            print("並將其放在 models 目錄中")
    
    else:
        main()