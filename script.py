"""
MediaPipe + YOLOv11n-pose 整合偵測器
- 手部21個關鍵點 (MediaPipe)
- 臉部248個關鍵點 (MediaPipe)  
- 人體17個姿態關鍵點 (YOLOv11n-pose)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO

class IntegratedLandmarkDetector:
    def __init__(self):
        # 初始化MediaPipe解決方案
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # 初始化MediaPipe偵測器
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,  # 248個點
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 初始化YOLOv11n-pose
        try:
            print("正在載入YOLOv11n-pose模型...")
            # 使用官方模型名稱，會自動下載
            self.yolo_model = YOLO('yolo11n-pose.pt')  # 注意是yolo11n不是yolov11n
            print("✅ YOLOv11n-pose 模型載入成功")
        except Exception as e:
            print(f"❌ 載入YOLOv11n-pose失敗: {e}")
            print("請檢查以下事項：")
            print("1. 確認已安裝ultralytics: pip install ultralytics")
            print("2. 確認網路連接正常（首次使用需下載模型）")
            print("3. 或手動下載模型檔案到當前目錄")
            print("嘗試使用替代模型...")
            
            # 嘗試其他可能的模型名稱
            alternative_models = ['yolov8n-pose.pt', 'yolov5s.pt']
            self.yolo_model = None
            
            for model_name in alternative_models:
                try:
                    print(f"嘗試載入 {model_name}...")
                    self.yolo_model = YOLO(model_name)
                    print(f"✅ 成功載入替代模型: {model_name}")
                    break
                except Exception as e2:
                    print(f"❌ {model_name} 載入失敗: {e2}")
                    continue
            
            if self.yolo_model is None:
                print("⚠️  無法載入任何YOLO模型，將只使用MediaPipe功能")
        
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
        
        # 統計資訊
        self.hand_landmarks_count = 0
        self.face_landmarks_count = 0
        self.pose_landmarks_count = 0
        
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
                
                # 確保數據格式正確
                if len(keypoints_data.shape) == 3:
                    keypoints = keypoints_data[0]  # 取第一個人
                else:
                    keypoints = keypoints_data
                
                # 轉換為numpy array以便處理
                keypoints = keypoints.cpu().numpy() if hasattr(keypoints, 'cpu') else keypoints
                
                # 偵測並顯示關鍵點
                valid_points = []
                for i, point in enumerate(keypoints):
                    if len(point) >= 3:  # x, y, confidence
                        x, y, conf = point[0], point[1], point[2]
                    elif len(point) == 2:  # 只有x, y
                        x, y, conf = point[0], point[1], 1.0
                    else:
                        continue
                    
                    if conf > 0.3:  # 降低信心度閾值
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
                
                # 顯示偵測到的人數
                if len(valid_points) > 0:
                    cv2.putText(frame, f"Person detected", (10, frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    def process_frame(self, frame):
        """處理單一幀"""
        # 轉換BGR到RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 重置計數器
        self.hand_landmarks_count = 0
        self.face_landmarks_count = 0
        self.pose_landmarks_count = 0
        
        # 手部偵測 (MediaPipe)
        if self.show_hands:
            hands_results = self.hands.process(rgb_frame)
            if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
                self.hand_landmarks_count = len(hands_results.multi_hand_landmarks)
                
                for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, 
                                                    hands_results.multi_handedness):
                    self.draw_hand_landmarks(frame, hand_landmarks, handedness)
        
        # 臉部偵測 (MediaPipe - 248個點)
        if self.show_face:
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                self.face_landmarks_count = len(face_results.multi_face_landmarks[0].landmark)
                
                for face_landmarks in face_results.multi_face_landmarks:
                    self.draw_face_landmarks(frame, face_landmarks)
        
        # 人體姿態偵測 (YOLOv11n-pose)
        if self.show_pose and self.yolo_model is not None:
            try:
                # 使用YOLO進行推理，關閉詳細輸出
                pose_results = self.yolo_model(frame, verbose=False, conf=0.3)
                self.draw_pose_landmarks(frame, pose_results)
            except Exception as e:
                if not hasattr(self, 'yolo_error_shown'):
                    print(f"YOLO姿態偵測錯誤: {e}")
                    print("將關閉YOLO功能，只使用MediaPipe")
                    self.yolo_error_shown = True
                self.yolo_model = None
        
        return frame
    
    def add_info_panel(self, frame):
        """添加資訊面板"""
        h, w = frame.shape[:2]
        
        # 只顯示FPS（右上角）
        if hasattr(self, 'fps'):
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def run_camera(self):
        """執行攝影機偵測"""
        cap = cv2.VideoCapture("D:\\NTUT\\MultiPose\\source\\test.mp4")
        
        # 設定攝影機參數
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("MediaPipe + YOLOv11n-pose 整合偵測器已啟動！")
        print("=" * 60)
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
        print("  Q - 退出程式")
        print("-" * 60)
        
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取攝影機畫面")
                break
            
            # 水平翻轉畫面（鏡像效果）
            frame = cv2.flip(frame, 1)
            
            # 處理關鍵點偵測
            frame = self.process_frame(frame)
            
            # 計算FPS
            current_time = time.time()
            self.fps = 1.0 / (current_time - prev_time)
            prev_time = current_time
            
            # 添加資訊面板
            self.add_info_panel(frame)
            
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
        
        # 清理資源
        cap.release()
        cv2.destroyAllWindows()
        print("程式已結束")
    
    def save_landmarks_data(self, frame, filename="integrated_landmarks_data.txt"):
        """儲存所有關鍵點資料到檔案"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Integrated Landmarks Data (MediaPipe + YOLOv11n-pose)\n")
            f.write("=" * 60 + "\n\n")
            
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
                        if result.keypoints is not None:
                            keypoints = result.keypoints.data[0]
                            f.write("Pose Landmarks (17 points - YOLOv11n-pose):\n")
                            for j, (x, y, conf) in enumerate(keypoints):
                                if conf > 0.5:
                                    f.write(f"  {j:2d} {self.pose_keypoints[j]:15s}: ({x:7.1f}, {y:7.1f}, conf:{conf:5.3f})\n")
                            f.write("\n")
                except Exception as e:
                    f.write(f"Pose detection error: {e}\n")
        
        print(f"整合關鍵點資料已儲存至: {filename}")
    
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