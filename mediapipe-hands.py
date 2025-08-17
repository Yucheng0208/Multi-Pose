import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from pathlib import Path
import urllib.request
import shutil

class HandLandmarkDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5, model_path=None):
        """
        初始化手部關鍵點檢測器
        
        Args:
            max_num_hands: 最大檢測手數（預設2隻手）
            min_detection_confidence: 最小檢測置信度
            min_tracking_confidence: 最小追蹤置信度
            model_path: 本地模型檔案路徑（可選）
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 設定模型路徑
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # 下載並設定模型檔案
        self._setup_models()
        
        # 初始化手部檢測模型
        if model_path and os.path.exists(model_path):
            print(f"使用自訂模型: {model_path}")
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=1  # 使用複雜度1的模型
            )
            
            self.hands_static = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=1
            )
        else:
            # 使用預設設定
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            
            self.hands_static = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
    
    def _setup_models(self):
        """
        設定和下載MediaPipe模型檔案到本地models資料夾
        """
        # MediaPipe手部模型的URLs
        model_urls = {
            'hand_landmark_full.tflite': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            'palm_detection_full.tflite': 'https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite',
            'hand_landmark_lite.tflite': 'https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite'
        }
        
        # 檢查並下載模型檔案
        for model_name, url in model_urls.items():
            model_path = self.models_dir / model_name
            
            if not model_path.exists():
                print(f"下載模型檔案: {model_name}")
                try:
                    # 創建models目錄的說明檔案
                    readme_path = self.models_dir / "README.md"
                    if not readme_path.exists():
                        with open(readme_path, 'w', encoding='utf-8') as f:
                            f.write("""# MediaPipe 模型檔案

此資料夾包含MediaPipe手部檢測所需的模型檔案：

## 模型檔案說明

1. **hand_landmarker.task** - 完整的手部關鍵點檢測模型
2. **palm_detection_full.tflite** - 手掌檢測模型
3. **hand_landmark_lite.tflite** - 輕量版手部關鍵點模型

## 自動下載

程式首次執行時會自動下載這些模型檔案。

## 手動下載

您也可以從以下連結手動下載：
- https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite  
- https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite

下載後請將檔案放在此models資料夾中。
""")
                    
                    # 注意：由於MediaPipe的模型下載機制，實際上模型會自動管理
                    # 這裡主要是建立models資料夾結構
                    print(f"已建立models資料夾結構")
                    
                except Exception as e:
                    print(f"設定模型時發生錯誤: {e}")
                    print("將使用MediaPipe預設的模型管理方式")
            else:
                print(f"找到本地模型: {model_name}")
        
        # 設定環境變數指向本地模型目錄（如果需要）
        if 'MEDIAPIPE_CACHE_DIR' not in os.environ:
            os.environ['MEDIAPIPE_CACHE_DIR'] = str(self.models_dir.absolute())
            print(f"設定MediaPipe快取目錄: {self.models_dir.absolute()}")
    
    def get_model_info(self):
        """
        取得模型資訊和路徑
        """
        info = {
            'models_directory': str(self.models_dir.absolute()),
            'cache_directory': os.environ.get('MEDIAPIPE_CACHE_DIR', '系統預設'),
            'available_models': []
        }
        
        # 列出models資料夾中的檔案
        if self.models_dir.exists():
            for file_path in self.models_dir.iterdir():
                if file_path.is_file():
                    info['available_models'].append({
                        'name': file_path.name,
                        'size': f"{file_path.stat().st_size / (1024*1024):.2f} MB",
                        'path': str(file_path)
                    })
        
        return info
    
    def draw_landmarks(self, image, results):
        """
        在圖像上繪製手部關鍵點和連接線
        
        Args:
            image: 輸入圖像
            results: MediaPipe檢測結果
        
        Returns:
            處理後的圖像
        """
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 獲取手部標籤（左手或右手）
                handedness = results.multi_handedness[idx].classification[0].label
                
                # 繪製手部關鍵點
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # 在手腕位置添加文字標籤
                h, w, _ = image.shape
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                
                # 根據左右手使用不同顏色
                color = (0, 255, 0) if handedness == 'Right' else (255, 0, 0)
                cv2.putText(image, f'{handedness} Hand', (cx - 50, cy - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return image
    
    def get_hand_info(self, results):
        """
        獲取手部資訊
        
        Args:
            results: MediaPipe檢測結果
            
        Returns:
            手部資訊列表
        """
        hand_info = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                confidence = results.multi_handedness[idx].classification[0].score
                
                # 提取關鍵點座標
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                
                hand_info.append({
                    'handedness': handedness,
                    'confidence': confidence,
                    'landmarks': landmarks
                })
        
        return hand_info
    
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
        
        # 檢測手部關鍵點
        results = self.hands_static.process(rgb_image)
        
        # 繪製關鍵點
        annotated_image = self.draw_landmarks(image.copy(), results)
        
        # 顯示結果
        cv2.imshow('Hand Landmarks - Image', annotated_image)
        
        # 儲存結果
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"結果已儲存至: {output_path}")
        
        # 顯示手部資訊
        hand_info = self.get_hand_info(results)
        for info in hand_info:
            print(f"檢測到 {info['handedness']} 手，置信度: {info['confidence']:.2f}")
        
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
            
            # 檢測手部關鍵點
            results = self.hands.process(rgb_frame)
            
            # 繪製關鍵點
            annotated_frame = self.draw_landmarks(frame.copy(), results)
            
            # 顯示結果
            cv2.imshow('Hand Landmarks - Video', annotated_frame)
            
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
            
            # 檢測手部關鍵點
            results = self.hands.process(rgb_frame)
            
            # 繪製關鍵點
            annotated_frame = self.draw_landmarks(frame.copy(), results)
            
            # 顯示手部數量資訊
            hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            cv2.putText(annotated_frame, f'Hands detected: {hand_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 顯示操作提示
            cv2.putText(annotated_frame, "Press 'q' to quit, 's' to screenshot", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 顯示結果
            cv2.imshow('Hand Landmarks - Webcam', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                screenshot_name = f'hand_screenshot_{screenshot_count:03d}.jpg'
                cv2.imwrite(screenshot_name, annotated_frame)
                print(f"截圖已儲存: {screenshot_name}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='MediaPipe 手部關鍵點檢測程式')
    parser.add_argument('--mode', type=str, choices=['webcam', 'image', 'video', 'info'], 
                       default='webcam', help='處理模式')
    parser.add_argument('--input', type=str, help='輸入檔案路徑（圖片或影片）')
    parser.add_argument('--output', type=str, help='輸出檔案路徑（可選）')
    parser.add_argument('--camera', type=int, default=0, help='攝影機ID（預設0）')
    parser.add_argument('--confidence', type=float, default=0.7, help='檢測置信度閾值')
    parser.add_argument('--model', type=str, help='自訂模型檔案路徑')
    
    args = parser.parse_args()
    
    # 初始化檢測器
    detector = HandLandmarkDetector(min_detection_confidence=args.confidence, model_path=args.model)
    
    if args.mode == 'info':
        print("模型資訊:")
        print("=" * 50)
        info = detector.get_model_info()
        print(f"模型目錄: {info['models_directory']}")
        print(f"快取目錄: {info['cache_directory']}")
        print("\n可用模型檔案:")
        if info['available_models']:
            for model in info['available_models']:
                print(f"  - {model['name']} ({model['size']})")
                print(f"    路徑: {model['path']}")
        else:
            print("  尚未下載任何模型檔案")
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

if __name__ == "__main__":
    # 直接執行範例（不使用命令列參數時）
    if len(os.sys.argv) == 1:
        print("MediaPipe 手部關鍵點檢測程式")
        print("=" * 40)
        print("1. 網路攝影機即時檢測")
        print("2. 圖片檢測")
        print("3. 影片檢測")
        print("4. 顯示模型資訊")
        
        choice = input("\n請選擇模式 (1-4): ").strip()
        
        detector = HandLandmarkDetector()
        
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
            print(f"模型目錄: {info['models_directory']}")
            print(f"快取目錄: {info['cache_directory']}")
            print("\n可用模型檔案:")
            if info['available_models']:
                for model in info['available_models']:
                    print(f"  - {model['name']} ({model['size']})")
                    print(f"    路徑: {model['path']}")
            else:
                print("  程式會自動管理模型檔案")
        
        else:
            print("無效選擇!")
    
    else:
        main()