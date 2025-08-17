# YOLOv11 完整檢測與儲存系統 - 含顯示畫面
# pip install ultralytics opencv-python pillow matplotlib torch

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import argparse
import torch

class YOLOv11DetectionSystem:
    def __init__(self, model_path='models/yolo11n-pose.pt', output_dir='detections', device='auto'):
        """
        初始化YOLOv11檢測系統
        model_path: 模型檔案路徑，例如 'models/yolo11n.pt' 或 'models/custom_model.pt'
        output_dir: 輸出目錄
        device: 運行設備 'cpu', 'gpu', 'cuda', 'auto'
        """
        # 模型路徑設定 - 在此修改你的模型路徑
        MODEL_PATH = model_path  # 修改這裡來指定你的模型檔案
        
        # 設備選擇
        self.device = self.setup_device(device)
        
        self.model = YOLO(MODEL_PATH)
        self.model.to(self.device)
        
        self.output_dir = Path(output_dir)
        self.setup_output_directories()
        
        print(f"已載入模型: {MODEL_PATH}")
        print(f"運行設備: {self.device}")
        print(f"輸出目錄: {self.output_dir.absolute()}")
    
    def setup_device(self, device):
        """設定運行設備"""
        if device == 'auto':
            if torch.cuda.is_available():
                selected_device = 'cuda'
                print(f"自動選擇: CUDA (GPU) - {torch.cuda.get_device_name()}")
            else:
                selected_device = 'cpu'
                print("自動選擇: CPU (CUDA不可用)")
        elif device in ['gpu', 'cuda']:
            if torch.cuda.is_available():
                selected_device = 'cuda'
                print(f"選擇: CUDA (GPU) - {torch.cuda.get_device_name()}")
            else:
                selected_device = 'cpu'
                print("警告: 請求GPU但CUDA不可用，改用CPU")
        else:
            selected_device = 'cpu'
            print("選擇: CPU")
        
        return selected_device
    
    def setup_output_directories(self):
        """建立輸出目錄結構"""
        directories = [
            'images',
            'videos', 
            'webcam',
            'results',
            'logs'
        ]
        
        for dir_name in directories:
            dir_path = self.output_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("輸出目錄結構已建立")
    
    def get_timestamp(self):
        """獲取時間戳"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_detection_info(self, results, source_path, detection_type):
        """儲存檢測資訊到JSON"""
        timestamp = self.get_timestamp()
        info = {
            'timestamp': timestamp,
            'source': str(source_path),
            'type': detection_type,
            'device': str(self.device),
            'model': str(self.model.model_name) if hasattr(self.model, 'model_name') else 'yolo11',
            'detections': []
        }
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    label = self.model.names[cls]
                    
                    detection = {
                        'class': label,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'center': [(x1+x2)/2, (y1+y2)/2],
                        'area': (x2-x1) * (y2-y1)
                    }
                    info['detections'].append(detection)
        
        # 儲存JSON檔案
        json_file = self.output_dir / 'logs' / f"{detection_type}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        return info
    
    def detect_image(self, image_path, conf_threshold=0.5, save_original=True):
        """
        檢測單張圖片並儲存結果，顯示圖片直到按下q鍵
        """
        print(f"正在處理圖片: {image_path}")
        
        # 執行檢測
        results = self.model(image_path, conf=conf_threshold, device=self.device)
        timestamp = self.get_timestamp()
        
        # 獲取原始圖片名稱
        image_name = Path(image_path).stem
        
        # 儲存檢測結果圖片
        result_image_path = self.output_dir / 'images' / f"{image_name}_detected_{timestamp}.jpg"
        annotated_frame = results[0].plot()
        cv2.imwrite(str(result_image_path), annotated_frame)
        
        # 儲存原始圖片副本（可選）
        if save_original:
            original_image_path = self.output_dir / 'images' / f"{image_name}_original_{timestamp}.jpg"
            original_image = cv2.imread(image_path)
            cv2.imwrite(str(original_image_path), original_image)
        
        # 儲存檢測資訊
        detection_info = self.save_detection_info(results, image_path, 'image')
        
        # 顯示檢測結果
        self.print_detection_results(detection_info)
        
        # 顯示圖片，等待按鍵
        print("顯示檢測結果，按 'q' 鍵結束")
        
        # 調整圖片大小以適應螢幕
        display_frame = self.resize_for_display(annotated_frame)
        
        cv2.imshow(f'圖片檢測結果 - {image_name}', display_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        print(f"圖片檢測結果已儲存: {result_image_path}")
        return results, detection_info
    
    def resize_for_display(self, frame, max_width=1200, max_height=800):
        """調整圖片大小以適應顯示"""
        height, width = frame.shape[:2]
        
        # 計算縮放比例
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # 不放大，只縮小
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        else:
            return frame
    
    def detect_video(self, video_path, conf_threshold=0.5, save_frames=False, frame_interval=30, show_video=True):
        """
        檢測影片並儲存結果，可選擇是否顯示
        save_frames: 是否儲存檢測框架
        frame_interval: 儲存框架的間隔
        show_video: 是否顯示影片
        """
        print(f"正在處理影片: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"錯誤: 無法開啟影片 {video_path}")
            return None, None
        
        # 獲取影片資訊
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"影片資訊: {width}x{height}, {fps}fps, {total_frames}框架")
        if show_video:
            print("顯示影片處理過程，按 'q' 提前結束，按 's' 儲存當前框架")
        
        # 設定輸出影片
        timestamp = self.get_timestamp()
        video_name = Path(video_path).stem
        output_video_path = self.output_dir / 'videos' / f"{video_name}_detected_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # 建立框架儲存目錄
        if save_frames:
            frames_dir = self.output_dir / 'videos' / f"{video_name}_frames_{timestamp}"
            frames_dir.mkdir(exist_ok=True)
        
        frame_count = 0
        all_detections = []
        manual_saved_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 執行檢測
            results = self.model(frame, conf=conf_threshold, device=self.device)
            
            # 繪製結果
            annotated_frame = results[0].plot()
            
            # 添加進度資訊
            progress_text = f"Frame: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)"
            cv2.putText(annotated_frame, progress_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 寫入輸出影片
            out.write(annotated_frame)
            
            # 顯示影片
            if show_video:
                display_frame = self.resize_for_display(annotated_frame)
                cv2.imshow(f'影片檢測 - {video_name}', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("使用者提前結束影片處理")
                    break
                elif key == ord('s'):
                    # 手動儲存當前框架
                    manual_frame_path = self.output_dir / 'videos' / f"{video_name}_manual_frame_{frame_count}_{timestamp}.jpg"
                    cv2.imwrite(str(manual_frame_path), annotated_frame)
                    print(f"手動儲存框架: {manual_frame_path}")
                    manual_saved_frames += 1
            
            # 儲存特定框架
            if save_frames and frame_count % frame_interval == 0:
                frame_path = frames_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), annotated_frame)
            
            # 收集檢測資訊
            if results[0].boxes is not None:
                frame_detections = {
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'detections': []
                }
                
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    label = self.model.names[cls]
                    
                    detection = {
                        'class': label,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    }
                    frame_detections['detections'].append(detection)
                
                all_detections.append(frame_detections)
            
            frame_count += 1
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"處理進度: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        cap.release()
        out.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # 儲存影片檢測資訊
        video_info = {
            'timestamp': timestamp,
            'source': str(video_path),
            'type': 'video',
            'device': str(self.device),
            'video_info': {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': frame_count,
                'duration': frame_count / fps
            },
            'manual_saved_frames': manual_saved_frames,
            'frame_detections': all_detections
        }
        
        json_file = self.output_dir / 'logs' / f"video_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(video_info, f, indent=2, ensure_ascii=False)
        
        print(f"影片檢測完成: {output_video_path}")
        print(f"處理了 {frame_count} 框架，檢測到 {len(all_detections)} 個包含物體的框架")
        if manual_saved_frames > 0:
            print(f"手動儲存了 {manual_saved_frames} 個框架")
        
        return output_video_path, video_info
    
    def detect_webcam(self, conf_threshold=0.5, camera_id=0, save_interval=10, max_duration=300):
        """
        即時網路攝影機檢測並儲存
        save_interval: 儲存間隔（秒）
        max_duration: 最大錄製時間（秒）
        """
        print(f"開始即時檢測 (攝影機 {camera_id})")
        print(f"運行設備: {self.device}")
        print(f"每 {save_interval} 秒自動儲存一次")
        print(f"最大錄製時間: {max_duration} 秒")
        print("按 's' 手動儲存當前框架")
        print("按 'r' 開始/停止錄製")
        print("按 'q' 退出")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"錯誤: 無法開啟攝影機 {camera_id}")
            return None
        
        # 獲取攝影機資訊
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"攝影機資訊: {width}x{height}, {fps}fps")
        
        # 錄製變數
        is_recording = False
        video_writer = None
        recording_start_time = None
        
        # 儲存變數
        last_save_time = time.time()
        session_timestamp = self.get_timestamp()
        frame_count = 0
        saved_frames = 0
        
        start_time = time.time()
        
        # 計算FPS
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("錯誤: 無法讀取攝影機框架")
                break
            
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # 檢查最大錄製時間
            if elapsed_time > max_duration:
                print(f"達到最大錄製時間 {max_duration} 秒，自動停止")
                break
            
            # 執行檢測
            results = self.model(frame, conf=conf_threshold, device=self.device)
            annotated_frame = results[0].plot()
            
            # 計算FPS
            fps_counter += 1
            if current_time - fps_start_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = current_time
            
            # 在框架上顯示資訊
            info_lines = [
                f"Device: {self.device}",
                f"FPS: {current_fps}",
                f"Time: {elapsed_time:.1f}s | Frames: {frame_count} | Saved: {saved_frames}",
            ]
            
            if is_recording:
                info_lines.append("Status: [RECORDING]")
            else:
                info_lines.append("Status: [LIVE]")
            
            for i, line in enumerate(info_lines):
                y_pos = 30 + i * 25
                cv2.putText(annotated_frame, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 顯示結果
            display_frame = self.resize_for_display(annotated_frame)
            cv2.imshow('YOLOv11 即時檢測', display_frame)
            
            # 自動儲存框架
            if current_time - last_save_time >= save_interval:
                frame_path = self.output_dir / 'webcam' / f"webcam_{session_timestamp}_frame_{saved_frames:04d}.jpg"
                cv2.imwrite(str(frame_path), annotated_frame)
                
                # 儲存檢測資訊
                detection_info = self.save_detection_info(results, f"webcam_frame_{saved_frames}", 'webcam')
                
                print(f"自動儲存框架: {frame_path}")
                last_save_time = current_time
                saved_frames += 1
            
            # 錄製影片
            if is_recording:
                if video_writer is None:
                    video_path = self.output_dir / 'webcam' / f"webcam_recording_{session_timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
                    recording_start_time = current_time
                
                video_writer.write(annotated_frame)
            
            # 處理按鍵
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 手動儲存
                manual_save_path = self.output_dir / 'webcam' / f"webcam_{session_timestamp}_manual_{int(current_time)}.jpg"
                cv2.imwrite(str(manual_save_path), annotated_frame)
                detection_info = self.save_detection_info(results, f"webcam_manual_save", 'webcam')
                print(f"手動儲存: {manual_save_path}")
                saved_frames += 1
            elif key == ord('r'):
                # 切換錄製狀態
                if is_recording:
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        recording_duration = current_time - recording_start_time
                        print(f"停止錄製，錄製時間: {recording_duration:.1f} 秒")
                    is_recording = False
                else:
                    is_recording = True
                    print("開始錄製...")
            
            frame_count += 1
        
        # 清理資源
        if video_writer is not None:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        
        session_info = {
            'session_timestamp': session_timestamp,
            'total_frames': frame_count,
            'saved_frames': saved_frames,
            'duration': elapsed_time,
            'camera_id': camera_id,
            'device': str(self.device)
        }
        
        print(f"即時檢測結束")
        print(f"總框架數: {frame_count}")
        print(f"儲存框架數: {saved_frames}")
        print(f"總時間: {elapsed_time:.1f} 秒")
        
        return session_info
    
    def batch_detect_images(self, image_folder, conf_threshold=0.5, show_images=True):
        """批量檢測圖片，可選擇是否顯示每張圖片"""
        image_folder = Path(image_folder)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_folder.glob(f'*{ext}'))
            image_files.extend(image_folder.glob(f'*{ext.upper()}'))
        
        print(f"找到 {len(image_files)} 張圖片")
        if show_images:
            print("將顯示每張圖片的檢測結果，按 'q' 進入下一張，按 'ESC' 跳過剩餘圖片")
        
        batch_results = []
        for i, image_path in enumerate(image_files):
            print(f"處理 ({i+1}/{len(image_files)}): {image_path.name}")
            
            if show_images:
                results, info = self.detect_image_batch_display(str(image_path), conf_threshold)
                if results is None:  # 使用者按ESC跳過
                    print("跳過剩餘圖片")
                    break
            else:
                results, info = self.detect_image_no_display(str(image_path), conf_threshold)
            
            batch_results.append(info)
        
        # 儲存批量處理結果
        batch_info = {
            'timestamp': self.get_timestamp(),
            'type': 'batch_images',
            'total_images': len(image_files),
            'processed_images': len(batch_results),
            'source_folder': str(image_folder),
            'device': str(self.device),
            'results': batch_results
        }
        
        json_file = self.output_dir / 'logs' / f"batch_images_{batch_info['timestamp']}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, indent=2, ensure_ascii=False)
        
        print(f"批量處理完成，結果已儲存: {json_file}")
        return batch_results
    
    def detect_image_batch_display(self, image_path, conf_threshold=0.5):
        """批量處理時的圖片檢測（含顯示）"""
        # 執行檢測
        results = self.model(image_path, conf=conf_threshold, device=self.device)
        timestamp = self.get_timestamp()
        
        # 獲取原始圖片名稱
        image_name = Path(image_path).stem
        
        # 儲存檢測結果圖片
        result_image_path = self.output_dir / 'images' / f"{image_name}_detected_{timestamp}.jpg"
        annotated_frame = results[0].plot()
        cv2.imwrite(str(result_image_path), annotated_frame)
        
        # 儲存檢測資訊
        detection_info = self.save_detection_info(results, image_path, 'image')
        
        # 顯示檢測結果
        self.print_detection_results(detection_info)
        
        # 顯示圖片
        display_frame = self.resize_for_display(annotated_frame)
        cv2.imshow(f'批量檢測 - {image_name}', display_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return results, detection_info
            elif key == 27:  # ESC鍵
                cv2.destroyAllWindows()
                return None, None
    
    def detect_image_no_display(self, image_path, conf_threshold=0.5):
        """批量處理時的圖片檢測（不顯示）"""
        # 執行檢測
        results = self.model(image_path, conf=conf_threshold, device=self.device)
        timestamp = self.get_timestamp()
        
        # 獲取原始圖片名稱
        image_name = Path(image_path).stem
        
        # 儲存檢測結果圖片
        result_image_path = self.output_dir / 'images' / f"{image_name}_detected_{timestamp}.jpg"
        annotated_frame = results[0].plot()
        cv2.imwrite(str(result_image_path), annotated_frame)
        
        # 儲存檢測資訊
        detection_info = self.save_detection_info(results, image_path, 'image')
        
        # 顯示檢測結果
        self.print_detection_results(detection_info)
        
        return results, detection_info
    
    def print_detection_results(self, detection_info):
        """印出檢測結果"""
        detections = detection_info['detections']
        if len(detections) == 0:
            print("未檢測到任何物體")
            return
        
        print(f"檢測到 {len(detections)} 個物體:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class']}: {det['confidence']:.2f}")
    
    def get_detection_statistics(self):
        """獲取檢測統計資訊"""
        logs_dir = self.output_dir / 'logs'
        json_files = list(logs_dir.glob('*.json'))
        
        stats = {
            'total_sessions': len(json_files),
            'image_sessions': 0,
            'video_sessions': 0,
            'webcam_sessions': 0,
            'total_detections': 0,
            'class_counts': {}
        }
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data['type'] == 'image':
                stats['image_sessions'] += 1
                for det in data['detections']:
                    stats['total_detections'] += 1
                    class_name = det['class']
                    stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
            
            elif data['type'] == 'video':
                stats['video_sessions'] += 1
                for frame_det in data.get('frame_detections', []):
                    for det in frame_det['detections']:
                        stats['total_detections'] += 1
                        class_name = det['class']
                        stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
            
            elif data['type'] == 'webcam':
                stats['webcam_sessions'] += 1
                for det in data['detections']:
                    stats['total_detections'] += 1
                    class_name = det['class']
                    stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='YOLOv11 檢測系統')
    parser.add_argument('--mode', choices=['image', 'video', 'webcam', 'batch'], 
                       required=True, help='檢測模式')
    parser.add_argument('--input', type=str, help='輸入檔案或目錄路徑')
    parser.add_argument('--model', type=str, default='models/yolo11n.pt', 
                       help='模型檔案路徑')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度閾值')
    parser.add_argument('--output', type=str, default='detections', help='輸出目錄')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'gpu', 'cuda'], 
                       default='auto', help='運行設備')
    parser.add_argument('--camera', type=int, default=0, help='攝影機ID')
    parser.add_argument('--save-interval', type=int, default=10, help='即時檢測儲存間隔（秒）')
    parser.add_argument('--max-duration', type=int, default=300, help='最大錄製時間（秒）')
    parser.add_argument('--no-display', action='store_true', help='不顯示檢測畫面')
    
    args = parser.parse_args()
    
    # 初始化檢測系統
    detector = YOLOv11DetectionSystem(model_path=args.model, output_dir=args.output, device=args.device)
    
    if args.mode == 'image':
        if not args.input:
            print("錯誤: 圖片模式需要指定 --input 參數")
            return
        detector.detect_image(args.input, args.conf)
    
    elif args.mode == 'video':
        if not args.input:
            print("錯誤: 影片模式需要指定 --input 參數")
            return
        detector.detect_video(args.input, args.conf, show_video=not args.no_display)
    
    elif args.mode == 'webcam':
        detector.detect_webcam(
            conf_threshold=args.conf,
            camera_id=args.camera,
            save_interval=args.save_interval,
            max_duration=args.max_duration
        )
    
    elif args.mode == 'batch':
        if not args.input:
            print("錯誤: 批量模式需要指定 --input 目錄")
            return
        detector.batch_detect_images(args.input, args.conf, show_images=not args.no_display)
    
    # 顯示統計資訊
    stats = detector.get_detection_statistics()
    print("\n檢測統計:")
    print(f"總會話數: {stats['total_sessions']}")
    print(f"總檢測數: {stats['total_detections']}")
    print("類別統計:")
    for class_name, count in sorted(stats['class_counts'].items()):
        print(f"  {class_name}: {count}")

if __name__ == "__main__":
    # 如果直接運行，提供互動式選單
    if len(os.sys.argv) == 1:
        print("YOLOv11 檢測系統")
        print("1. 圖片檢測")
        print("2. 影片檢測") 
        print("3. 即時攝影機檢測")
        print("4. 批量圖片檢測")
        print("5. 檢視統計資訊")
        
        choice = input("請選擇功能 (1-5): ")
        
        # 選擇運行設備
        print("\n選擇運行設備:")
        print("1. 自動選擇 (推薦)")
        print("2. 強制使用CPU")
        print("3. 強制使用GPU")
        
        device_choice = input("請選擇設備 (1-3): ")
        device_map = {'1': 'auto', '2': 'cpu', '3': 'gpu'}
        device = device_map.get(device_choice, 'auto')
        
        detector = YOLOv11DetectionSystem(model_path='models/yolo11n-pose.pt', device=device)
        
        if choice == '1':
            image_path = input("請輸入圖片路徑: ")
            detector.detect_image(image_path)
        
        elif choice == '2':
            video_path = input("請輸入影片路徑: ")
            show_choice = input("是否顯示影片處理過程? (y/n): ").lower()
            show_video = show_choice in ['y', 'yes', '是']
            detector.detect_video(video_path, show_video=show_video)
        
        elif choice == '3':
            camera_id = input("請輸入攝影機ID (預設0): ")
            camera_id = int(camera_id) if camera_id.isdigit() else 0
            detector.detect_webcam(camera_id=camera_id)
        
        elif choice == '4':
            folder_path = input("請輸入圖片資料夾路徑: ")
            show_choice = input("是否顯示每張圖片? (y/n): ").lower()
            show_images = show_choice in ['y', 'yes', '是']
            detector.batch_detect_images(folder_path, show_images=show_images)
        
        elif choice == '5':
            stats = detector.get_detection_statistics()
            print("\n檢測統計:")
            print(f"總會話數: {stats['total_sessions']}")
            print(f"總檢測數: {stats['total_detections']}")
            print("類別統計:")
            for class_name, count in sorted(stats['class_counts'].items()):
                print(f"  {class_name}: {count}")
        
        else:
            print("無效選擇")
    else:
        main()