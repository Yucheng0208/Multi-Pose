#!/usr/bin/env python3
"""
調試版本的轉換腳本
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import ast

def debug_parse_keypoints_data(keypoints_str):
    """調試版本的關鍵點解析"""
    print(f"原始資料類型: {type(keypoints_str)}")
    print(f"原始資料長度: {len(str(keypoints_str)) if keypoints_str else 0}")
    
    try:
        if pd.isna(keypoints_str) or keypoints_str == '[]':
            print("資料為空或NaN")
            return []
        
        # 嘗試直接解析JSON
        if isinstance(keypoints_str, str):
            print("資料是字串，嘗試解析...")
            # 處理可能的單引號問題
            keypoints_str = keypoints_str.replace("'", '"')
            data = json.loads(keypoints_str)
        else:
            print("資料不是字串，直接使用")
            data = keypoints_str
            
        print(f"解析後資料類型: {type(data)}")
        print(f"解析後資料長度: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
        
        if isinstance(data, list) and len(data) > 0:
            print(f"第一個元素類型: {type(data[0])}")
            if 'keypoints' in data[0]:
                print("找到keypoints欄位")
                return data[0]['keypoints']
            else:
                print("沒有keypoints欄位，直接返回")
                return data
        return []
    except Exception as e:
        print(f"解析錯誤: {e}")
        return []

def debug_convert_pose_data(csv_path, output_dir):
    """調試版本的姿勢資料轉換"""
    print(f"轉換姿勢資料: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV讀取成功，共 {len(df)} 行")
        print(f"欄位: {list(df.columns)}")
        
        # 檢查前幾行資料
        print("\n前3行資料:")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            print(f"行 {i}: frame_id={row['frame_id']}, detected={row['detected']}")
            if row['detected']:
                keypoints = debug_parse_keypoints_data(row['keypoints'])
                print(f"  關鍵點數量: {len(keypoints)}")
        
        current_date = datetime.now().strftime("%Y%m%d")
        
        for frame_id in df['frame_id'].unique():
            frame_data = df[df['frame_id'] == frame_id].iloc[0]
            
            if not frame_data['detected']:
                print(f"跳過frame {frame_id}: 未檢測到")
                continue
                
            print(f"\n處理frame {frame_id}...")
            keypoints = debug_parse_keypoints_data(frame_data['keypoints'])
            
            if not keypoints:
                print(f"  frame {frame_id}: 沒有關鍵點資料")
                continue
                
            print(f"  frame {frame_id}: 找到 {len(keypoints)} 個關鍵點")
            
            # 重新組織資料結構
            structured_data = []
            
            for person_idx, person_data in enumerate(keypoints):
                if isinstance(person_data, dict) and 'keypoints' in person_data:
                    person_keypoints = person_data['keypoints']
                else:
                    person_keypoints = person_data
                    
                print(f"    人物 {person_idx}: {len(person_keypoints)} 個關鍵點")
                
                for kp_idx, kp in enumerate(person_keypoints):
                    if isinstance(kp, dict):
                        structured_data.append({
                            'person_id': person_idx,
                            'model': 'pose',
                            'keypoints': kp_idx,
                            'coor_x': float(kp.get('x', 0)),
                            'coor_y': float(kp.get('y', 0)),
                            'confidence': float(kp.get('conf', 0))
                        })
            
            if structured_data:
                print(f"  frame {frame_id}: 結構化資料 {len(structured_data)} 個點")
                # 轉換為numpy陣列
                data_array = np.array([[
                    item['person_id'],
                    item['keypoints'],
                    item['coor_x'],
                    item['coor_y'],
                    item['confidence']
                ] for item in structured_data], dtype=np.float32)
                
                # 儲存為numpy檔案
                output_filename = f"pose_{current_date}_{frame_id:06d}.npy"
                output_path = os.path.join(output_dir, output_filename)
                np.save(output_path, data_array)
                print(f"  儲存: {output_filename} (shape: {data_array.shape})")
            else:
                print(f"  frame {frame_id}: 沒有結構化資料")
                
    except Exception as e:
        print(f"轉換姿勢資料時發生錯誤: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函數"""
    # 設定路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'output')
    new_output_dir = os.path.join(current_dir, 'output_structured')
    
    # 創建新的輸出目錄
    os.makedirs(new_output_dir, exist_ok=True)
    
    print("開始調試轉換output資料結構...")
    print(f"原始資料目錄: {output_dir}")
    print(f"新資料目錄: {new_output_dir}")
    
    # 檢查檔案是否存在
    pose_csv = os.path.join(output_dir, 'multimodal_session_pose.csv')
    
    if os.path.exists(pose_csv):
        print(f"找到姿勢CSV檔案: {pose_csv}")
        debug_convert_pose_data(pose_csv, new_output_dir)
    else:
        print(f"警告: 找不到姿勢資料CSV檔案: {pose_csv}")
    
    print("\n調試完成!")

if __name__ == "__main__":
    main()