#!/usr/bin/env python3
"""
診斷導入問題的腳本
"""

import sys
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def debug_imports():
    """診斷導入問題"""
    print("=" * 60)
    print("診斷導入問題")
    print("=" * 60)
    
    # 檢查 Python 路徑
    print(f"Python 路徑: {sys.path}")
    print(f"當前工作目錄: {sys.path[0]}")
    
    # 測試各模組導入
    modules_to_test = [
        ("src.hand_recorder", "HandRecorder"),
        ("src.multimodal_data", "MultimodalDataIntegrator"),
        ("src.multimodal_processor", "MultimodalProcessor"),
    ]
    
    for module_path, class_name in modules_to_test:
        print(f"\n--- 測試 {module_path} ---")
        
        try:
            # 嘗試導入模組
            module = __import__(module_path, fromlist=[class_name])
            print(f"✓ 模組 {module_path} 導入成功")
            
            # 嘗試獲取類別
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                print(f"✓ 類別 {class_name} 獲取成功")
                
                # 嘗試實例化
                try:
                    if class_name == "HandRecorder":
                        instance = cls(max_num_hands=1, model_complexity=0)
                        print(f"✓ {class_name} 實例化成功")
                        instance.close()
                    elif class_name == "MultimodalDataIntegrator":
                        instance = cls()
                        print(f"✓ {class_name} 實例化成功")
                    elif class_name == "MultimodalProcessor":
                        instance = cls(enable_face=False, enable_pose=False, enable_hand=True)
                        print(f"✓ {class_name} 實例化成功")
                        instance.close()
                except Exception as e:
                    print(f"❌ {class_name} 實例化失敗: {e}")
                    
            else:
                print(f"❌ 類別 {class_name} 不存在於模組中")
                
        except ImportError as e:
            print(f"❌ 模組 {module_path} 導入失敗: {e}")
        except Exception as e:
            print(f"❌ 其他錯誤: {e}")
    
    # 檢查文件是否存在
    import os
    files_to_check = [
        "src/__init__.py",
        "src/hand_recorder.py",
        "src/multimodal_data.py",
        "src/multimodal_processor.py"
    ]
    
    print(f"\n--- 檢查文件存在性 ---")
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path} 存在")
        else:
            print(f"❌ {file_path} 不存在")

def main():
    """主函數"""
    debug_imports()

if __name__ == "__main__":
    main()
