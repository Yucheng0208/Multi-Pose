#!/usr/bin/env python3
"""
專案設定檢查腳本
檢查所有必要的依賴和設定是否正確
"""

import sys
import importlib
import subprocess
from pathlib import Path

def check_python_version():
    """檢查 Python 版本"""
    print("檢查 Python 版本...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 版本過舊: {version.major}.{version.minor}")
        print("   需要 Python 3.10+")
        return False
    else:
        print(f"✅ Python 版本: {version.major}.{version.minor}.{version.micro}")
        return True

def check_dependencies():
    """檢查依賴套件"""
    print("\n檢查依賴套件...")
    
    required_packages = [
        "ultralytics",
        "opencv-python",
        "numpy",
        "pandas",
        "pyarrow"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "opencv-python":
                importlib.import_module("cv2")
            elif package == "pyarrow":
                importlib.import_module("pyarrow")
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安裝")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少套件: {', '.join(missing_packages)}")
        print("請執行: pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """檢查模型檔案"""
    print("\n檢查模型檔案...")
    
    model_files = [
        "yolo11n-pose.pt",
        "yolo11s-pose.pt",
        "yolo11m-pose.pt"
    ]
    
    found_models = []
    for model in model_files:
        if Path(model).exists():
            print(f"✅ {model}")
            found_models.append(model)
        else:
            print(f"❌ {model} - 未找到")
    
    if not found_models:
        print("\n未找到任何 YOLO 模型檔案")
        print("首次執行時會自動下載 yolo11n-pose.pt")
        print("或手動下載: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n-pose.pt")
    
    return len(found_models) > 0

def check_directories():
    """檢查目錄結構"""
    print("\n檢查目錄結構...")
    
    required_dirs = [
        "src",
        "config",
        "tests",
        "examples",
        "docs"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - 目錄不存在")
            all_exist = False
    
    return all_exist

def check_gpu_support():
    """檢查 GPU 支援"""
    print("\n檢查 GPU 支援...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA 版本: {torch.version.cuda}")
            return True
        else:
            print("ℹ️  CUDA 不可用，將使用 CPU 模式")
            return False
    except ImportError:
        print("ℹ️  PyTorch 未安裝，無法檢查 GPU 支援")
        return False

def main():
    """主檢查函式"""
    print("=" * 50)
    print("YOLOv11-pose 專案設定檢查")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_directories(),
        check_model_files(),
        check_gpu_support()
    ]
    
    print("\n" + "=" * 50)
    print("檢查結果摘要")
    print("=" * 50)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"✅ 所有檢查通過 ({passed}/{total})")
        print("\n專案已準備就緒！")
        print("執行方式:")
        print("  python main.py --source 0 --visualize true")
        print("  python main.py --source video.mp4 --output results.csv")
    else:
        print(f"❌ 部分檢查失敗 ({passed}/{total})")
        print("\n請解決上述問題後再執行專案")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)