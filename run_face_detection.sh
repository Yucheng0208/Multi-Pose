#!/bin/bash

# 臉部關鍵點檢測執行腳本
# 使用 MediaPipe 進行臉部檢測

echo "========================================"
echo "MediaPipe 臉部關鍵點檢測"
echo "========================================"
echo

# 檢查 Python 是否安裝
if ! command -v python3 &> /dev/null; then
    echo "錯誤：未找到 Python3，請先安裝 Python 3.10+"
    exit 1
fi

# 檢查必要套件
echo "檢查必要套件..."

if ! python3 -c "import mediapipe" &> /dev/null; then
    echo "安裝 MediaPipe..."
    pip3 install "mediapipe>=0.10.14"
fi

if ! python3 -c "import cv2" &> /dev/null; then
    echo "安裝 OpenCV..."
    pip3 install "opencv-python>=4.9"
fi

if ! python3 -c "import pandas" &> /dev/null; then
    echo "安裝 Pandas..."
    pip3 install "pandas>=2.0"
fi

echo "套件檢查完成！"
echo

# 顯示使用選項
echo "請選擇執行模式："
echo "1. 網路攝影機檢測（即時）"
echo "2. 影片檔案檢測"
echo "3. 單一影像檢測"
echo "4. 執行範例程式"
echo "5. 執行測試"
echo

read -p "請輸入選項 (1-5): " choice

case $choice in
    1)
        echo
        echo "啟動網路攝影機臉部檢測..."
        echo "按 ESC 鍵退出"
        python3 src/face_recorder.py --source 0 --output face_records.csv --visualize
        ;;
    2)
        echo
        read -p "請輸入影片檔案路徑: " video_path
        if [ -f "$video_path" ]; then
            python3 src/face_recorder.py --source "$video_path" --output face_records.csv --video-out overlay.mp4 --visualize
        else
            echo "錯誤：影片檔案不存在"
        fi
        ;;
    3)
        echo
        read -p "請輸入影像檔案路徑: " image_path
        if [ -f "$image_path" ]; then
            python3 src/face_recorder.py --source "$image_path" --output face_records.csv
        else
            echo "錯誤：影像檔案不存在"
        fi
        ;;
    4)
        echo
        echo "執行範例程式..."
        python3 examples/face_detection_example.py
        ;;
    5)
        echo
        echo "執行測試..."
        python3 -m pytest tests/test_face_recorder.py -v
        ;;
    *)
        echo "無效選項"
        ;;
esac

echo
echo "執行完成！"
