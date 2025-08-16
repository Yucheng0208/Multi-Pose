@echo off
echo ========================================
echo YOLOv11-pose + MediaPipe Face Mesh
echo 快速啟動腳本
echo ========================================
echo.

echo 選擇檢測模式:
echo 1. 骨架檢測 (Pose Detection)
echo 2. 臉部網格檢測 (Face Mesh Detection)
echo 3. 雙模組檢測 (Both Modes)
echo 4. 1080x1920 視窗臉部檢測
echo 5. 測試整合功能
echo 6. 查看幫助
echo.

set /p choice="請輸入選項 (1-6): "

if "%choice%"=="1" (
    echo 啟動骨架檢測...
    python main.py --mode pose --source 0 --visualize --enhanced-viz
) else if "%choice%"=="2" (
    echo 啟動臉部網格檢測 (預設 1920x1080 視窗)...
    python main.py --mode face --face-use-full-mesh --source 0 --visualize
) else if "%choice%"=="3" (
    echo 啟動雙模組檢測...
    python main.py --mode both --face-use-full-mesh --source 0 --visualize --enhanced-viz
) else if "%choice%"=="4" (
    echo 啟動 1080x1920 視窗臉部檢測...
    python main.py --mode face --face-use-full-mesh --source 0 --visualize --face-window-width 1080 --face-window-height 1920
) else if "%choice%"=="5" (
    echo 測試整合功能...
    python test_main_integration.py
) else if "%choice%"=="6" (
    echo 顯示幫助...
    python main.py --help
) else (
    echo 無效選項，請重新執行腳本
)

echo.
echo 按任意鍵退出...
pause >nul
