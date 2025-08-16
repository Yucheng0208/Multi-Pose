# 專案狀態報告

## 專案概述
本專案旨在建立一個完整的電腦視覺分析系統，整合多種 AI 模型進行人體姿態和臉部特徵檢測。

## 已完成模組

### 1. 骨架檢測模組 (pose_recorder.py) ✅
- **狀態**: 已完成並測試，已更新為統一輸出格式
- **功能**: 使用 YOLOv11-pose 進行人體骨架檢測
- **特點**: 
  - 支援 17 個 COCO 關鍵點
  - 整合 ByteTrack 追蹤器
  - 支援統一輸出格式：`<person_id><keypoints><coor_x><coor_y><confidence>`
  - 可作為獨立程式或模組使用
- **測試狀態**: 測試檔案已建立
- **文件**: API.md 和 SETUP.md 已建立

### 2. 臉部關鍵點檢測模組 (face_recorder.py) ✅
- **狀態**: 已完成並測試，已更新為統一輸出格式
- **功能**: 使用 MediaPipe Face Mesh 進行臉部關鍵點檢測
- **特點**:
  - 支援 468 個臉部關鍵點（完整版）
  - 提供 16 個簡化關鍵點（實用版）
  - 多臉部同時檢測支援
  - 統一輸出格式，支援信心度值
  - 支援即時視覺化和影片輸出
- **測試狀態**: 測試檔案已建立
- **文件**: FACE_DETECTION.md 已建立

### 3. 手部檢測模組 (hand_recorder.py) ✅
- **狀態**: 已完成並測試，已更新為統一輸出格式
- **功能**: 使用 MediaPipe Hand Landmarker 進行手部關鍵點檢測
- **特點**:
  - 支援 21 個手部關鍵點
  - 多手部同時檢測支援
  - 手部動作分類功能
  - 統一輸出格式，支援信心度值
- **測試狀態**: 測試檔案已建立

### 4. 統一輸出管理器 (unified_output_manager.py) ✅
- **狀態**: 新完成
- **功能**: 標準化所有模組的輸出格式
- **特點**:
  - 統一資料結構：`<person_id><keypoints><coor_x><coor_y><confidence>`
  - 支援多種輸出格式：npy、JSON、CSV、pickle
  - 自動生成元資料和摘要
  - 支援臉部、姿態、手部檢測結果整合
- **檔案格式**: `<modelname>_framenum.npy`
- **文件**: UNIFIED_OUTPUT_FORMAT.md 已建立

### 5. 多模態處理器 (multimodal_processor.py) ✅
- **狀態**: 已完成並更新為使用統一輸出格式
- **功能**: 整合臉部、姿態和手部檢測功能
- **特點**:
  - 並行處理多種模態
  - 自動使用統一輸出管理器
  - 支援會話管理和統計分析
  - 生成統一的npy檔案輸出

## 專案結構

```
yc/
├── src/
│   ├── __init__.py
│   ├── pose_recorder.py          # 骨架檢測模組（已更新為統一格式）
│   ├── face_recorder.py          # 臉部檢測模組（已更新為統一格式）
│   ├── hand_recorder.py          # 手部檢測模組（已更新為統一格式）
│   ├── unified_output_manager.py # 統一輸出管理器（新增）
│   ├── multimodal_processor.py   # 多模態處理器（已更新）
│   ├── multimodal_data.py        # 多模態資料整合器
│   ├── pose_visualizer.py        # 姿態視覺化器
│   └── body_analyzer.py          # 身體分析器
├── tests/
│   ├── __init__.py
│   ├── test_pose_recorder.py     # 骨架檢測測試
│   ├── test_face_recorder.py     # 臉部檢測測試
│   └── test_hand_recorder.py     # 手部檢測測試
├── examples/
│   ├── basic_usage.py            # 基本使用範例
│   ├── face_detection_example.py # 臉部檢測範例
│   ├── hand_recorder_example.py  # 手部檢測範例
│   └── multimodal_integration_example.py # 多模態整合範例
├── docs/
│   ├── API.md                    # API 文件
│   ├── SETUP.md                  # 安裝設定文件
│   ├── FACE_DETECTION.md         # 臉部檢測文件
│   └── UNIFIED_OUTPUT_FORMAT.md  # 統一輸出格式說明（新增）
├── config/
│   └── bytetrack.yaml            # 追蹤器設定
├── requirements.txt               # 依賴套件
├── test_unified_output.py        # 統一輸出格式測試腳本（新增）
├── run.bat                       # Windows 執行腳本
├── run.sh                        # Linux/Mac 執行腳本
├── run_face_detection.bat        # Windows 臉部檢測腳本
├── run_face_detection.sh         # Linux/Mac 臉部檢測腳本
└── PROJECT_STATUS.md             # 本文件
```

## 技術特點

### 統一輸出格式
- 所有模組使用統一的資料結構：`<person_id><keypoints><coor_x><coor_y><confidence>`
- 支援多種輸出格式：npy、JSON、CSV、pickle
- 每幀資料獨立保存為 `<modelname>_framenum.npy` 檔案
- 自動生成元資料和摘要資訊

### 資料結構一致性
- 臉部、姿態、手部檢測模組使用相同的輸出格式
- 支援信心度值（0.0-1.0）而非公分值
- 統一的座標系統和關鍵點命名規範

### 模組化設計
- 可獨立執行或作為副程式調用
- 清晰的 API 介面
- 完整的錯誤處理和日誌記錄

### 效能優化
- 支援 GPU 加速（CUDA/MPS）
- 可調整的檢測參數
- 支援跳幀處理和解析度調整

## 依賴套件

### 核心依賴
- **Python**: 3.10+
- **ultralytics**: >=8.2.0 (YOLOv11-pose)
- **mediapipe**: >=0.10.14 (臉部檢測)
- **opencv-python**: >=4.9 (影像處理)
- **numpy**: >=2.0 (數值運算)
- **pandas**: >=2.0 (資料處理)

### 選用依賴
- **pyarrow**: >=15.0 (Parquet 格式)
- **lapx**: >=0.0.3 (線性分配求解)

## 使用方式

### 骨架檢測
```bash
# 網路攝影機
python src/pose_recorder.py --source 0 --visualize

# 影片檔案
python src/pose_recorder.py --source video.mp4 --output pose_records.csv
```

### 臉部檢測
```bash
# 網路攝影機
python src/face_recorder.py --source 0 --visualize

# 影片檔案
python src/face_recorder.py --source video.mp4 --output face_records.csv
```

### 批次執行
```bash
# Windows
run_face_detection.bat

# Linux/Mac
./run_face_detection.sh
```

## 測試狀態

### 骨架檢測測試
- ✅ 模組初始化測試
- ✅ 關鍵點定義測試
- ✅ 資料結構一致性測試
- ✅ 輸出格式測試

### 臉部檢測測試
- ✅ 模組初始化測試
- ✅ 關鍵點定義測試
- ✅ 資料結構一致性測試
- ✅ 輸出格式測試
- ✅ 關鍵點映射測試

## 文件完整性

### 技術文件
- ✅ API 文件 (API.md)
- ✅ 安裝設定文件 (SETUP.md)
- ✅ 臉部檢測文件 (FACE_DETECTION.md)
- ✅ 專案狀態文件 (PROJECT_STATUS.md)

### 使用範例
- ✅ 基本使用範例 (basic_usage.py)
- ✅ 臉部檢測範例 (face_detection_example.py)
- ✅ 命令列參數說明
- ✅ 程式設計介面說明

## 待完成項目

### 短期目標
1. **整合測試**: 建立兩個模組的整合測試
2. **效能測試**: 測試不同硬體環境下的效能表現
3. **錯誤處理**: 完善邊界情況的錯誤處理

### 中期目標
1. **主程式整合**: 建立統一的 main.py 來協調兩個模組
2. **資料庫整合**: 支援資料庫儲存和查詢
3. **Web 介面**: 建立簡單的 Web 介面

### 長期目標
1. **多模態分析**: 整合語音、手勢等其他模態
2. **即時分析**: 支援串流資料的即時分析
3. **雲端部署**: 支援雲端服務部署

## 品質指標

### 程式碼品質
- **測試覆蓋率**: 目標 >80%
- **程式碼風格**: 遵循 PEP 8 標準
- **文件完整性**: 所有公開 API 都有文件說明
- **錯誤處理**: 完整的異常處理和日誌記錄

### 效能指標
- **處理速度**: 目標 30 FPS (1080p)
- **記憶體使用**: 目標 <2GB RAM
- **準確度**: 與標準資料集比較

## 貢獻指南

### 開發流程
1. Fork 專案
2. 建立功能分支
3. 實作功能並通過測試
4. 提交 Pull Request

### 程式碼標準
- 使用 Python 3.10+ 語法
- 遵循 PEP 8 程式碼風格
- 撰寫完整的 docstring
- 包含適當的測試案例

## 聯絡資訊

如有問題或建議，請：
1. 查看 GitHub Issues
2. 提交新的 Issue
3. 聯繫專案維護者

---

**最後更新**: 2024年12月
**版本**: 1.0.0
**狀態**: 開發中