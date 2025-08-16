# YOLOv11-pose 骨架偵測與 MediaPipe 臉部網格點檢測整合系統

## 概述

本專案是一個多模態人體分析系統，整合了三個主要的檢測模組：

1. **臉部檢測** (Face Detection) - 使用 MediaPipe Face Landmarker
2. **姿態檢測** (Pose Detection) - 使用 YOLOv11-pose
3. **手部檢測** (Hand Detection) - 使用 MediaPipe Hand Landmarker

## 新增模組

### 1. 手部檢測模組 (`src/hand_recorder.py`)

使用 MediaPipe Hand Landmarker 進行手部關鍵點檢測，支援：

- **21 個手部關鍵點**：手腕、拇指、食指、中指、無名指、小指的各個關節
- **手部動作識別**：拳頭、張開手掌、指向、比耶、拇指向上等
- **左右手區分**：自動識別左手和右手
- **3D 座標**：提供 x, y, z 三維座標資訊
- **信心度評估**：每個關鍵點的檢測信心度

### 2. 多模態資料整合模組 (`src/multimodal_data.py`)

提供統一的資料結構和整合介面：

- **FrameData**：單幀的多模態資料容器
- **MultimodalSession**：多模態會話管理
- **MultimodalDataIntegrator**：資料整合器
- **MultimodalAnalyzer**：多模態資料分析器

### 3. 多模態處理器 (`src/multimodal_processor.py`)

整合三個檢測模組的統一處理器：

- **並行處理**：使用多執行緒同時處理各模態
- **統一介面**：提供單一的 API 進行多模態檢測
- **會話管理**：支援多個檢測會話
- **即時處理**：支援攝影機即時檢測

## 主要功能

1. **手部動作識別**：自動識別 10+ 種手部動作
2. **多模態整合**：同時處理臉部、姿態和手部
3. **即時處理**：支援攝影機即時檢測
4. **資料分析**：提供統計報告和品質評估
5. **並行處理**：使用多執行緒提高效能

## 使用方式

### 基本使用

```bash
# 手部檢測
python main.py --mode hand --source 0 --visualize
python main.py --mode hand --source video.mp4 --output hand_results.csv

# 多模態整合檢測
python main.py --mode multimodal --source 0 --visualize
python main.py --mode multimodal --source video.mp4 --output multimodal_results.csv

# 臉部檢測
python main.py --mode face --source 0 --visualize
python main.py --mode face --source video.mp4 --output face_mesh_results.csv

# 姿態檢測
python main.py --mode pose --source 0 --visualize
python main.py --mode pose --source video.mp4 --output pose_results.csv
```

### 進階參數

#### 手部檢測參數

```bash
python main.py --mode hand \
    --source 0 \
    --hand-max-hands 2 \
    --hand-model-complexity 1 \
    --hand-detection-conf 0.5 \
    --hand-tracking-conf 0.5 \
    --visualize \
    --output hand_data.csv
```

#### 多模態整合參數

```bash
python main.py --mode multimodal \
    --source 0 \
    --enable-face \
    --enable-pose \
    --enable-hand \
    --max-workers 3 \
    --session-id "my_session" \
    --output-dir "output" \
    --save-video \
    --save-csv \
    --visualize
```

#### 通用參數

```bash
python main.py --mode multimodal \
    --source video.mp4 \
    --pixel-to-cm 0.1 \
    --video-out "output_video.mp4" \
    --save-fps 30 \
    --loglevel DEBUG
```

## 程式碼範例

### 基本手部檢測

```python
from src.hand_recorder import HandRecorder

# 創建檢測器
hand_recorder = HandRecorder()

# 處理圖片
image = cv2.imread("hand_image.jpg")
hands_data = hand_recorder._detect_hands(image)

for hand in hands_data:
    print(f"檢測到 {hand['handedness']} 手，動作：{hand['gesture'].value}")
```

### 多模態整合處理

```python
from src.multimodal_processor import MultimodalProcessor

# 創建處理器
processor = MultimodalProcessor(
    enable_face=True,
    enable_pose=True, 
    enable_hand=True
)

# 處理影片
result = processor.process_video(
    source="input.mp4",
    session_id="multimodal_session",
    output_dir=Path("output")
)
```

### 即時多模態檢測

```python
import cv2
from src.multimodal_processor import MultimodalProcessor

processor = MultimodalProcessor()
processor.start_session("realtime")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 進行多模態檢測
    results = processor.process_frame(frame)
    
    # 繪製結果
    annotated_frame = processor._draw_results(frame, results)
    cv2.imshow('Multimodal Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
processor.stop_session()
```

## 資料結構

### 手部關鍵點 (21 個)

```
手腕 (0)
├── 拇指 (1-4): cmc, mcp, ip, tip
├── 食指 (5-8): mcp, pip, dip, tip
├── 中指 (9-12): mcp, pip, dip, tip
├── 無名指 (13-16): mcp, pip, dip, tip
└── 小指 (17-20): mcp, pip, dip, tip
```

### 手部動作類型

- `fist` - 拳頭
- `open_palm` - 張開手掌
- `thumb_up` - 拇指向上
- `thumb_down` - 拇指向下
- `pointing` - 指向
- `peace` - 比耶
- `ok` - OK 手勢
- `rock` - 搖滾手勢
- `paper` - 布
- `scissors` - 剪刀

### 多模態資料格式

每幀資料包含：

```python
{
    'frame_id': 0,
    'timestamp': 0.0,
    'face_data': {
        'detected': True,
        'num_faces': 1,
        'confidence': 0.8,
        # ... 臉部關鍵點資料
    },
    'pose_data': {
        'detected': True,
        'num_persons': 1,
        'confidence': 0.7,
        # ... 姿態關鍵點資料
    },
    'hand_data': {
        'detected': True,
        'num_hands': 2,
        'hands': [
            {
                'hand_id': 0,
                'handedness': 'Right',
                'gesture': 'open_palm',
                'landmarks': [...]
            }
        ]
    }
}
```

## 輸出格式

### CSV 輸出

手部資料 CSV 包含以下欄位：

- `frame_id` - 幀編號
- `hand_id` - 手部 ID
- `handedness` - 左右手
- `landmark` - 關鍵點名稱
- `x`, `y`, `z` - 3D 座標
- `confidence` - 信心度
- `gesture` - 手部動作
- `model` - 模型標籤

### 影片輸出

- 手部關鍵點和連線的可視化
- 手部動作標籤
- 左右手識別標籤

## 效能考量

### 模型複雜度

- **複雜度 0**：較快，適合即時應用
- **複雜度 1**：較準確，適合離線分析

### 並行處理

多模態處理器使用多執行緒同時處理各模態，提高整體效能：

```python
processor = MultimodalProcessor(max_workers=3)  # 3 個並行執行緒
```

### 記憶體管理

- 自動資源清理
- 支援長時間處理
- 可選的幀儲存功能

## 依賴要求

確保已安裝以下套件：

```bash
pip install mediapipe>=0.10.14
pip install opencv-python>=4.9
pip install numpy>=2.0
pip install pandas>=2.0
pip install ultralytics>=8.2.0
```

## 專案結構

```
yc/
├── main.py                          # 主程式入口點
├── src/
│   ├── __init__.py
│   ├── face_recorder.py            # 臉部檢測模組
│   ├── pose_recorder.py            # 姿態檢測模組
│   ├── hand_recorder.py            # 手部檢測模組（新增）
│   ├── body_analyzer.py            # 身體分析模組
│   ├── pose_visualizer.py          # 姿態視覺化模組
│   ├── multimodal_data.py          # 多模態資料整合（新增）
│   └── multimodal_processor.py     # 多模態處理器（新增）
├── examples/
│   ├── basic_usage.py
│   ├── face_detection_example.py
│   ├── pose_detection_example.py
│   ├── hand_detection_example.py   # 手部檢測範例（新增）
│   └── multimodal_integration_example.py  # 多模態整合範例（新增）
├── tests/
│   ├── test_face_recorder.py
│   ├── test_pose_recorder.py
│   └── test_hand_recorder.py       # 手部檢測測試（新增）
├── docs/
│   ├── API.md
│   ├── FACE_DETECTION.md
│   ├── SETUP.md
│   └── MULTIMODAL_INTEGRATION.md   # 多模態整合說明（新增）
├── requirements.txt
└── README.md
```

## 注意事項

1. **MediaPipe 版本**：需要 MediaPipe 0.10.14 或更新版本
2. **GPU 支援**：手部檢測支援 GPU 加速（如果可用）
3. **即時處理**：建議使用複雜度 0 進行即時應用
4. **記憶體使用**：長時間處理時注意記憶體使用量
5. **錯誤處理**：各模組都有完善的錯誤處理機制

## 未來擴展

- 更多手部動作識別
- 手部動作序列分析
- 與其他模態的深度整合
- 自訂動作訓練
- 手部動作預測

## 授權

本專案採用 MIT 授權條款。