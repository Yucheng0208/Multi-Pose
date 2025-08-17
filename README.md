# Multi-Pose: A Mesh-Structured Network of Keypoints for Landmark Detection
*(多模態姿態：用於特徵點偵測的網格結構化關鍵點網絡)*

![Demo GIF](https_github_com_polite-AI_Real-time-Sign-Language-to-Speech-Conversion/assets/95745582/d0ba326b-d3a9-467f-94a2-97210986716d)
*(建議：請將上方連結替換為您自己錄製的專案展示 GIF 或圖片)*

**Multi-Pose** 是一個高效能、即時的框架，專為捕捉人體全面的動態特徵點而設計。本框架的核心思想是融合不同模型的優勢——結合 **YOLOv8-Pose** 的快速全身姿態估計與 **Google MediaPipe** 的高精度手部及臉部細節——從而為單一人物生成一個統一的、網格結構化的 **537個特徵點（Landmarks）** 網絡。

這個系統不僅僅是一個偵測工具，它更是一個強大的數據採集引擎，為手語辨識 (SLR)、虛擬化身控制 (Avatar Control)、情感分析 (Emotion Analysis) 和進階人機互動 (HCI) 等應用提供了極其豐富、結構化的數據來源。

## ✨ 核心功能 (Core Features)

-   **🚀 即時高效能 (Real-Time High Performance)**：在 GPU 加速下實現流暢的即時偵測，並能精確計算與顯示真實的處理幀率 (Real FPS)，以評估系統性能。
-   **🧩 混合模型架構 (Hybrid Model Architecture)**：
    -   **全身姿態 (Full Body)**: 使用 `YOLOv8-Pose` 進行快速且穩健的人體偵測與17個主要關節點定位。
    -   **精細手部 (High-Fidelity Hands)**: 使用 `MediaPipe Hands` 在全畫幅上對雙手進行偵測，捕捉每隻手21個細微的指關節點。
    -   **密集臉部 (Dense Face Mesh)**: 使用 `MediaPipe Face Mesh` 在 YOLO 定位的臉部區域內，生成高達478個特徵點的密集網格，精準捕捉表情細節。
-   **📊 全面的特徵點覆蓋 (Comprehensive Landmark Coverage - 537 Points)**：
    -   **身體姿態 (Pose)**: 17 個特徵點
    -   **雙手 (Hands)**: 42 個特徵點 (21 左 + 21 右)
    -   **臉部網格 (Face Mesh)**: 478 個特徵點
-   **💾 結構化 JSON 輸出 (Structured JSON Output)**：
    -   為影片的每一幀生成一個獨立、序列編號的 JSON 檔案 (e.g., `000000000001.json`)。
    -   數據格式清晰，詳細記錄了幀ID、偵測人數、人物ID，以及身體、雙手、臉部所有特徵點的索引、`x, y` 座標與信心度。
-   **🖥️ 資訊豐富的視覺化介面 (Informative Visualization)**：
    -   即時渲染所有模型的偵測結果，包括骨架連線、手部關節和臉部網格。
    -   動態顯示真實FPS、各模組偵測狀態 (OK/X)、總人數等關鍵資訊。

## 🛠️ 技術堆疊 (Tech Stack)

-   **Pose Estimation**: [Ultralytics YOLOv8-Pose](https://github.com/ultralytics/ultralytics)
-   **Hand & Face Landmarks**: [Google MediaPipe](https://developers.google.com/mediapipe)
-   **Core Framework**: PyTorch
-   **Image Processing**: OpenCV
-   **Numerical Computing**: NumPy

## ⚙️ 環境設定與安裝 (Setup and Installation)

### 1. 前置需求
-   Python 3.8+
-   **強烈建議**: NVIDIA GPU with CUDA & cuDNN for real-time performance.

### 2. 複製此儲存庫
```bash
git clone [您的GitHub儲存庫連結]
cd Multi-Pose
```

### 3. 建立並啟用 Python 虛擬環境
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. 安裝依賴套件
```bash
pip install -r requirements.txt
```
若無 `requirements.txt`，請手動安裝：
```bash
pip install ultralytics mediapipe opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
*(請根據您的 CUDA 版本選擇對應的 PyTorch 指令)*

### 5. 下載預訓練模型
在專案根目錄下建立 `models` 資料夾，並下載以下模型檔案放入其中。

```
Multi-Pose/
└── models/
    ├── yolo11n-pose.pt       # YOLOv8-Pose model
    ├── hand_landmarker.task  # MediaPipe Hands model
    └── face_landmarker.task  # MediaPipe Face Mesh model
```
-   YOLO Models: [Ultralytics GitHub Releases](https://github.com/ultralytics/assets/releases)
-   MediaPipe Task Models: [MediaPipe for Python Models Page](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#models)

## 🚀 運行系統 (Running the System)

透過 `main.py` 啟動。目前主要支援即時鏡頭模式。

### Real-Time Webcam Mode
此模式將啟動預設攝影機，並將結構化的 JSON 數據即時儲存至 `output_json/`。
```bash
python main.py --mode realtime
```
-   指定不同的攝影機:
    ```bash
    python main.py --mode realtime --camera 1
    ```
-   使用 CPU 運行 (效能將顯著降低):
    ```bash
    python main.py --device cpu
    ```

## 📦 輸出 JSON 數據結構 (Output JSON Data Structure)

每一幀都會生成一個 JSON 檔案，其數據結構設計清晰，便於解析與使用。

```json
{
    "frame_id": 123,
    "num_persons": 1,
    "persons": [
        {
            "person_id": 0,
            "keypoints": {
                "pose": [
                    { "id": 0, "x": 640.5, "y": 320.1, "confidence": 0.95 },
                    ... 16 more ...
                ],
                "left_hand": [
                    { "id": 0, "x": 410.2, "y": 450.7, "confidence": 0.99 },
                    ... 20 more ...
                ],
                "right_hand": [
                    { "id": 0, "x": 810.2, "y": 450.7, "confidence": 0.99 },
                    ... 20 more ...
                ],
                "face": [
                    { "id": 0, "x": 630.1, "y": 280.6, "confidence": 1.0 },
                    ... 477 more ...
                ]
            }
        }
    ]
}
```
-   **`frame_id`**: 幀的序列號。
-   **`num_persons`**: 畫面中偵測到的總人數。
-   **`persons`**: 包含所有人物數據的列表。
    -   **`person_id`**: 人物的唯一ID（目前主要追蹤ID 0）。
    -   **`keypoints`**: 包含該人物所有特徵點的物件。
        -   **`pose`**, **`left_hand`**, **`right_hand`**, **`face`**: 各部位的特徵點列表。
            -   **`id`**: 該部位內特徵點的索引 (e.g., 0 for nose in pose)。
            -   **`x`, `y`**: 在原始影像畫幅中的絕對像素座標。
            -   **`confidence`**: 該特徵點的信賴分數。

## 🌱 未來規劃 (Roadmap)

-   [ ] **多人追蹤 (Multi-Person Tracking)**: 擴充系統以同時追蹤並為畫面中的多個人生成唯一的ID和JSON數據。
-   [ ] **3D 座標支援 (3D Coordinate Support)**: 將 MediaPipe 提供的 `z` 座標整合到 JSON 輸出中，實現完整的3D姿態數據。
-   [ ] **即時辨識模組 (Real-Time Recognition Module)**: 基於輸出的關鍵點序列，開發一個用於手語或動作辨識的即時分類模組。
-   [ ] **效能優化 (Performance Optimization)**: 針對不同硬體進行模型推論優化，例如使用 TensorRT。
-   [ ] **Docker 支援 (Dockerization)**: 提供 Dockerfile 以簡化部署流程。

## 🤝 貢獻 (Contributing)

歡迎任何形式的貢獻！無論是回報問題 (Issues)、請求新功能，還是提交程式碼合併請求 (Pull Requests)，都對本專案有極大幫助。

## 📄 授權 (License)

本專案採用 [MIT License](LICENSE) 授權。