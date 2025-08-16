# 1) 設計目標（Design Goals）

* 同時支援：

  * 直接執行（`python pose_recorder.py --source 0`）
  * 被其他模組 `import` 後呼叫（提供清楚的函式 API）
* 使用 **YOLOv11-pose**（Ultralytics）進行骨架偵測（pose estimation），並透過 **追蹤器（tracker）** 維持跨幀一致的 `id`。
* 每幀輸出每個人、每個關鍵點（keypoint）的紀錄列，欄位固定為：

  * `id`（追蹤 ID）
  * `keypoints`（關鍵點名稱，如 nose、left\_shoulder）
  * `model`（模型名或權重檔名，例：`yolo11n-pose.pt`）
  * `coor_x`、`coor_y`（像素座標）
  * `cm`（以公分為單位之實距；無標定時可輸出 `NA` 或 `-1` 佔位）
* 可切換輸出格式：CSV（預設）、Parquet。
* 可選擇是否輸出偵測疊圖影片（overlayed video）。
* 預留 **相機標定（camera calibration）** 與 **像素到公分（pixel-to-cm）** 的轉換掛勾（hook），未來可用 MediaPipe 或自有標定流程補上。

# 2) 依賴與版本（Dependencies & Versions）

為了未來與 **MediaPipe** 串接穩定，建議使用下列版本範圍（Version ranges）：

* Python ≥ 3.10
* **ultralytics**（Ultralytics YOLO）：`>=8.2.0`（支援 YOLOv11-pose 與 `model.track`）
* **opencv-python**（OpenCV）：`>=4.9`
* **numpy**：`>=1.26`
* **pandas**：`>=2.0`
* **pyarrow**：`>=15.0`（若要輸出 Parquet）
* **mediapipe**：`>=0.10.14`（預留未來整合；本程式不直接依賴）
* （選用）**lapx** 或 **scipy**：供特定 tracker 或後處理使用（依需求）

> 相容性建議（Compatibility Notes）
>
> * Ultralytics 的 `model.track` 預設支援 ByteTrack（`bytetrack.yaml`），與 pose 任務可共用追蹤 ID（`persist=True`）。
> * MediaPipe（`mediapipe`）0.10.x 與 OpenCV 4.9+ 搭配通常穩定；如需 GPU/硬體加速，請依平台另行評估。
> * 若將來改用 StrongSORT/OC-SORT，請保留 tracker 設定入口（`--tracker` 參數）。

# 3) Keypoints 定義（COCO-17）

預設採 **COCO-17**：
`[nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]`

> 注意：Ultralytics pose 輸出為 Nx17x3（x, y, confidence），本程式只取 x、y；可選擇另外輸出 `score`。

# 4) 輸出資料結構（Data Schema）

* 欄位（Columns）：

  1. `id`（int）：追蹤 ID（tracker assigned）
  2. `keypoints`（str）：關鍵點名稱（例：`left_wrist`）
  3. `model`（str）：模型權重名（例：`yolo11n-pose.pt`）
  4. `coor_x`（int/float）：像素 X
  5. `coor_y`（int/float）：像素 Y
  6. `cm`（float/str）：公分（無標定則 `NA`）
* 範例（CSV 一列）：
  `3,left_wrist,yolo11n-pose.pt,512.3,241.8,NA`

# 5) 指令列（CLI）與環境變數

* 基本參數：

  * `--source`：影像來源，預設 `0`（攝影機），也可放影片路徑或 RTSP/HTTP URL
  * `--weights`：YOLOv11-pose 權重，預設 `yolo11n-pose.pt`
  * `--device`：運算裝置（cpu / cuda:0 / mps），預設自動
  * `--conf`：偵測信心（confidence）閾值，預設 `0.25`
  * `--iou`：NMS iou 閾值，預設 `0.7`
  * `--tracker`：追蹤器設定（`bytetrack.yaml`），可自訂路徑
  * `--output`：輸出資料路徑（CSV/Parquet 由副檔名決定，如 `records.csv` 或 `records.parquet`）
  * `--video-out`：選填，輸出疊圖影片路徑（如 `overlay.mp4`）
  * `--visualize`：顯示視窗（true/false）
  * `--save-fps`：疊圖影片 FPS，預設跟來源相同
  * `--pixel-to-cm`：像素轉公分比例（`float`），若未提供則輸出 `NA`
  * `--kp-score-min`：關鍵點最小分數（score）門檻，低於則略過該點，預設 `0.0`

* 典型執行：

  ```bash
  python pose_recorder.py --source 0 --weights yolo11n-pose.pt \
    --output records.csv --tracker bytetrack.yaml --visualize true
  ```

# 6) 模組 API（供 import 使用）

```python
from pose_recorder import PoseRecorder, run_inference

rec = PoseRecorder(
    weights="yolo11n-pose.pt",
    tracker="bytetrack.yaml",
    device="cuda:0",
    conf=0.25,
    iou=0.7,
    pixel_to_cm=None,              # or float 比例
    kp_score_min=0.0
)
rec.process_source(
    source=0,
    output_path="records.parquet",
    video_out_path="overlay.mp4",
    visualize=False,
)

# 或使用便利函式
run_inference(
    source="video.mp4",
    weights="yolo11n-pose.pt",
    output="records.csv",
    video_out=None,
    visualize=False
)
```

# 7) 日誌（Logging）與錯誤處理

* 以 `logging`（INFO/DEBUG/ERROR）輸出關鍵事件：模型載入、tracker 啟用、每幀人數、寫檔進度。
* 針對常見錯誤顯示明確訊息：

  * 權重檔不存在
  * 無法開啟影像來源
  * 追蹤器設定檔載入失敗
  * 寫檔失敗（路徑/權限/磁碟空間）

# 8) 未來與 MediaPipe 整合的預留點

* `pixel_to_cm` 轉換：保留 `calibrate_to_cm(points, meta)` 掛勾，可替換為 MediaPipe 標定或深度估計（depth estimation）。
* `export_mediapipe_proto`：可新增輸出為 MediaPipe-friendly 的結構（如 per-frame dict 或 TFRecord）。
* 關鍵點命名表與索引對照：集中定義於 `KEYPOINTS_C17`，若未來改用 25/33 點，僅需更新此表與索引 map。

---

## 可直接使用的 Python 程式骨架（Script Skeleton）

```python
# pose_recorder.py
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any, Tuple, Generator

import numpy as np
import pandas as pd
import cv2

try:
    from ultralytics import YOLO  # YOLOv11-pose
except Exception as e:
    raise RuntimeError("Ultralytics (YOLO) is required. Please `pip install ultralytics`.") from e


# ---- Keypoint spec (COCO-17) ----
KEYPOINTS_C17: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


class PoseRecorder:
    """
    Record per-frame, per-ID, per-keypoint rows:
    <id><keypoints><model><coor_x><coor_y><cm>
    """

    def __init__(
        self,
        weights: str = "yolo11n-pose.pt",
        tracker: Optional[str] = "bytetrack.yaml",
        device: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.7,
        pixel_to_cm: Optional[float] = None,
        kp_score_min: float = 0.0,
        model_label: Optional[str] = None,
    ) -> None:
        self.weights = weights
        self.tracker = tracker
        self.device = device
        self.conf = conf
        self.iou = iou
        self.pixel_to_cm = pixel_to_cm
        self.kp_score_min = kp_score_min
        self.model_label = model_label or Path(weights).name

        # Load model
        self.model = YOLO(self.weights)
        logging.info("Loaded model: %s", self.model_label)

    def _iter_track_results(
        self, source: Any
    ) -> Generator[Any, None, None]:
        """
        Stream tracking results using Ultralytics .track(stream=True, persist=True).
        """
        # Note: task is inferred from the model (pose). persist keeps IDs consistent.
        results_gen = self.model.track(
            source=source,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker,
            device=self.device,
            stream=True,
            persist=True,
            verbose=False,
        )
        for r in results_gen:
            yield r

    def _rows_from_result(self, r: Any) -> List[Tuple[int, str, str, float, float, Any]]:
        """
        Convert one result to rows of (id, keypoint_name, model, x, y, cm)
        """
        rows: List[Tuple[int, str, str, float, float, Any]] = []

        # boxes.id: (N,) tensor-like or None
        ids = None
        if hasattr(r, "boxes") and hasattr(r.boxes, "id") and r.boxes.id is not None:
            try:
                ids = r.boxes.id.cpu().numpy().astype(int)
            except Exception:
                ids = np.array([-1] * len(r))

        # keypoints: shape (N, 17, 3)
        kps = getattr(r, "keypoints", None)
        if kps is None or kps.data is None:
            return rows

        kp_arr = kps.data.cpu().numpy()  # (N, 17, 3)
        n = kp_arr.shape[0]
        if ids is None:
            ids = np.arange(n, dtype=int)

        for i in range(n):
            pid = int(ids[i]) if i < len(ids) else -1
            for k_idx, k_name in enumerate(KEYPOINTS_C17):
                x, y, score = kp_arr[i, k_idx, :]
                if np.isnan(x) or np.isnan(y):
                    continue
                if score < self.kp_score_min:
                    continue
                cm_val = (
                    (float(self.pixel_to_cm) * float(np.hypot(1.0, 0.0)))
                    if self.pixel_to_cm is not None else "NA"
                )
                rows.append((pid, k_name, self.model_label, float(x), float(y), cm_val))
        return rows

    def process_source(
        self,
        source: Any,
        output_path: Optional[str] = "records.csv",
        video_out_path: Optional[str] = None,
        visualize: bool = False,
        save_fps: Optional[float] = None,
    ) -> None:
        """
        Run tracking+pose on a source and dump rows to CSV/Parquet.
        """
        writer = None
        video_writer = None
        out_is_parquet = bool(output_path and str(output_path).lower().endswith(".parquet"))
        accum: List[Tuple[int, str, str, float, float, Any]] = []

        vc_fps = None

        try:
            for r in self._iter_track_results(source):
                # video frame (BGR) for drawing/visualization
                frame = getattr(r, "orig_img", None)

                # collect rows
                rows = self._rows_from_result(r)
                accum.extend(rows)

                # draw if needed
                if (visualize or video_out_path) and frame is not None:
                    # Ultralytics provides plotted image via r.plot()
                    plotted = r.plot()  # ndarray BGR
                    if visualize:
                        cv2.imshow("pose_recorder", plotted)
                        if cv2.waitKey(1) & 0xFF == 27:  # ESC
                            break

                    if video_out_path:
                        if video_writer is None:
                            h, w = plotted.shape[:2]
                            if save_fps is None:
                                # try recover fps from source
                                try:
                                    cap = cv2.VideoCapture(source if isinstance(source, (str, int)) else 0)
                                    vc_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                                    cap.release()
                                except Exception:
                                    vc_fps = 30.0
                            else:
                                vc_fps = float(save_fps)

                            _ensure_dir(Path(video_out_path))
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            video_writer = cv2.VideoWriter(str(video_out_path), fourcc, vc_fps, (w, h))
                        video_writer.write(plotted)

            # dump
            if output_path:
                _ensure_dir(Path(output_path))
                df = pd.DataFrame(accum, columns=["id", "keypoints", "model", "coor_x", "coor_y", "cm"])
                if out_is_parquet:
                    df.to_parquet(output_path, index=False)
                else:
                    df.to_csv(output_path, index=False)
                logging.info("Saved records to %s (%d rows)", output_path, len(df))

        finally:
            if video_writer is not None:
                video_writer.release()
            if visualize:
                cv2.destroyAllWindows()


def run_inference(
    source: Any,
    weights: str = "yolo11n-pose.pt",
    output: str = "records.csv",
    tracker: Optional[str] = "bytetrack.yaml",
    device: Optional[str] = None,
    conf: float = 0.25,
    iou: float = 0.7,
    pixel_to_cm: Optional[float] = None,
    kp_score_min: float = 0.0,
    video_out: Optional[str] = None,
    visualize: bool = False,
    save_fps: Optional[float] = None,
) -> None:
    rec = PoseRecorder(
        weights=weights, tracker=tracker, device=device,
        conf=conf, iou=iou, pixel_to_cm=pixel_to_cm, kp_score_min=kp_score_min
    )
    rec.process_source(
        source=source,
        output_path=output,
        video_out_path=video_out,
        visualize=visualize,
        save_fps=save_fps,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLOv11-pose recorder")
    p.add_argument("--source", type=str, default="0", help="0 for webcam, or path/URL")
    p.add_argument("--weights", type=str, default="yolo11n-pose.pt")
    p.add_argument("--device", type=str, default=None, help="cpu / cuda:0 / mps")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--tracker", type=str, default="bytetrack.yaml")
    p.add_argument("--output", type=str, default="records.csv")
    p.add_argument("--video-out", type=str, default=None)
    p.add_argument("--visualize", type=str, default="false")
    p.add_argument("--save-fps", type=float, default=None)
    p.add_argument("--pixel-to-cm", type=float, default=None)
    p.add_argument("--kp-score-min", type=float, default=0.0)
    p.add_argument("--loglevel", type=str, default="INFO")
    return p


def _to_bool(s: str) -> bool:
    return str(s).lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.loglevel).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    source: Any
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    run_inference(
        source=source,
        weights=args.weights,
        output=args.output,
        tracker=args.tracker,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        pixel_to_cm=args.pixel_to_cm,
        kp_score_min=args.kp_score_min,
        video_out=args.video_out,
        visualize=_to_bool(args.visualize),
        save_fps=args.save_fps,
    )


if __name__ == "__main__":
    main()
```

---

⚠️ 注意事項

* 若裝置不支援 GPU，`--device cpu` 可避免 CUDA 相關錯誤（English term: CUDA）。
* `cm` 欄位目前提供掛勾，**未標定（calibration）** 時請保持 `NA`；一旦你有單位長度或相機內外參，即可在 `pixel_to_cm` 或自訂轉換函式加上實距換算。
* 影片疊圖輸出（overlay）可能造成效能下降；若追求最高效能，關閉 `--visualize` 與 `--video-out`。
* 若後續更換為 25/33 關節模型，請同步更新 `KEYPOINTS_C17` 與 `self._rows_from_result` 的索引長度檢查。
* 與 **MediaPipe** 整合時，若需同時載入兩邊的 **GPU 後端（backend）**，請留意驅動與 runtime 版本一致性，避免衝突。
