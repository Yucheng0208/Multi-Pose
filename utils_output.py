# -*- coding: utf-8 -*-
from pathlib import Path
import json, csv, numpy as np
import cv2, torch
import config

README_TEXT = """\
格式說明：
每個檔案為一筆樣本（通常是一幀中的一個人）。
欄位：frame_idx, person_id, keypoint_id(0..542), x(ROI內0..1), y(ROI內0..1), confidence
重新編號規則：0..467=Face(468點), 468..500=Pose(33點), 501..521=左手(21點), 522..542=右手(21點)
備註：若你改成 FACE_USE_468=False，face 會是 478 點，總點數將 > 543。
"""

def ensure_readmes():
    for d in [config.JSON_DIR, config.CSV_DIR, config.NPY_DIR, config.TENSOR_DIR]:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        readme = d / "README.txt"
        if not readme.exists():
            readme.write_text(README_TEXT, encoding="utf-8")

def get_video_writer(path, fps, size_wh):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size_wh
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (w, h))

def _records_from_kp(base_dict, kp_all):
    recs = []
    for kp in kp_all:
        recs.append([base_dict["frame_idx"], base_dict["person_id"], kp["kid"], kp["x"], kp["y"], kp["conf"]])
    return recs

def write_records_per_person(basename: str, frame_idx: int, person_id: int, kp_all: list):
    """
    basename: '00000000000001' 這樣的流水號
    依 config 的 SAVE_*，在各自目錄寫出一份檔案
    """
    meta = {"frame_idx": int(frame_idx), "person_id": int(person_id)}
    recs = _records_from_kp(meta, kp_all)

    # JSON
    if config.SAVE_JSON:
        p = Path(config.JSON_DIR) / f"{basename}.json"
        with open(p, "w", encoding="utf-8") as f:
            # 展平結構
            json.dump(
                [{"frame_idx": r[0], "person_id": r[1], "keypoint_id": r[2], "x": r[3], "y": r[4], "confidence": r[5]}
                 for r in recs],
                f, ensure_ascii=False, indent=2
            )

    # CSV
    if config.SAVE_CSV:
        p = Path(config.CSV_DIR) / f"{basename}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "person_id", "keypoint_id", "x", "y", "confidence"])
            for r in recs:
                w.writerow(r)

    # NPY
    if config.SAVE_NPY:
        p = Path(config.NPY_DIR) / f"{basename}.npy"
        arr = np.array(recs, dtype=np.float32)
        np.save(p, arr)

    # Tensor（PyTorch）
    if config.SAVE_TENSOR:
        p = Path(config.TENSOR_DIR) / f"{basename}.pt"
        tensor = torch.tensor(recs, dtype=torch.float32)
        torch.save(tensor, p)
