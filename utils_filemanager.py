# -*- coding: utf-8 -*-
import shutil, datetime
from pathlib import Path
import config

def ensure_dirs():
    for d in [config.OUTPUT_DIR, config.VIDEO_DIR, config.JSON_DIR, config.CSV_DIR, config.NPY_DIR, config.TENSOR_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

def archive_old_outputs():
    base = Path(config.OUTPUT_DIR)
    base.mkdir(parents=True, exist_ok=True)

    # 找出 outputs 底下已有的「檔案」
    files = [p for p in base.glob("**/*") if p.is_file()]
    if not files:
        return

    today = datetime.datetime.now().strftime("%Y%m%d")
    idx = 1
    while True:
        folder = base / f"{today}_{idx:04d}"
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            break
        idx += 1

    for f in files:
        rel = f.relative_to(base)
        dst = folder / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dst))

    print(f"[INFO] 舊輸出已搬至: {folder}")

def next_sequential_name(folder, pad: int = 14) -> str:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # 只看有副檔名的檔案
    nums = []
    for p in folder.iterdir():
        if p.is_file():
            try:
                nums.append(int(p.stem))
            except Exception:
                pass
    next_id = max(nums) + 1 if nums else 1
    return str(next_id).zfill(pad)
