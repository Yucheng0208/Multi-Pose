# -*- coding: utf-8 -*-
# ---- 靜音與 OpenMP 防雷（務必在任何第三方 import 前）----
import os, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
# 臨時解：避免 OMP #15 直接崩潰（長期解請用 nomkl/乾淨環境）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2, time, argparse, datetime
from pathlib import Path
import config
from detector_yolo import PersonDetector
from processor_mediapipe import MediapipeProcessor
from utils_filemanager import archive_old_outputs, ensure_dirs, next_sequential_name
from utils_output import ensure_readmes, get_video_writer, write_records_per_person

# ---------- YouTube / 來源開啟 ----------
def open_source(src: str):
    # 攝影機
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        return cap, "camera"
    # YouTube
    if getattr(config, "YOUTUBE_ENABLE", False) and ("youtube.com" in src.lower() or "youtu.be" in src.lower()):
        try:
            import pafy  # pip install yt-dlp pafy
        except Exception as e:
            raise RuntimeError("YouTube 需要：pip install yt-dlp pafy") from e
        v = pafy.new(src)
        best = v.getbest(preftype="mp4")
        cap = cv2.VideoCapture(best.url)
        return cap, "youtube"
    # 影片 / 圖片
    cap = cv2.VideoCapture(src)
    if cap.isOpened():
        return cap, "file"
    raise RuntimeError(f"無法開啟來源: {src}")

def is_image_file(path: str) -> bool:
    p = path.lower()
    return p.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"))

# ---------- 視覺化 ----------
def draw_id_on_top(img, box_xyxy, text, color=(0,255,0)):
    x1, y1, x2, y2 = map(int, box_xyxy)
    cx = (x1 + x2) // 2
    y  = max(0, y1 - 8)
    cv2.putText(img, text, (max(0, cx - 40), y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def _pick_primary_dir():
    """決定逐檔輸出的「主目錄」。流水號以主目錄為準，其他格式共用同一號碼。"""
    from pathlib import Path
    if getattr(config, "SAVE_JSON", False):   return Path(config.JSON_DIR)
    if getattr(config, "SAVE_CSV", False):    return Path(config.CSV_DIR)
    if getattr(config, "SAVE_NPY", False):    return Path(config.NPY_DIR)
    if getattr(config, "SAVE_TENSOR", False): return Path(config.TENSOR_DIR)
    return Path(config.JSON_DIR)  # fallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="攝影機=0, 影片/圖片=path, YouTube=url")

    # 互斥群組：有給才覆蓋 config，預設 None
    g_wv = parser.add_mutually_exclusive_group()
    g_wv.add_argument("--write-video",    dest="write_video", action="store_true",  help="啟用寫出影片")
    g_wv.add_argument("--no-write-video", dest="write_video", action="store_false", help="停用寫出影片")

    g_kp = parser.add_mutually_exclusive_group()
    g_kp.add_argument("--draw-kp",    dest="draw_kp", action="store_true",  help="在畫面上畫 keypoints")
    g_kp.add_argument("--no-draw-kp", dest="draw_kp", action="store_false", help="不要畫 keypoints")

    g_js = parser.add_mutually_exclusive_group()
    g_js.add_argument("--save-json",    dest="save_json", action="store_true",  help="儲存 JSON（每幀每人）")
    g_js.add_argument("--no-save-json", dest="save_json", action="store_false", help="不儲存 JSON")

    g_cs = parser.add_mutually_exclusive_group()
    g_cs.add_argument("--save-csv",    dest="save_csv", action="store_true",  help="儲存 CSV（每幀每人）")
    g_cs.add_argument("--no-save-csv", dest="save_csv", action="store_false", help="不儲存 CSV")

    # 單向開關（預設 None，不覆蓋 config；有給就 True）
    parser.add_argument("--save-npy",    dest="save_npy",    action="store_true", default=None, help="儲存 NPY（每幀每人）")
    parser.add_argument("--save-tensor", dest="save_tensor", action="store_true", default=None, help="儲存 Tensor（每幀每人）")

    # 預設為 None：代表使用 config 既定值
    parser.set_defaults(write_video=None, draw_kp=None, save_json=None, save_csv=None)

    args = parser.parse_args()

    # 先建資料夾、再歸檔舊檔、放 README
    ensure_dirs()
    archive_old_outputs()   # 若 outputs 有殘留，會以日期_流水號打包移轉
    ensure_readmes()

    # 覆蓋 config 開關（只在使用者有提供時才覆蓋）
    if args.write_video is not None: config.WRITE_VIDEO   = args.write_video
    if args.draw_kp    is not None: config.DRAW_KEYPOINT = args.draw_kp
    if args.save_json  is not None: config.SAVE_JSON     = args.save_json
    if args.save_csv   is not None: config.SAVE_CSV      = args.save_csv
    if args.save_npy   is not None: config.SAVE_NPY      = args.save_npy
    if args.save_tensor is not None: config.SAVE_TENSOR  = args.save_tensor

    # 開啟來源
    cap, src_type = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"來源無法開啟：{args.source}")

    # FPS 與尺寸
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    if not fps_src or fps_src <= 0:
        fps_src = getattr(config, "VIDEO_FPS_FALLBACK", 30.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # 影片輸出
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = f"{ts}.mp4"
    video_path = Path(config.VIDEO_DIR) / video_name
    writer = get_video_writer(video_path, fps_src, (W, H)) if getattr(config, "WRITE_VIDEO", False) else None

    # 建立 YOLO 與 Mediapipe
    det = PersonDetector(config.YOLO_WEIGHTS, conf=config.YOLO_CONF, iou=config.YOLO_IOU,
                         device=config.YOLO_DEVICE, tracker=config.YOLO_TRACKER)
    mp_proc = MediapipeProcessor(config.FACE_TASK, config.POSE_TASK, config.HAND_TASK,
                                 use_face=config.USE_FACE, use_pose=config.USE_POSE, use_hands=config.USE_HANDS)

    frame_idx = 0
    t_prev = time.time()

    # 圖片模式：讀一張就停；為了輸出影片，單張重複一秒
    single_image_mode = (src_type == "file" and is_image_file(args.source))
    total_repeat_for_image = int(fps_src)  # 1 秒

    primary_dir = _pick_primary_dir()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_vis = frame_bgr.copy()

        # YOLO 追蹤
        result = det.track_on_frame(frame_bgr, classes=[config.YOLO_PERSON_CLASS])
        xyxy_list, confs, tids, clss = det.parse_boxes(result)

        for i, xyxy in enumerate(xyxy_list):
            if clss[i] != config.YOLO_PERSON_CLASS:
                continue
            pid = tids[i] if tids[i] is not None else -1
            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # ROI
            crop_bgr = frame_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

            # MediaPipe：只在 FRONT_ONLY 條件下才算 ID 與輸出
            is_front, kp_all, _ = mp_proc.analyze_crop(
                crop_rgb, timestamp_ms=int(frame_idx * (1000.0 / fps_src)),
                kp_face_count=config.KP_FACE_COUNT, kp_pose_count=config.KP_POSE_COUNT,
                need_front_pose=config.FRONT_ONLY
            )

            if is_front:
                # 逐檔寫出（同一流水號寫入 JSON/CSV/NPY/Tensor）
                base = next_sequential_name(primary_dir, config.ID_PAD)
                write_records_per_person(base, frame_idx=frame_idx, person_id=pid, kp_all=kp_all)

                # 畫框與 ID
                if getattr(config, "DRAW_BOX_ID", True):
                    draw_id_on_top(frame_vis, xyxy, f"ID:{pid}", color=(0,255,0))

                # 畫 keypoints（回投影）
                if getattr(config, "DRAW_KEYPOINT", False):
                    mp_proc.draw_keypoints_on(frame_vis, kp_all, (x1,y1,x2,y2))
            else:
                # 非正面：只畫框
                if getattr(config, "DRAW_BOX_ID", True):
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 180, 255), 2)

        # FPS
        if getattr(config, "DRAW_FPS", True):
            now = time.time()
            fps = 1.0 / (now - t_prev) if now > t_prev else 0.0
            t_prev = now
            text = f"FPS: {fps:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(frame_vis, text, (W - tw - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # 寫影片
        if writer is not None:
            writer.write(frame_vis)

        # 顯示（沒有 "Press Q" 字樣）
        cv2.imshow("Multi-Person FrontOnly Pipeline", frame_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1
        if single_image_mode and frame_idx >= total_repeat_for_image:
            break

    # 收尾
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    mp_proc.close()

if __name__ == "__main__":
    main()
