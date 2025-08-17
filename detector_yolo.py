# -*- coding: utf-8 -*-
from pathlib import Path
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, weights, conf=0.5, iou=0.5, device="cpu", tracker="bytetrack.yaml"):
        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.device = device
        self.tracker = self._normalize_tracker(tracker)

    def _normalize_tracker(self, tracker):
        if not tracker:
            return None
        t = str(tracker).strip().lower()
        if t in ("bytetrack", "botsort"):
            return t + ".yaml"
        if not (t.endswith(".yaml") or t.endswith(".yml")):
            return "bytetrack.yaml"
        return t

    def track_on_frame(self, frame_bgr, classes=None):
        # persist=True 保持多幀 ID 連續；verbose=False 靜音
        return self.model.track(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=classes,
            tracker=self.tracker,
            persist=True,
            verbose=False,
            stream=False,
        )

    @staticmethod
    def parse_boxes(result):
        """
        回傳：xyxy_list, confs, tids, clss
        """
        xyxy_list, confs, tids, clss = [], [], [], []
        if not result:
            return xyxy_list, confs, tids, clss
        res0 = result[0]  # 單張影像
        if getattr(res0, "boxes", None) is None:
            return xyxy_list, confs, tids, clss
        for b in res0.boxes:
            # xyxy: (x1,y1,x2,y2)
            xyxy_list.append(b.xyxy[0].cpu().numpy())
            confs.append(float(b.conf[0].cpu().numpy()))
            tids.append(int(b.id[0].cpu().numpy()) if b.id is not None else None)
            clss.append(int(b.cls[0].cpu().numpy()))
        return xyxy_list, confs, tids, clss
