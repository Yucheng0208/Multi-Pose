from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np


class DebugDumper:
	"""簡易偵錯資料輸出工具：將結構化摘要寫入單一 JSONL 檔案"""

	def __init__(self, output_file: Union[str, Path], enabled: bool = True) -> None:
		self.output_file = Path(output_file)
		self.output_file.parent.mkdir(parents=True, exist_ok=True)
		self.enabled = enabled
		# 以追加模式寫入，避免覆蓋過去內容
		try:
			self.output_file.touch(exist_ok=True)
		except Exception as e:
			logging.warning("DebugDumper 初始化失敗: %s", e)
			self.enabled = False

	def _json_safe(self, obj: Any) -> Any:
		"""將物件轉換為可 JSON 序列化的安全表示"""
		try:
			if obj is None:
				return None
			if isinstance(obj, (str, int, float, bool)):
				return obj
			if isinstance(obj, dict):
				return {str(k): self._json_safe(v) for k, v in obj.items()}
			if isinstance(obj, (list, tuple)):
				return [self._json_safe(v) for v in obj]
			if isinstance(obj, np.ndarray):
				return {
					"type": "ndarray",
					"shape": list(obj.shape),
					"dtype": str(obj.dtype),
					"preview": obj.flatten()[:10].tolist() if obj.size > 0 else []
				}
			# 一般物件採用簡短字串表示
			return str(obj)
		except Exception as e:
			return f"<unserializable: {type(obj).__name__}: {e}>"

	def summarize(self, obj: Any, sample_limit: int = 5, nested: bool = False) -> Dict[str, Any]:
		"""回傳物件的結構摘要，避免大物件造成輸出膨脹"""
		try:
			if obj is None:
				return {"type": "None"}
			if isinstance(obj, dict):
				keys = list(obj.keys())
				preview = {}
				for k in keys[:sample_limit]:
					v = obj[k]
					preview[str(k)] = self.summarize(v, sample_limit=3, nested=True) if not nested else str(type(v).__name__)
				return {"type": "dict", "len": len(obj), "keys": [str(k) for k in keys[:20]], "preview": preview}
			if isinstance(obj, (list, tuple)):
				preview = [self.summarize(v, sample_limit=3, nested=True) for v in obj[:sample_limit]]
				return {"type": type(obj).__name__, "len": len(obj), "preview": preview}
			if isinstance(obj, np.ndarray):
				return {"type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
			return {"type": type(obj).__name__, "repr": str(obj)[:200]}
		except Exception as e:
			return {"type": f"<error summarizing: {e}>"}

	def log(self, event: str, data: Dict[str, Any] | None = None) -> None:
		"""寫入一筆 JSON 行，包含事件名稱與資料摘要"""
		if not self.enabled:
			return
		try:
			record = {
				"ts": datetime.utcnow().isoformat() + "Z",
				"event": event,
			}
			if data is not None:
				# 僅做淺層 JSON 安全轉換
				record["data"] = self._json_safe(data)
			with self.output_file.open("a", encoding="utf-8") as f:
				f.write(json.dumps(record, ensure_ascii=False) + "\n")
		except Exception as e:
			logging.warning("DebugDumper 寫入失敗: %s", e)

	def log_summary(self, event: str, obj: Any) -> None:
		"""快速寫入一個物件的結構摘要"""
		self.log(event, {"summary": self.summarize(obj)})

	def log_exception(self, where: str, exc: Exception, context: Dict[str, Any] | None = None) -> None:
		payload = {"where": where, "exception": {
			"type": type(exc).__name__,
			"message": str(exc),
		}}
		if context:
			payload["context"] = self._json_safe(context)
		self.log("exception", payload)
