"""
后端上传客户端
将检测为卡牌的图片上传到后端 API。
"""

import os
import logging

import requests

logger = logging.getLogger(__name__)


class BackendClient:
    def __init__(self, base_url="http://localhost:8000", upload_path="/api/upload",
                 timeout=30, max_retries=3, headers=None):
        self.base_url = base_url.rstrip("/")
        self.upload_url = f"{self.base_url}{upload_path}"
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        if headers:
            self.session.headers.update(headers)

    def upload(self, image_path, confidence, bounding_box=None, extra_fields=None):
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        filename = os.path.basename(image_path)
        data = {"confidence": str(round(confidence, 4))}

        if bounding_box:
            data["bbox_x"] = str(bounding_box[0])
            data["bbox_y"] = str(bounding_box[1])
            data["bbox_w"] = str(bounding_box[2])
            data["bbox_h"] = str(bounding_box[3])
        if extra_fields:
            data.update(extra_fields)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with open(image_path, "rb") as f:
                    files = {"file": (filename, f, "image/jpeg")}
                    logger.info(f"上传到 {self.upload_url} (尝试 {attempt}/{self.max_retries})")
                    resp = self.session.post(self.upload_url, files=files, data=data, timeout=self.timeout)
                    resp.raise_for_status()
                    result = resp.json()
                    logger.info(f"上传成功: {result}")
                    return result
            except requests.RequestException as e:
                last_error = e
                logger.warning(f"上传失败 (尝试 {attempt}): {e}")

        raise last_error

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
