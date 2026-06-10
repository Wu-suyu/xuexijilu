"""
游戏卡牌识别 — 主入口

用法:
    python main.py <图片路径>              # 检测单张图片
    python main.py <目录路径>              # 批量处理目录下所有图片
    python main.py <图片路径> --no-upload  # 只检测不上传
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

from card_detector import CardDetector
from backend_client import BackendClient

DEFAULT_BACKEND_URL = "http://localhost:8000"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".jfif"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def collect_images(source):
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        return sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(source)
            for f in files
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
        )
    raise FileNotFoundError(f"路径不存在: {source}")


def process_image(image_path, detector, client=None, threshold=None):
    filename = os.path.basename(image_path)
    result = detector.detect(image_path, threshold=threshold)

    # 控制台实时输出
    tag = "✅ 卡牌" if result.is_card else "❌ 非卡牌"
    score = ", ".join(f"{k}={v:.3f}" for k, v in result.details.items()) if result.details else ""
    print(f"  {filename}  →  {tag}  置信度: {result.confidence:.4f}")
    print(f"    维度得分: {score}")

    output = {
        "file": filename,
        "is_card": bool(result.is_card),
        "confidence": float(result.confidence),
        "uploaded": False,
    }
    if result.bounding_box:
        output["bounding_box"] = [int(v) for v in result.bounding_box]

    if result.is_card and client is not None:
        try:
            resp = client.upload(image_path, result.confidence, result.bounding_box)
            output["uploaded"] = True
            output["backend_response"] = resp
            print(f"    上传成功 ✓")
            logger.info(f"上传成功: {filename}")
        except Exception as e:
            output["upload_error"] = str(e)
            print(f"    上传失败 ✗: {e}")
            logger.error(f"上传失败 {filename}: {e}")
    elif result.is_card:
        print(f"    (未配置后端地址，跳过上传)")
    else:
        print(f"    (跳过)")

    return output


def main():
    parser = argparse.ArgumentParser(description="游戏卡牌识别 — 检测图片是否为游戏卡牌，如果是则上传到后端")
    parser.add_argument("source", help="图片路径或包含图片的目录")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL, help=f"后端地址 (默认: {DEFAULT_BACKEND_URL})")
    parser.add_argument("--upload-path", default="/api/upload", help="上传接口路径 (默认: /api/upload)")
    parser.add_argument("--no-upload", action="store_true", help="只检测不上传")
    parser.add_argument("--threshold", type=float, default=None, help="检测阈值 (默认: 0.55)")
    parser.add_argument("--output", help="输出 JSON 结果文件路径")
    parser.add_argument("--header", action="append", default=[], help="额外请求头 Key:Value (可多次使用)")

    args = parser.parse_args()

    headers = {}
    for h in args.header:
        if ":" in h:
            k, v = h.split(":", 1)
            headers[k.strip()] = v.strip()

    detector = CardDetector()
    client = None if args.no_upload else BackendClient(
        base_url=args.backend_url, upload_path=args.upload_path,
        headers=headers or None,
    )

    try:
        images = collect_images(args.source)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not images:
        logger.warning("未找到任何图片")
        sys.exit(0)

    print(f"\n共找到 {len(images)} 张图片，开始检测...\n")

    results = [process_image(p, detector, client, threshold=args.threshold) for p in images]

    card_count = sum(1 for r in results if r["is_card"])
    upload_count = sum(1 for r in results if r.get("uploaded"))
    print(f"\n{'='*50}")
    print(f"  检测完成: {len(results)} 张图片 | {card_count} 张卡牌 | {upload_count} 张上传成功")
    print(f"{'='*50}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {args.output}")

    if client:
        client.close()


if __name__ == "__main__":
    main()
