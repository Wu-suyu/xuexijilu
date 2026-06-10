"""
游戏卡牌检测器 — 基于 OpenCV 传统视觉方案
通过多维度特征综合评分，判断图片是否为游戏卡牌。

评分维度（加权）：
  轮廓矩形(0.30) + 宽高比(0.25) + 面积占比(0.15) + 角点分布(0.15) + 边缘纹理(0.15)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CardDetectionResult:
    is_card: bool
    confidence: float  # 0.0 ~ 1.0
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    details: Optional[dict] = None


class CardDetector:
    WEIGHTS = {"contour": 0.30, "aspect_ratio": 0.25, "area_ratio": 0.15,
               "corner": 0.15, "edge_texture": 0.15}
    DEFAULT_THRESHOLD = 0.55

    # 卡牌宽高比范围（长边/短边）
    ASPECT_RATIO_MIN, ASPECT_RATIO_MAX, ASPECT_RATIO_IDEAL = 1.3, 1.85, 1.55
    # 卡牌占画面面积范围
    AREA_RATIO_MIN, AREA_RATIO_MAX, AREA_RATIO_IDEAL = 0.05, 1.0, 0.35

    def detect(self, image_path: str, threshold: float = None) -> CardDetectionResult:
        img = cv2.imread(image_path)
        if img is None:
            return CardDetectionResult(False, 0.0, details={"error": "无法读取图片"})

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        total_area = h * w

        # 轮廓检测 → 获取评分和边界框
        scores = {}
        scores["contour"], bbox = self._score_contour(gray, total_area)

        # 宽高比 & 面积占比（依赖边界框）
        if bbox:
            bw, bh = bbox[2], bbox[3]
            ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) else 0
            scores["aspect_ratio"] = self._score_by_ratio(
                ratio, self.ASPECT_RATIO_MIN, self.ASPECT_RATIO_MAX, self.ASPECT_RATIO_IDEAL)

            card_area = bw * bh
            area_ratio = card_area / total_area if total_area else 0
            # 满幅卡牌（bbox 几乎等于整图）：给固定高分，避免 _score_by_ratio 边缘值归零
            if area_ratio > 0.90:
                scores["area_ratio"] = 0.75
            else:
                scores["area_ratio"] = self._score_by_ratio(
                    area_ratio, self.AREA_RATIO_MIN, self.AREA_RATIO_MAX, self.AREA_RATIO_IDEAL)
        else:
            scores["aspect_ratio"] = 0.0
            scores["area_ratio"] = 0.0

        scores["corner"] = self._score_corners(gray, h, w)
        scores["edge_texture"] = self._score_edge_texture(gray)

        confidence = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)

        return CardDetectionResult(
            is_card=confidence >= (threshold or self.DEFAULT_THRESHOLD),
            confidence=round(confidence, 4),
            bounding_box=bbox,
            details=scores,
        )

    def _score_contour(self, gray, total_area):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 三种边缘检测方法取最优：
        # Canny → 强对比边缘；Otsu 二值化 → 柔和边界；Sobel → 梯度变化（适合满幅卡牌）
        best_score, best_bbox = 0.0, None
        for edges in [self._canny_edges(blurred),
                      self._threshold_edges(gray),
                      self._sobel_edges(gray)]:
            score, bbox = self._find_best_rect_contour(edges, gray, total_area)
            if score > best_score:
                best_score, best_bbox = score, bbox

        return min(best_score, 1.0), best_bbox

    @staticmethod
    def _canny_edges(blurred):
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    @staticmethod
    def _sobel_edges(gray):
        # Sobel 梯度检测 — 适合卡牌占满画面、边界对比度低的情况
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2).astype(np.uint8)
        _, th = cv2.threshold(sobel_mag, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=3)

    @staticmethod
    def _threshold_edges(gray):
        # Otsu 二值化 — 对柔和边界（如纯色边框）更敏感
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        return thresh

    def _find_best_rect_contour(self, edges, gray, total_area):
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, None

        best_score, best_bbox = 0.0, None
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(cnt)
            if area / total_area < 0.02:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            score, bbox = 0.0, None

            if len(approx) == 4 and cv2.isContourConvex(approx):
                angles = [self._angle_at(approx, i) for i in range(4)]
                angle_score = max(0.0, 1.0 - sum(abs(a - 90) for a in angles) / 120.0)
                rect_area = cv2.contourArea(approx)
                rect_fill = area / rect_area if rect_area else 0

                x, y, bw, bh = cv2.boundingRect(approx)
                # 宽高比合理性：卡牌长边/短边应在 1.3~1.8 之间（收紧以排除笔、湿巾等）
                aspect = max(bw, bh) / min(bw, bh) if min(bw, bh) else 0
                if aspect < self.ASPECT_RATIO_MIN or aspect > self.ASPECT_RATIO_MAX:
                    score = 0.0
                elif self._is_white_background(gray, x, y, bw, bh):
                    score = 0.0
                else:
                    score = 0.6 * angle_score + 0.4 * rect_fill
                bbox = (x, y, bw, bh)

            elif len(approx) > 4:
                rect = cv2.minAreaRect(cnt)
                rect_area = rect[1][0] * rect[1][1]
                rect_fill = area / rect_area if rect_area else 0
                # 收紧条件：fill 必须更高，且面积占比更大，才接受 minAreaRect
                bw2, bh2 = rect[1]
                aspect2 = max(bw2, bh2) / min(bw2, bh2) if min(bw2, bh2) else 0
                if (rect_fill > 0.80 and area / total_area > 0.05
                        and self.ASPECT_RATIO_MIN <= aspect2 <= self.ASPECT_RATIO_MAX):
                    box = cv2.boxPoints(rect).astype(int)
                    # 验证：排除纯白背景
                    x2, y2, bw_r, bh_r = cv2.boundingRect(box)
                    if not self._is_white_background(gray, x2, y2, bw_r, bh_r):
                        score = rect_fill * 0.7
                    x, y, bw, bh = x2, y2, bw_r, bh_r
                    bbox = (x, y, bw, bh)

            if score > best_score:
                best_score, best_bbox = score, bbox

        return min(best_score, 1.0), best_bbox

    @staticmethod
    def _angle_at(approx, i):
        n = len(approx)
        p1, p2, p3 = (approx[(i-1) % n][0].astype(float),
                      approx[i][0].astype(float),
                      approx[(i+1) % n][0].astype(float))
        v1, v2 = p1 - p2, p3 - p2
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    @staticmethod
    def _score_by_ratio(value, vmin, vmax, ideal):
        if value < vmin or value > vmax:
            return 0.0
        dist = abs(value - ideal)
        max_dist = max(ideal - vmin, vmax - ideal)
        return max(0.0, 1.0 - dist / max_dist)

    @staticmethod
    def _is_white_background(gray, x, y, w, h):
        """检测矩形四边是否为纯白色背景（截图/UI元素特征）。

        真实卡牌的四边有颜色变化（均值不会接近255，std不会太小）。
        截图/UI 的"矩形"四边通常是纯白色（均值>230 且 std<10）。
        """
        edge_pixels = []
        edge_pixels.extend(gray[y:y+8, x:x+w].flatten())
        edge_pixels.extend(gray[y+h-8:y+h, x:x+w].flatten())
        edge_pixels.extend(gray[y:y+h, x:x+8].flatten())
        edge_pixels.extend(gray[y:y+h, x+w-8:x+w].flatten())
        if not edge_pixels:
            return False
        arr = np.array(edge_pixels)
        return np.mean(arr) > 230 and np.std(arr) < 10

    def _score_corners(self, gray, h, w):
        dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        corners = np.argwhere(dst > 0.01 * dst.max())
        if len(corners) < 4:
            return 0.0

        mid_y, mid_x = h // 2, w // 2
        quads = set()
        for cy, cx in corners:
            q = (0 if cy < mid_y else 1) * 2 + (0 if cx < mid_x else 1)
            quads.add(q)
            if len(quads) == 4:
                return 1.0
        return len(quads) / 4.0

    @staticmethod
    def _score_edge_texture(gray):
        edges = cv2.Canny(gray, 50, 150)
        density = np.count_nonzero(edges) / gray.size
        if density < 0.02:
            return 0.1
        if density > 0.40:
            return 0.2
        if 0.05 <= density <= 0.20:
            return 1.0
        if 0.03 <= density < 0.05:
            return 0.6
        if 0.20 < density <= 0.25:
            return 0.7
        return 0.4
