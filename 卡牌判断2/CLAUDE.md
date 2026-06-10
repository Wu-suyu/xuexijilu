# 游戏卡牌识别系统

基于 OpenCV 传统视觉方案，自动判断图片是否为游戏卡牌。识别为卡牌则上传到后端 API，非卡牌则跳过。

## 项目结构

```
卡牌判断/
├── main.py               # 主入口：CLI 命令行工具
├── card_detector.py      # 卡牌检测核心逻辑（多维度加权评分）
├── backend_client.py     # 后端上传客户端（支持重试）
├── requirements.txt      # Python 依赖
└── data/                 # 图片数据集目录（直接丢图片进去即可）
```

## 核心逻辑

### 卡牌检测（card_detector.py）

通过 5 个维度加权评分判断是否为卡牌（总分 ≥ 0.55 判定为卡牌）：

| 维度 | 权重 | 说明 |
|------|------|------|
| 轮廓矩形 | 30% | Canny 边缘检测 + 多边形近似，寻找凸四边形 |
| 宽高比 | 25% | 卡牌长边/短边比值在 1.2~2.0 之间 |
| 面积占比 | 15% | 卡牌占画面面积在 5%~85% 之间 |
| 角点分布 | 15% | Harris 角点检测，4 个角点需分布在 4 个象限 |
| 边缘纹理 | 15% | 边缘密度在 0.02~0.40 之间（非纯色、非过度复杂） |

### 后端上传（backend_client.py）

- `multipart/form-data` 格式上传，附带 confidence 和 bounding_box 参数
- 失败自动重试 3 次
- 支持自定义请求头（如认证 token）

## 使用方法

```bash
# 1. 安装依赖（仅首次）
pip install -r requirements.txt

# 2. 把图片放到 data/ 目录（无需分类）

# 3. 只检测不上传（调试/预览用）
python main.py data/ --no-upload

# 4. 检测 + 上传后端
python main.py data/

# 5. 自定义后端地址和认证
python main.py data/ --backend-url http://xxx:8000 --header "Authorization: Bearer xxx"

# 6. 调整检测阈值（越高越严格）
python main.py data/ --no-upload --threshold 0.7

# 7. 导出 JSON 结果
python main.py data/ --no-upload --output result.json
```

## 依赖

- Python 3.12+
- opencv-python
- numpy
- requests
- Pillow
