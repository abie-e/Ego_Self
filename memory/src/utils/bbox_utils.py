"""
BBox处理工具函数

提供bbox格式转换、IoU计算、缩放等功能
"""

import numpy as np
from typing import List, Union


def xywh_to_xyxy(bbox: Union[List[float], np.ndarray]) -> List[float]:
    """
    将[x,y,w,h]格式转换为[x1,y1,x2,y2]格式
    
    Args:
        bbox: [x, y, width, height]
    
    Returns:
        [x1, y1, x2, y2]
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_xywh(bbox: Union[List[float], np.ndarray]) -> List[float]:
    """
    将[x1,y1,x2,y2]格式转换为[x,y,w,h]格式

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        [x, y, width, height]
    """
    x1, y1, x2, y2 = bbox
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def calculate_iou(
    bbox1: Union[List[float], np.ndarray], 
    bbox2: Union[List[float], np.ndarray],
    format: str = "xywh"
) -> float:
    """
    计算两个bbox的IoU
    
    Args:
        bbox1: 第一个bbox
        bbox2: 第二个bbox
        format: bbox格式，"xywh" 或 "xyxy"
    
    Returns:
        IoU值 (0-1)
    """
    # 转换为xyxy格式
    if format == "xywh":
        b1 = xywh_to_xyxy(bbox1)
        b2 = xywh_to_xyxy(bbox2)
    else:
        b1 = list(bbox1)
        b2 = list(bbox2)
    
    # 计算交集区域
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集区域
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def scale_bbox(
    bbox: Union[List[float], np.ndarray], 
    scale: float
) -> List[float]:
    """
    缩放bbox坐标
    
    Args:
        bbox: [x, y, w, h] 或 [x1, y1, x2, y2]
        scale: 缩放比例
    
    Returns:
        缩放后的bbox（格式保持不变）
    """
    return [coord * scale for coord in bbox]


def clip_bbox_to_image(
    bbox: Union[List[float], np.ndarray],
    image_width: int,
    image_height: int,
    format: str = "xywh"
) -> List[float]:
    """
    将bbox裁剪到图像范围内
    
    Args:
        bbox: bbox坐标
        image_width: 图像宽度
        image_height: 图像高度
        format: bbox格式
    
    Returns:
        裁剪后的bbox
    """
    if format == "xywh":
        x, y, w, h = bbox
        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))
        w = min(w, image_width - x)
        h = min(h, image_height - y)
        return [x, y, w, h]
    else:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))
        return [x1, y1, x2, y2]


def is_valid_bbox(bbox: Union[List[float], np.ndarray], format: str = "xywh") -> bool:
    """
    检查bbox是否有效（非空、坐标合理）
    
    Args:
        bbox: bbox坐标
        format: bbox格式
    
    Returns:
        True表示有效，False表示无效
    """
    if bbox is None or len(bbox) != 4:
        return False
    
    if format == "xywh":
        x, y, w, h = bbox
        # 宽高必须大于0
        return w > 0 and h > 0
    else:
        x1, y1, x2, y2 = bbox
        # x2 > x1 且 y2 > y1
        return x2 > x1 and y2 > y1


def normalize_bbox_to_xyxy(bbox: Union[List[float], np.ndarray], format: str = "xywh") -> List[float]:
    """
    标准化bbox格式为[x1, y1, x2, y2]
    
    用于entity存储中统一bbox格式，所有bbox都存储为x1,y1,x2,y2格式
    
    Args:
        bbox: bbox坐标
        format: 输入bbox格式，"xywh" 或 "xyxy"
    
    Returns:
        [x1, y1, x2, y2]格式的bbox
    """
    if format == "xywh":
        return xywh_to_xyxy(bbox)
    else:
        return list(bbox)


