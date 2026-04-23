"""
Entity模块

提供物体追踪和实体管理功能
"""

import os
import sys

# 添加src目录到路径
src_dir = os.path.dirname(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 检查文件是否存在，避免import错误
import os
_current_dir = os.path.dirname(__file__)

# 延迟导入ObjectTracker和GroundedSAM2Wrapper（它们依赖cv2等重量级库）
# 只在实际使用时才导入
try:
    from entity.object_tracker import ObjectTracker
    from entity.grounded_sam2_wrapper import GroundedSAM2Wrapper
except ImportError:
    ObjectTracker = None
    GroundedSAM2Wrapper = None

from entity.entity_storage import EntityStorage

# 新增的全局实体匹配模块
if os.path.exists(os.path.join(_current_dir, "global_entity_manager.py")):
    from entity.global_entity_manager import GlobalEntityManager
if os.path.exists(os.path.join(_current_dir, "text_matcher.py")):
    from entity.text_matcher import TextMatcher
if os.path.exists(os.path.join(_current_dir, "vision_matcher_iou.py")):
    from entity.vision_matcher_iou import VisionMatcherIoU

__all__ = [
    "ObjectTracker",
    "GroundedSAM2Wrapper",
    "EntityStorage",
    "GlobalEntityManager",
    "TextMatcher",
    "VisionMatcherIoU"
]

