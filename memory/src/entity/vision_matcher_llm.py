"""
视觉匹配器 - 方法2 (VL model based)

使用Qwen3-VL-4B-Instruct模型批量比较new_object_crops和
候选实体的代表图像，计算视觉相似度
"""

import os
import sys
import re
import torch
from typing import Dict, List
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# 添加src目录到路径
src_dir = os.path.dirname(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


class VisionMatcherLM:
    """视觉匹配器 - LM模型方法"""
    
    def __init__(self, config):
    
    我现在想要实现基于LM的方法，
    我们在算完topn的candidate之后，会把全局的描述，和event的object的描述一起输入进去，然后让大模型判断是否是一个物体
    输入：event的object，event对应的topn个的叙述
    模型输出物体的相似关系，比如object和对应物体的相似度000，那就是新物体，010，那就是和第二个对应
    prompt写到/data/xuyuan/Egolife_env/ego-things/personalization/configs/prompts里，增加一个py
    然后我们load本地的qwen模型，路径有config控制，比如/data/xuyuan/Egolife_env/ego-things/personalization/models/entity/Qwen2.5-1.5B
    然后输出让/data/xuyuan/Egolife_env/ego-things/personalization/src/entity/global_entity_manager.py能够兼容解析




