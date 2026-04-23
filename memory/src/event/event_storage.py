"""
Event存储模块

功能：
1. 保存Event JSON文件
2. 保存特征向量（embedding, vision, audio）
3. 加载Event数据
"""

import os
import json
import re
import sys
from typing import Dict, Optional
import numpy as np

# 添加src目录到路径
src_dir = os.path.dirname(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 不再需要save_json_compact，直接使用标准json


class EventStorage:
    """Event存储管理器"""
    
    def __init__(self, events_dir: str, features_dir: str):
        """
        初始化Event存储管理器
        
        Args:
            events_dir: Events目录路径
            features_dir: Features目录路径
        """
        self.events_dir = events_dir
        self.features_dir = features_dir
        
        # 确保目录存在
        os.makedirs(events_dir, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)
    
    def save_event(
        self,
        event_data: Dict,
        embedding: Optional[np.ndarray] = None,
        vision_feature: Optional[np.ndarray] = None,
        audio_feature: Optional[np.ndarray] = None
    ) -> str:
        """
        保存event
        
        Args:
            event_data: Event数据字典
            embedding: Caption embedding（可选）
            vision_feature: 视觉特征（可选）
            audio_feature: 音频特征（可选）
        
        Returns:
            Event JSON文件路径
        """
        event_id = event_data["event_id"]
        day = event_data["metadata"]["day"]
        
        # 确保features对象存在
        if "features" not in event_data:
            event_data["features"] = {}
        
        # 保存embedding到data/features/event/text/event_id.npy
        if embedding is not None:
            event_text_dir = os.path.join(self.features_dir, "event", "text")
            os.makedirs(event_text_dir, exist_ok=True)
            emb_path = os.path.join(event_text_dir, f"{event_id}.npy")
            np.save(emb_path, embedding)
            event_data["features"]["caption_embedding_path"] = emb_path
        
        # 保存vision feature到data/features/eventid/vision.npy
        if vision_feature is not None:
            event_feature_dir = os.path.join(self.features_dir, event_id)
            os.makedirs(event_feature_dir, exist_ok=True)
            vision_path = os.path.join(event_feature_dir, "vision.npy")
            np.save(vision_path, vision_feature)
            event_data["features"]["vision_feature_path"] = vision_path
        
        # 保存audio feature到data/features/eventid/audio.npy
        if audio_feature is not None:
            event_feature_dir = os.path.join(self.features_dir, event_id)
            os.makedirs(event_feature_dir, exist_ok=True)
            audio_path = os.path.join(event_feature_dir, "audio.npy")
            np.save(audio_path, audio_feature)
            event_data["features"]["audio_feature_path"] = audio_path
        
        # 保存JSON（标准格式）
        day_dir = os.path.join(self.events_dir, f"DAY{day}")
        os.makedirs(day_dir, exist_ok=True)
        
        json_path = os.path.join(day_dir, f"{event_id}.json")
        
        # 使用标准json.dump保存event数据
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(event_data, f, indent=2, ensure_ascii=False)
        
        return json_path
    
    def load_event(self, event_id: str, day: int) -> Dict:
        """
        加载event（不含特征向量）
        
        Args:
            event_id: Event ID
            day: 天数
        
        Returns:
            Event数据字典
        """
        json_path = os.path.join(
            self.events_dir,
            f"DAY{day}",
            f"{event_id}.json"
        )
        
        with open(json_path, 'r', encoding='utf-8') as f:
            event_data = json.load(f)
        
        return event_data
    
    def load_event_with_features(self, event_id: str, day: int) -> Dict:
        """
        加载event并加载所有特征向量
        
        Args:
            event_id: Event ID
            day: 天数
        
        Returns:
            Event数据字典（包含特征向量）
        """
        event_data = self.load_event(event_id, day)
        
        # 加载embedding
        if "features" in event_data and event_data["features"].get("caption_embedding_path"):
            emb_path = event_data["features"]["caption_embedding_path"]
            if os.path.exists(emb_path):
                event_data["caption_embedding"] = np.load(emb_path)
        
        # 加载vision feature
        if "features" in event_data and event_data["features"]["vision_feature_path"]:
            vision_path = event_data["features"]["vision_feature_path"]
            if os.path.exists(vision_path):
                event_data["vision_feature"] = np.load(vision_path)
        
        # 加载audio feature
        if "features" in event_data and event_data["features"]["audio_feature_path"]:
            audio_path = event_data["features"]["audio_feature_path"]
            if os.path.exists(audio_path):
                event_data["audio_feature"] = np.load(audio_path)
        
        return event_data
    
    def _custom_json_format(self, data: Dict) -> str:
        """
        将数据转换为自定义格式的JSON字符串
        
        功能：将speech_segments、interaction_segments等list[dict]字段压缩为单行
        原理：复用save_json_compact的逻辑，但返回字符串而非写入文件
        
        Args:
            data: 要格式化的数据字典
        
        Returns:
            格式化后的JSON字符串
        """
        # 指定需要压缩的字段（与save_event中的compact_fields保持一致）
        compact_fields = [
            "interaction_segments", 
            "speech_segments", 
            "interaction_language",
            "bboxes",
            "timesteps"
        ]
        
        # 复制data避免修改原始数据
        data_copy = json.loads(json.dumps(data))
        
        # 存储字段的单行JSON表示
        compact_values = {}
        counter = [0]
        
        # 递归处理所有嵌套结构（与save_json_compact相同的逻辑）
        def process_obj(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in compact_fields and isinstance(value, list):
                        # list[dict]类型：每个dict压缩为一行
                        if value and isinstance(value[0], dict):
                            unique_id = f"__COMPACT_LIST_DICT_{counter[0]}__"
                            counter[0] += 1
                            compact_items = [json.dumps(item, ensure_ascii=False, separators=(',', ':')) for item in value]
                            compact_values[unique_id] = {
                                'type': 'list_dict',
                                'items': compact_items
                            }
                            obj[key] = unique_id
                        # list[primitive]或list[list]：整个list压缩为一行
                        else:
                            unique_id = f"__COMPACT_LIST_{counter[0]}__"
                            counter[0] += 1
                            compact_str = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                            compact_values[unique_id] = {
                                'type': 'list',
                                'value': compact_str
                            }
                            obj[key] = unique_id
                    else:
                        process_obj(value)
            elif isinstance(obj, list):
                for item in obj:
                    process_obj(item)
        
        process_obj(data_copy)
        
        # 格式化整个JSON
        json_str = json.dumps(data_copy, ensure_ascii=False, indent=2)
        
        # 将所有标记替换回压缩的JSON字符串
        for unique_id, compact_data in compact_values.items():
            if compact_data['type'] == 'list':
                # 简单列表：直接替换
                json_str = json_str.replace(f'"{unique_id}"', compact_data['value'])
            elif compact_data['type'] == 'list_dict':
                # list[dict]：构建多行格式，每个dict一行
                items = compact_data['items']
                pattern = rf'^(\s*)"[^"]+": "' + re.escape(unique_id) + r'"'
                match = re.search(pattern, json_str, re.MULTILINE)
                if match:
                    key_indent = match.group(1)
                    item_indent = key_indent + "  "
                    items_str = ',\n' + item_indent
                    replacement = '[\n' + item_indent + items_str.join(items) + '\n' + key_indent + ']'
                    json_str = json_str.replace(f'"{unique_id}"', replacement)
        
        return json_str