"""
实体存储管理：负责全局实体的创建、更新、保存
"""

import os
import json
from datetime import datetime
from typing import Dict, List
import sys

# 添加src目录到路径
src_dir = os.path.dirname(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 不再需要save_json_compact，直接使用标准json


class EntityStorage:
    """实体存储管理器：维护全局实体库"""
    
    def __init__(
        self, 
        entities_path: str,
        max_descriptions: int = 10,
        max_images: int = 10
    ):
        """
        初始化实体存储管理器
        
        Args:
            entities_path: 全局实体JSON文件路径
            max_descriptions: 每个实体保留的最大描述数量
            max_images: 每个实体保留的最大代表图像数量
        """
        self.entities_path = entities_path
        self.max_descriptions = max_descriptions
        self.max_images = max_images
        
        # 确保目录存在
        os.makedirs(os.path.dirname(entities_path), exist_ok=True)
        
        # 加载全局实体库
        self.entities = self.load()
    
    def load(self) -> Dict:
        """
        从JSON加载全局实体库
        
        Returns:
            全局实体字典，key为global_name
        """
        if not os.path.exists(self.entities_path):
            return {}
        
        with open(self.entities_path, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        
        return entities
    
    def save(self):
        """保存全局实体库到JSON（标准格式）"""
        # 使用标准json.dump保存，保持良好的可读性
        with open(self.entities_path, 'w', encoding='utf-8') as f:
            json.dump(self.entities, f, indent=2, ensure_ascii=False)
    
    def reset(self):
        """重置全局实体库（删除所有实体）"""
        self.entities = {}
        self.save()
    
    def generate_global_name(self, obj_name: str) -> str:
        """
        生成新的global_name
        
        Args:
            obj_name: 原始object名称（如human1, object1）
        
        Returns:
            新的global_name（如person_001, object_001）
        """
        # 区分人和物体
        if "human" in obj_name.lower() or "person" in obj_name.lower():
            prefix = "person"
        else:
            prefix = "object"
        
        # 找到已有的最大编号
        max_id = 0
        for gname in self.entities.keys():
            if gname.startswith(prefix + "_"):
                try:
                    num = int(gname.split("_")[1])
                    max_id = max(max_id, num)
                except:
                    pass
        
        # 生成新编号
        new_id = max_id + 1
        return f"{prefix}_{new_id:03d}"
    
    def create_entity(
        self, 
        obj_info: Dict, 
        event_id: str, 
        obj_crops_paths: List[str],
        match_score: Dict = None,
        unmatched_candidates: Dict = None
    ) -> str:
        """
        创建新的全局实体
        
        Args:
            obj_info: object信息字典（包含name, description, action等）
            event_id: 所属event ID
            obj_crops_paths: crop图像路径列表
            match_score: 匹配分数（可选）
            unmatched_candidates: 未匹配上的候选实体及其分数 {global_name: score}
        
        Returns:
            新创建的global_name
        """
        global_name = self.generate_global_name(obj_info["name"])
        
        # 构建description对象
        desc_obj = {
            "text": obj_info["description"],
            "action": obj_info["action"],
            "source_event": event_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 构建related_event对象（单行存储）
        # 确保match_score包含完整字段
        default_match_score = {
            "is_new": True,
            "text_similarity": 0.0, 
            "vision_similarity": 0.0,
            "final_score": 0.0,
            "unmatched_candidates": {}
        }
        if match_score:
            default_match_score.update(match_score)
        
        related_event_obj = {
            "event_id": event_id,
            "local_name": obj_info["name"],
            "match_score": default_match_score
        }
        
        # 构建新实体数据
        # descriptions和related_events存储为list[dict]，由save_json_compact统一处理格式
        new_entity = {
            "global_name": global_name,
            "descriptions": [desc_obj],  # 直接存储dict对象
            "related_events": [related_event_obj],  # 直接存储dict对象
            "representative_images": obj_crops_paths[:5],  # 初始保留最多5张
            "unmatched_candidates": unmatched_candidates or {},  # 存储未匹配上的候选实体score
            "metadata": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "match_count": 1
            }
        }
        
        self.entities[global_name] = new_entity
        print(f"  [新实体] 创建 {global_name} <- {obj_info['name']} (event: {event_id})")
        
        return global_name
    
    def update_entity(
        self, 
        global_name: str, 
        obj_info: Dict, 
        event_id: str, 
        obj_crops_paths: List[str],
        match_score: Dict
    ):
        """
        更新已有的全局实体
        
        Args:
            global_name: 全局实体名称
            obj_info: object信息字典（包含name, description, action等）
            event_id: 所属event ID
            obj_crops_paths: crop图像路径列表
            match_score: 匹配分数
        """
        entity = self.entities[global_name]
        
        # 构建新描述对象，直接存储dict
        new_desc_obj = {
            "text": obj_info["description"],
            "action": obj_info["action"],
            "source_event": event_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        entity["descriptions"].append(new_desc_obj)  # 直接append dict对象
        
        # 如果描述过多，保留最新的max_descriptions个
        if len(entity["descriptions"]) > self.max_descriptions:
            entity["descriptions"] = entity["descriptions"][-self.max_descriptions:]
        
        # 构建关联event对象，直接存储dict
        related_event_obj = {
            "event_id": event_id,
            "local_name": obj_info["name"],
            "match_score": match_score
        }
        entity["related_events"].append(related_event_obj)  # 直接append dict对象
        
        # 更新代表图像（添加新的crops，保持最多max_images张）
        entity["representative_images"].extend(obj_crops_paths)
        entity["representative_images"] = entity["representative_images"][:self.max_images]
        
        # 更新元数据
        entity["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entity["metadata"]["match_count"] = len(entity["related_events"])
        
        print(f"  [更新] {global_name} <- {obj_info['name']} "
              f"(event: {event_id}, score: {match_score['final_score']:.3f})")
    
    def get_entity(self, global_name: str) -> Dict:
        """获取指定实体"""
        return self.entities.get(global_name)
    
    def get_all_entities(self) -> Dict:
        """获取所有实体"""
        return self.entities

