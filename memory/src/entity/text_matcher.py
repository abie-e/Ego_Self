"""
文本匹配器：计算object描述与全局实体的文本相似度
"""

import numpy as np
import json
from typing import Dict, List, Tuple
import sys
import os

# 添加src目录到路径
src_dir = os.path.dirname(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.text_encoder import get_text_encoder


class TextMatcher:
    """文本匹配器：使用句子嵌入计算描述相似度"""
    
    def __init__(self, config):
        """
        初始化文本匹配器
        
        Args:
            config: 配置对象，包含models.sentence_bert配置
        """
        self.config = config
        
        # 从原始配置字典获取sentence_bert配置
        text_model_config = config._config["entity"]["models"]["sentence_bert"].copy()
        
        # 展开local_dir中的路径变量（如 ${paths.models_root}）
        if "local_dir" in text_model_config:
            text_model_config["local_dir"] = config._expand_path(text_model_config["local_dir"])
        
        # 初始化文本编码器（使用缓存，避免重复加载）
        self.text_encoder = get_text_encoder("sentence_bert", text_model_config)
        
        # 从原始配置字典获取text2text匹配参数
        matching_config = config._config["entity"]["global_matching"]["text2text"]
        self.top_k = matching_config["top_k"]
        self.threshold = matching_config["threshold"]  # 文本相似度阈值
        self.global_desc_count = matching_config.get("global_desc_count", 5)
    
    def match(
        self, 
        new_description: str, 
        global_entities: Dict
    ) -> List[Tuple[str, float]]:
        """
        计算新描述与全局实体的文本相似度
        
        Args:
            new_description: 新object的描述文本
            global_entities: 全局实体字典 {global_name: entity_data}
        
        Returns:
            候选列表: [(global_name, similarity_score), ...] 按相似度降序
                     最多返回top_k个
        """
        if not global_entities:
            return []
        
        # 编码新描述
        new_embedding = self.text_encoder.encode(new_description)  # shape: (1, D)
        
        # 收集所有全局实体的描述
        entity_names = []
        entity_descriptions = []
        
        for global_name, entity in global_entities.items():
            # 获取最近N个描述
            descriptions = entity["descriptions"][-self.global_desc_count:]
            for desc_item in descriptions:
                # 如果desc_item是字符串（从JSON加载的单行格式），解析为字典
                if isinstance(desc_item, str):
                    desc_item = json.loads(desc_item)
                entity_names.append(global_name)
                entity_descriptions.append(desc_item["text"])
        
        if not entity_descriptions:
            return []
        
        # 批量编码全局描述
        entity_embeddings = self.text_encoder.encode(entity_descriptions)  # shape: (M, D)
        
        # 计算余弦相似度（embedding已归一化，直接点积）
        similarities = np.dot(entity_embeddings, new_embedding.T).flatten()  # shape: (M,)
        
        # 按global_name聚合相似度（取最大值）
        name_to_max_sim = {}
        for name, sim in zip(entity_names, similarities):
            if name not in name_to_max_sim or sim > name_to_max_sim[name]:
                name_to_max_sim[name] = float(sim)
        
        # 过滤：只保留相似度大于threshold的候选
        filtered_candidates = [
            (name, sim) for name, sim in name_to_max_sim.items() 
            if sim > self.threshold
        ]
        
        # 打印过滤前的top-5候选（用于调试）
        all_sorted = sorted(name_to_max_sim.items(), key=lambda x: x[1], reverse=True)
        if len(all_sorted) > 0:
            print(f"    [文本匹配详情] 所有实体的最高相似度（Top-5）：")
            for name, sim in all_sorted[:5]:
                status = "✓" if sim > self.threshold else "✗"
                print(f"      {status} {name}: {sim:.3f}")
        
        # 排序并返回top_k
        sorted_candidates = sorted(
            filtered_candidates, 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_candidates[:self.top_k]
    
    def match_all(
        self, 
        new_description: str, 
        global_entities: Dict
    ) -> Dict[str, float]:
        """
        计算新描述与所有全局实体的文本相似度（不过滤阈值）
        用于记录 unmatched_candidates
        
        Args:
            new_description: 新object的描述文本
            global_entities: 全局实体字典 {global_name: entity_data}
        
        Returns:
            所有实体的相似度字典: {global_name: similarity_score}
        """
        if not global_entities:
            return {}
        
        # 编码新描述
        new_embedding = self.text_encoder.encode(new_description)
        
        # 收集所有全局实体的描述
        entity_names = []
        entity_descriptions = []
        
        for global_name, entity in global_entities.items():
            # 获取最近N个描述
            descriptions = entity["descriptions"][-self.global_desc_count:]
            for desc_item in descriptions:
                if isinstance(desc_item, str):
                    desc_item = json.loads(desc_item)
                entity_names.append(global_name)
                entity_descriptions.append(desc_item["text"])
        
        if not entity_descriptions:
            return {}
        
        # 批量编码全局描述
        entity_embeddings = self.text_encoder.encode(entity_descriptions)
        
        # 计算余弦相似度
        similarities = np.dot(entity_embeddings, new_embedding.T).flatten()
        
        # 按global_name聚合相似度（取最大值）
        name_to_max_sim = {}
        for name, sim in zip(entity_names, similarities):
            if name not in name_to_max_sim or sim > name_to_max_sim[name]:
                name_to_max_sim[name] = float(sim)
        
        return name_to_max_sim