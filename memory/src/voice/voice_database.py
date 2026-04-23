"""
VoiceprintDatabase - 全局声纹库管理模块

负责全局声纹库的加载、保存、查询、更新操作
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# 添加src目录到路径
src_dir = os.path.dirname(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 不再需要save_json_compact，直接使用标准json


class VoiceprintDatabase:
    """
    全局声纹库管理器
    
    职责：
    1. 加载/保存全局声纹库JSON文件
    2. 新增/更新/查询voice记录
    3. 管理EMA全局特征和历史特征
    """
    
    def __init__(self, database_path: str, embedding_dir: str, reset: bool = False):
        """
        初始化声纹库
        
        Args:
            database_path: 数据库JSON文件路径
            embedding_dir: embedding文件存储目录
            reset: 是否重置数据库（清空已有数据）
        """
        self.database_path = database_path
        self.embedding_dir = embedding_dir
        self.voices = []  # 所有voice记录列表
        
        # 确保目录存在
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        os.makedirs(embedding_dir, exist_ok=True)
        
        if reset:
            # 重置：清空数据库
            print(f"[DEBUG] 正在重置声纹库: {database_path}")
            self.voices = []
            self.save()
            print(f"[DEBUG] 声纹库已重置并保存")
        else:
            # 加载已有数据库
            self.load()
            print(f"[DEBUG] 已加载声纹库，当前包含 {len(self.voices)} 个voice记录")
    
    def load(self):
        """从文件加载数据库（直接加载list）"""
        if os.path.exists(self.database_path):
            with open(self.database_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    self.voices = json.loads(content)
                else:
                    # 文件为空，初始化为空列表
                    print(f"[DEBUG] 数据库文件为空，初始化为空列表")
                    self.voices = []
        else:
            self.voices = []
    
    def save(self):
        """保存数据库到文件（标准格式）"""
        # 使用标准json.dump保存，保持良好的可读性
        with open(self.database_path, 'w', encoding='utf-8') as f:
            json.dump(self.voices, f, indent=2, ensure_ascii=False)
    
    def get_all_voices(self) -> List[dict]:
        """获取所有voice记录"""
        return self.voices
    
    def get_voice_by_id(self, voice_id: str) -> Optional[dict]:
        """根据voice_id查询记录"""
        for voice in self.voices:
            if voice['voice_id'] == voice_id:
                return voice
        return None
    
    def add_voice(self, 
                  event_id: str,
                  event_speaker: str,
                  embedding: np.ndarray,
                  duration: float,
                  language_text: str,
                  descriptions: List[str],
                  event_time: str,
                  register_scores: Optional[Dict[str, float]] = None,
                  interaction_objects: Optional[List[Dict]] = None,
                  actions: Optional[List[str]] = None) -> str:
        """
        添加新voice记录
        
        Args:
            event_id: event标识
            event_speaker: event中的speaker标签（如"person1"）
            embedding: 声纹特征（192维）
            duration: 音频时长（秒）
            language_text: 语音文本内容
            descriptions: speaker描述列表
            event_time: event时间戳
            register_scores: 与已存在voice的相似度字典 {voice_id: match_score}
            interaction_objects: event中的interaction_object列表（包含global_name）
            actions: event中speaker的action列表
            
        Returns:
            voice_id: 新创建的voice_id
        """
        # 生成新voice_id
        voice_id = f"voice_{len(self.voices) + 1}"
        
        # 保存features到单个npy文件
        # 第0行：EMA全局特征（初始值为当前embedding）
        # 第1行：第一个历史特征（初始值也是当前embedding）
        feature_path = os.path.join(self.embedding_dir, f"{voice_id}_features.npy")
        features = np.vstack([embedding, embedding])  # shape: (2, 192)
        np.save(feature_path, features)
        
        # 创建voice记录
        voice_record = {
            "voice_id": voice_id,
            "name": f"person_{len(self.voices) + 1}",  # 默认名称
            "register_event": event_id,
            "register_time": event_time,
            "feature_path": feature_path,  # 单个npy文件路径
            "history_features": [
                {
                    "event_id": event_id,
                    "row_index": 1,  # 第1行（第0行是EMA）
                    "match_score": 1.0,  # 注册时match_score为1.0
                    "duration": duration  # 保存duration，用于后续筛选
                }
            ],
            "related_events": [
                {
                    "event_id": event_id,
                    "speaker_label": event_speaker,
                    "language": language_text,
                    "duration": duration,  # duration移到related_events中
                    "match_score": 1.0
                }
            ],
            "related_descriptions": [
                {
                    "event_id": event_id,
                    "descriptions": descriptions
                }
            ],
            # related_person: 记录speaker与interaction_object的关联
            # 格式: {"global_name": {"event_ids": [...], "count": N}}
            "related_person": {},
            # related_action: 记录speaker的action信息
            # 格式: [{"event_id": "xxx", "action": "..."}]
            "related_action": []
        }
        
        # 初始化related_person（如果有interaction_objects）
        if interaction_objects:
            for obj in interaction_objects:
                global_name = obj.get('global_name')
                if global_name:
                    voice_record['related_person'][global_name] = {
                        "event_ids": [event_id],
                        "count": 1
                    }
        
        # 初始化related_action（如果有actions）
        if actions:
            for action in actions:
                voice_record['related_action'].append({
                    "event_id": event_id,
                    "action": action
                })
        
        # 添加register_score字段（记录与已存在voice的相似度）
        if register_scores:
            voice_record["register_score"] = [
                {"voice_id": vid, "match_score": score}
                for vid, score in register_scores.items()
            ]
        
        self.voices.append(voice_record)
        self.save()
        
        return voice_id
    
    def update_voice(self,
                     voice_id: str,
                     event_id: str,
                     event_speaker: str,
                     embedding: np.ndarray,
                     duration: float,
                     language_text: str,
                     descriptions: List[str],
                     match_score: float,
                     ema_alpha: float,
                     ema_update_threshold: float,
                     max_history: int,
                     min_history_duration: float,
                     min_history_match_score: float,
                     interaction_objects: Optional[List[Dict]] = None,
                     actions: Optional[List[str]] = None):
        """
        更新已有voice记录
        
        更新逻辑：
        1. 更新EMA全局特征（如果match_score >= ema_update_threshold）
        2. 添加新的历史特征
        3. 筛选并保留符合条件的历史特征（最多max_history个）
        4. 更新related_events、related_descriptions、related_person和related_action
        
        Args:
            voice_id: 要更新的voice_id
            event_id: 当前event标识
            event_speaker: event中的speaker标签
            embedding: 新的声纹特征
            duration: 音频时长
            language_text: 语音文本
            descriptions: speaker描述
            match_score: 匹配分数
            ema_alpha: EMA系数
            ema_update_threshold: EMA更新阈值
            max_history: 最多保留的历史特征数
            min_history_duration: 历史特征最小时长要求
            min_history_match_score: 历史特征最小匹配分数要求
            interaction_objects: event中的interaction_object列表（包含global_name）
            actions: event中speaker的action列表
        """
        voice = self.get_voice_by_id(voice_id)
        if voice is None:
            return
        
        # 1. 读取当前features文件（第0行是EMA，第1-N行是历史features）
        feature_path = voice['feature_path']
        features = np.load(feature_path)  # shape: (N+1, 192)
        old_ema = features[0]  # 第0行：EMA
        history_embeddings = features[1:]  # 第1-N行：历史features
        
        # 2. 更新EMA全局特征（只有高质量匹配才更新）
        if match_score >= ema_update_threshold:
            # EMA更新：new_ema = alpha * new_emb + (1-alpha) * old_ema
            new_ema = ema_alpha * embedding + (1 - ema_alpha) * old_ema
            # 归一化：EMA更新后需要重新归一化
            norm = np.linalg.norm(new_ema)
            if norm > 0:
                new_ema = new_ema / norm
        else:
            new_ema = old_ema  # 不更新，保持原值
        
        # 3. 添加新的历史特征到history_features记录中
        new_history = {
            "event_id": event_id,
            "row_index": len(features),  # 新行的索引（当前是第N+1行）
            "match_score": match_score,
            "duration": duration  # 保存duration，用于后续筛选
        }
        voice['history_features'].append(new_history)
        
        # 4. 筛选历史特征：只有达到max_history上限时才做筛选
        if len(voice['history_features']) > max_history:
            # 筛选：保留满足duration和match_score条件的历史特征
            valid_histories = []
            for h in voice['history_features']:
                h_duration = h['duration']
                if h_duration >= min_history_duration and h['match_score'] >= min_history_match_score:
                    valid_histories.append(h)
            
            # 按match_score降序排序，只保留前max_history个
            valid_histories.sort(key=lambda x: x['match_score'], reverse=True)
            voice['history_features'] = valid_histories[:max_history]
        
        # 5. 重建features文件（第0行是新的EMA，第1-M行是筛选后的历史features）
        # 先收集要保留的历史embeddings
        valid_row_indices = [h['row_index'] for h in voice['history_features']]
        valid_embeddings = [history_embeddings[idx - 1] for idx in valid_row_indices if idx - 1 < len(history_embeddings)]
        # 添加新的embedding
        valid_embeddings.append(embedding)
        # 重新构建features：第0行是EMA，第1-M行是筛选后的历史features
        new_features = np.vstack([new_ema] + valid_embeddings)  # shape: (M+1, 192)
        np.save(feature_path, new_features)
        
        # 6. 更新history_features中的row_index（重新编号为1, 2, 3, ...）
        for i, h in enumerate(voice['history_features']):
            h['row_index'] = i + 1
        
        # 7. 更新related_events
        voice['related_events'].append({
            "event_id": event_id,
            "speaker_label": event_speaker,
            "language": language_text,
            "duration": duration,  # duration存在related_events中
            "match_score": match_score
        })
        
        # 8. 更新related_descriptions
        voice['related_descriptions'].append({
            "event_id": event_id,
            "descriptions": descriptions
        })
        
        # 9. 更新related_person（记录speaker与interaction_object的关联）
        # 确保related_person字段存在（兼容旧数据）
        if 'related_person' not in voice:
            voice['related_person'] = {}
        
        if interaction_objects:
            for obj in interaction_objects:
                global_name = obj.get('global_name')
                if global_name:
                    # 如果global_name已存在，更新event_ids和count
                    if global_name in voice['related_person']:
                        if event_id not in voice['related_person'][global_name]['event_ids']:
                            voice['related_person'][global_name]['event_ids'].append(event_id)
                            voice['related_person'][global_name]['count'] += 1
                    else:
                        # 新建global_name记录
                        voice['related_person'][global_name] = {
                            "event_ids": [event_id],
                            "count": 1
                        }
        
        # 10. 更新related_action（记录speaker的action信息）
        # 确保related_action字段存在（兼容旧数据）
        if 'related_action' not in voice:
            voice['related_action'] = []
        
        if actions:
            for action in actions:
                voice['related_action'].append({
                    "event_id": event_id,
                    "action": action
                })
        
        self.save()
    
    def get_next_voice_id(self) -> str:
        """获取下一个可用的voice_id"""
        return f"voice_{len(self.voices) + 1}"

