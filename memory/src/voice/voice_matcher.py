"""
VoiceMatcher - 声纹匹配引擎

实现多对多匹配逻辑：event中的多个speaker与全局库中的多个voice进行匹配
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class VoiceMatcher:
    """
    声纹匹配器
    
    匹配逻辑：
    1. 对event中每个speaker的embedding，与全局库中每个voice的历史embeddings计算相似度
    2. 对每个voice的历史embeddings进行padding（不足max_history则用0向量填充）
    3. 计算query与每个历史embedding的余弦相似度
    4. 取非padding部分的topK最高相似度，计算均值
    5. 如果均值 >= match_threshold，则认为匹配成功
    """
    
    def __init__(self, match_threshold: float, top_k: int, max_history: int):
        """
        初始化匹配器
        
        Args:
            match_threshold: 匹配阈值（相似度>=此值才认为匹配）
            top_k: 取前K个最高相似度计算均值
            max_history: 历史特征的最大数量（用于padding）
        """
        self.match_threshold = match_threshold
        self.top_k = top_k
        self.max_history = max_history
    
    def match_single_speaker_to_database(self,
                                         speaker_embedding: np.ndarray,
                                         database_voices: List[dict]) -> Optional[Tuple[str, float]]:
        """
        单个speaker与全局库中所有voice匹配
        
        Args:
            speaker_embedding: speaker的embedding向量 (192,)
            database_voices: 全局库中所有voice记录列表
            
        Returns:
            匹配结果: ("voice_1", 0.87) 或 None（无匹配）
        """
        best_voice_id = None
        best_score = 0.0
        
        for voice in database_voices:
            # 计算query与该voice的相似度
            score = self._compute_similarity_with_voice(speaker_embedding, voice)
            
            # 更新最佳匹配
            if score >= self.match_threshold and score > best_score:
                best_score = score
                best_voice_id = voice['voice_id']
        
        # 返回匹配结果
        if best_voice_id is not None:
            return (best_voice_id, best_score)
        else:
            return None
    
    def match_event_to_database(self, 
                                 event_embeddings: Dict[str, np.ndarray],
                                 database_voices: List[dict]) -> Dict[str, Optional[Tuple[str, float]]]:
        """
        多对多匹配：event中每个speaker与全局库中所有voice匹配
        
        Args:
            event_embeddings: {
                "person1": embedding_array (192,),
                "person2": embedding_array (192,),
                "null_0": embedding_array (192,),
                ...
            }
            database_voices: 全局库中所有voice记录列表
            
        Returns:
            匹配结果: {
                "person1": ("voice_1", 0.87),  # (voice_id, match_score)
                "person2": ("voice_2", 0.92),
                "null_0": None,  # 无匹配
                ...
            }
        """
        match_results = {}
        
        for speaker, query_emb in event_embeddings.items():
            # 使用单speaker匹配方法
            match_result = self.match_single_speaker_to_database(query_emb, database_voices)
            match_results[speaker] = match_result
        
        return match_results
    
    def _compute_similarity_with_voice(self, query_emb: np.ndarray, voice: dict) -> float:
        """
        计算query与voice的相似度（padding + topK + mean）
        
        流程：
        1. 加载voice的所有历史embeddings
        2. 如果history_features为空，则使用EMA特征（第0行）
        3. 计算query与每个历史embedding的余弦相似度
        4. 取topK最高相似度（不足K则全部取）
        5. 返回topK的均值
        
        Args:
            query_emb: 查询embedding (192,)
            voice: voice记录（包含history_features）
            
        Returns:
            相似度分数（0-1之间）
        """
        # 检测输入 embedding 是否包含 NaN
        if np.isnan(query_emb).any():
            raise ValueError(f"❌ 输入 query_emb 包含 NaN！voice_id: {voice.get('voice_id')}")
        
        # 1. 从单个npy文件加载特征（第0行是EMA，第1-N行是历史特征）
        feature_path = voice.get('feature_path')
        features = np.load(feature_path)  # shape: (N+1, 192)
        
        history_features = voice.get('history_features', [])
        
        # 2. 提取用于匹配的embeddings
        history_embeddings = []
        if len(history_features) == 0:
            # 如果history_features为空（如voice_1初始状态），使用EMA特征（第0行）
            if len(features) > 0:
                history_embeddings.append(features[0])
        else:
            # 根据history_features中的row_index提取历史embeddings
            for h in history_features:
                row_idx = h['row_index']  # row_index从1开始
                if row_idx < len(features):
                    history_embeddings.append(features[row_idx])
        
        if len(history_embeddings) == 0:
            return 0.0
        
        # 3. 计算相似度矩阵
        # query_emb: (192,) → reshape为 (1, 192)
        # history_embeddings: [(192,), (192,), ...] → stack为 (N, 192)
        query_emb_2d = query_emb.reshape(1, -1)  # (1, 192)
        history_emb_matrix = np.stack(history_embeddings, axis=0)  # (N, 192)
        
        # 检测历史 embedding 是否包含 NaN（数据库中可能存储了异常数据）
        if np.isnan(history_emb_matrix).any():
            raise ValueError(f"❌ 数据库中的历史 embedding 包含 NaN！voice_id: {voice.get('voice_id')}, feature_path: {feature_path}")
        
        # cosine_similarity返回 (1, N)，即query与每个历史embedding的相似度
        similarities = cosine_similarity(query_emb_2d, history_emb_matrix)[0]  # (N,)
        
        # 4. 取topK最高相似度（不足K则全部取）
        k = min(self.top_k, len(similarities))
        top_k_indices = np.argsort(similarities)[-k:]  # 取最大的k个索引
        top_k_scores = similarities[top_k_indices]
        
        # 5. 返回topK均值
        return float(np.mean(top_k_scores))

