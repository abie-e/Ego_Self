"""
文本特征编码统一接口

支持的模型：
- CLIP text encoder
- Sentence-BERT
"""

import numpy as np
from typing import List, Union
import torch
from pathlib import Path


class TextEncoder:
    """文本编码器统一接口"""
    
    def __init__(self, model_name: str, config: dict):
        """
        初始化文本编码器
        
        Args:
            model_name: 模型名称 ("clip", "sentence_bert")
            config: 模型配置字典
        """
        self.model_name = model_name
        self.config = config
        self.model = None
        self.device = config.get('device', 'cuda')
        
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if self.model_name == "clip":
            self._load_clip()
        elif self.model_name == "sentence_bert":
            self._load_sentence_bert()
        else:
            raise ValueError(f"Unsupported text model: {self.model_name}")
    
    def _load_clip(self):
        """加载CLIP模型"""
        import clip
        
        model_name = self.config['model_name']
        use_local = self.config.get('use_local', False)
        
        if use_local:
            # 从本地加载
            models_dir = Path(__file__).parent.parent.parent / "models"
            model_path = models_dir / self.config['model_path']
            self.model, self.preprocess = clip.load(model_name, device=self.device, download_root=str(model_path))
        else:
            # 从网络下载
            self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        self.model.eval()
    
    def _load_sentence_bert(self):
        """加载Sentence-BERT模型"""
        from sentence_transformers import SentenceTransformer
        import os
        import glob
        
        use_local = self.config.get('use_local', False)
        
        if use_local and 'local_dir' in self.config:
            # 离线模式：直接从本地缓存路径加载，避免联网检查
            cache_folder = self.config['local_dir']
            model_name = self.config['model_name']
            
            # 构建Hugging Face缓存路径格式：models--organization--model_name
            # hf_cache_name = f"models--sentence-transformers--{model_name}"
            hf_cache_path = os.path.join(cache_folder, model_name)
            if os.path.exists(hf_cache_path):
                self.model = SentenceTransformer(hf_cache_path, device=self.device)
            else:
                raise FileNotFoundError(f"Model cache not found: {hf_cache_path}")           
            # # 查找最新的snapshot（通常只有一个）
            # if os.path.exists(hf_cache_path):
            #     snapshots = glob.glob(os.path.join(hf_cache_path, "*"))
            #     if snapshots:
            #         # 使用第一个snapshot（通常只有一个版本）
            #         local_model_path = snapshots[0]
            #         print(f"Loading Sentence-BERT from local path: {local_model_path}")
            #         self.model = SentenceTransformer(local_model_path, device=self.device)
            #     else:
            #         raise FileNotFoundError(f"No snapshot found in {hf_cache_path}")
            # else:
            #     raise FileNotFoundError(f"Model cache not found: {hf_cache_path}")
        elif 'local_dir' in self.config:
            # 使用本地缓存，但允许下载
            cache_folder = self.config['local_dir']
            self.model = SentenceTransformer(
                self.config['model_name'], 
                device=self.device,
                cache_folder=cache_folder
            )
        else:
            # 使用默认缓存位置
            self.model = SentenceTransformer(self.config['model_name'], device=self.device)
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        编码文本
        
        Args:
            texts: 文本或文本列表
        
        Returns:
            features: shape (N, D) 特征矩阵，如果输入单个文本则为 (1, D)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model_name == "clip":
            return self._encode_clip(texts)
        elif self.model_name == "sentence_bert":
            return self._encode_sentence_bert(texts)
    
    def _encode_clip(self, texts: List[str]) -> np.ndarray:
        """使用CLIP编码文本"""
        import clip
        
        # Tokenize
        text_tokens = clip.tokenize(texts).to(self.device)
        
        # Encode
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def _encode_sentence_bert(self, texts: List[str]) -> np.ndarray:
        """使用Sentence-BERT编码文本"""
        # Sentence-BERT自动归一化
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings


# 全局缓存，避免重复加载模型
_text_encoder_cache = {}


def get_text_encoder(model_name: str, config: dict) -> TextEncoder:
    """
    获取文本编码器（带缓存）
    
    Args:
        model_name: 模型名称
        config: 模型配置
    
    Returns:
        TextEncoder实例
    """
    cache_key = model_name
    if cache_key not in _text_encoder_cache:
        _text_encoder_cache[cache_key] = TextEncoder(model_name, config)
    return _text_encoder_cache[cache_key]


def encode_text(texts: Union[str, List[str]], model_name: str, config: dict) -> np.ndarray:
    """
    编码文本的便捷函数
    
    Args:
        texts: 文本或文本列表
        model_name: 模型名称 ("clip", "sentence_bert")
        config: 模型配置
    
    Returns:
        features: shape (N, D) 特征矩阵
    """
    encoder = get_text_encoder(model_name, config)
    return encoder.encode(texts)

