"""
Embedding API客户端

功能：
1. 封装text-embedding-3-large API调用
2. 支持单文本和批量文本embedding
3. 实现重试机制
"""

import time
from typing import List, Union
from openai import OpenAI
import numpy as np


class EmbeddingClient:
    """Embedding API客户端"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "text-embedding-3-large",
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        初始化Embedding客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            max_retries: 最大重试次数
            timeout: 超时时间（秒）
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
    
    @classmethod
    def from_config(cls, config):
        """
        从Config对象创建EmbeddingClient
        
        Args:
            config: Config对象
        
        Returns:
            EmbeddingClient实例
        """
        # 获取text_embedding配置
        text_emb_config = config.get_text_embedding_config()
        
        return cls(
            api_key=text_emb_config["api_key"],
            base_url=text_emb_config["base_url"],
            model=text_emb_config["model"],
            max_retries=text_emb_config["max_retries"],
            timeout=text_emb_config["timeout"]
        )
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        获取单个文本的embedding
        
        Args:
            text: 输入文本
        
        Returns:
            embedding向量（3072维numpy数组）
        """
        # 重试机制
        for attempt in range(self.max_retries):
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                timeout=self.timeout
            )
            
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        
        # 所有重试都失败
        raise Exception(f"Embedding API调用失败，重试{self.max_retries}次后仍然失败")
    
    def get_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[np.ndarray]:
        """
        批量获取文本的embeddings
        
        Args:
            texts: 文本列表
            batch_size: 每批处理的文本数量
        
        Returns:
            embedding向量列表
        """
        embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 调用API
            for attempt in range(self.max_retries):
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts,
                    timeout=self.timeout
                )
                
                # 提取embeddings
                batch_embeddings = [
                    np.array(item.embedding, dtype=np.float32) 
                    for item in response.data
                ]
                embeddings.extend(batch_embeddings)
                break
        
        return embeddings


# 测试代码
if __name__ == "__main__":
    print("=== Embedding Client Test ===\n")
    print("注意：此测试需要真实的API密钥，请确保配置正确\n")
    
    # 从配置文件加载
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import Config
    
    config = Config.from_yaml("../../configs/config.yaml")
    
    # 获取embedding配置
    embedding_config = config.get_embedding_config()
    
    client = EmbeddingClient(
        api_key=embedding_config["api_key"],
        base_url=embedding_config["base_url"],
        model="text-embedding-3-large",  # 默认模型
        max_retries=3,
        timeout=30
    )
    
    print("✓ Embedding client initialized")
    
    # 测试单文本embedding
    print("\n[Test 1] Single text embedding:")
    text = "Hello world"
    embedding = client.get_embedding(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dtype: {embedding.dtype}")
    print(f"Embedding values (first 5): {embedding[:5]}")
    assert embedding.shape == (3072,)
    assert embedding.dtype == np.float32
    
    # 测试批量embedding
    print("\n[Test 2] Batch text embeddings:")
    texts = [
        "I am working on laptop",
        "I am talking with Alice",
        "I am eating lunch"
    ]
    embeddings = client.get_embeddings_batch(texts)
    print(f"Number of texts: {len(texts)}")
    print(f"Number of embeddings: {len(embeddings)}")
    assert len(embeddings) == len(texts)
    for i, emb in enumerate(embeddings):
        print(f"  Embedding {i}: shape={emb.shape}")
        assert emb.shape == (3072,)
    
    print("\n✓ All embedding client tests passed!")


