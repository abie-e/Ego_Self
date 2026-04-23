"""
Event Caption Embedding提取模块

功能：
1. 读取event的caption字段
2. 调用text-embedding-3-large API生成embedding
3. 存储embedding到data/features/event/text/event_id.npy
4. 更新event JSON中的features.caption_embedding_path字段

Usage:
    # 方式1: 作为模块运行（推荐）
    python -m src.event.event_embedding --config configs/config.yaml --event_json data/events/DAY1/DAY1_11094208_evt.json
    
    # 方式2: 直接运行
    python src/event/event_embedding.py --config configs/config.yaml --event_json data/events/DAY1/DAY1_11094208_evt.json
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Optional

# 添加src目录到路径（支持直接运行）
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config import Config
    from api import EmbeddingClient
    from event.event_storage import EventStorage
else:
    # 作为模块导入时使用相对导入
    from config import Config
    from api import EmbeddingClient
    from .event_storage import EventStorage


class EventEmbeddingExtractor:
    """Event Caption Embedding提取器"""
    
    def __init__(self, config: Config):
        """
        初始化Embedding提取器
        
        Args:
            config: Config对象
        """
        self.config = config
        self.embedding_client = EmbeddingClient.from_config(config)
        self.event_storage = EventStorage(config.events_dir, config.features_dir)
    
    def extract_and_save(self, event_json_path: str) -> str:
        """
        提取event的caption embedding并保存
        
        Args:
            event_json_path: event JSON文件路径
        
        Returns:
            embedding文件路径
        """
        # 读取event JSON
        with open(event_json_path, 'r', encoding='utf-8') as f:
            event_data = json.load(f)
        
        event_id = event_data["event_id"]
        caption = event_data.get("caption", "")
        
        if not caption:
            raise ValueError(f"Event {event_id} has no caption")
        
        print(f"[Embedding] 提取event: {event_id}")
        print(f"  Caption: {caption[:100]}...")
        
        # 调用Embedding API
        embedding = self.embedding_client.get_embedding(caption)
        print(f"  Embedding shape: {embedding.shape}")
        
        # 保存embedding到data/features/event/text/event_id.npy
        event_text_dir = os.path.join(self.config.features_dir, "event", "text")
        os.makedirs(event_text_dir, exist_ok=True)
        emb_path = os.path.join(event_text_dir, f"{event_id}.npy")
        np.save(emb_path, embedding)
        print(f"  保存到: {emb_path}")
        
        # 确保features对象存在
        if "features" not in event_data:
            event_data["features"] = {}
        
        # 更新event JSON中的features.caption_embedding_path字段
        event_data["features"]["caption_embedding_path"] = emb_path
        # 使用EventStorage的自定义格式化保存JSON
        json_str = self.event_storage._custom_json_format(event_data)
        with open(event_json_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f"  更新JSON: {event_json_path}")
        
        return emb_path
    
    def extract_caption_embedding(self, caption: str) -> np.ndarray:
        """
        仅提取caption的embedding（供event_annotator使用）
        
        Args:
            caption: caption文本
        
        Returns:
            embedding向量
        """
        return self.embedding_client.get_embedding(caption)


# 命令行入口
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="提取event的caption embedding并保存到features目录"
    )
    parser.add_argument(
        "--event_json",
        type=str,
        default="/data/xuyuan/Egolife_env/ego-things/personalization/data/events/DAY1/DAY1_11094208_evt.json",
        help="event JSON文件路径（默认：DAY1_11094208_evt.json）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default='/data/xuyuan/Egolife_env/ego-things/personalization/configs/config.yaml',
        help="配置文件路径（默认：自动查找configs/config.yaml）"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs",
            "config.yaml"
        )
    config = Config.from_yaml(config_path)
    
    # 创建提取器
    extractor = EventEmbeddingExtractor(config)
    
    # 提取并保存
    print(f"\n{'='*60}")
    print("Event Caption Embedding 提取")
    print(f"{'='*60}\n")
    
    emb_path = extractor.extract_and_save(args.event_json)
    
    print(f"\n{'='*60}")
    print(f"完成！Embedding已保存到: {emb_path}")
    print(f"{'='*60}\n")

