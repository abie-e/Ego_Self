"""
Event标注主流程 - 多模态视频标注版本

功能：
1. 从视频中均匀采样帧，转换为base64格式
2. 将视频帧+文本prompt送入GPT-4o进行多模态标注
3. 一次GPT调用生成caption和attributes
4. 每个视频生成一个event
5. 预留特征/实体接口但默认写入None
"""

from datetime import datetime
from typing import List
import time  # 新增用于统计API调用时间

from config import Config
from api import GPT4oClient, EmbeddingClient
from api.gemini_client import GeminiClient
from utils import (
    generate_event_id,
    parse_video_filename,
    timestamp_to_datetime,
    FrameExtractor,
)
from .event_storage import EventStorage


class EventAnnotator:
    """Event标注器"""
    
    def __init__(self, config: Config):
        """初始化Event标注器"""
        self.config = config
        
        # 根据配置选择客户端
        self.client_type = config.caption_client.lower()
        
        # 获取对应客户端的完整配置（包括API配置和模型参数）
        client_config = config.get_caption_config(self.client_type)
        
        # 判断是否需要帧提取（只有gpt4o需要，其他模型都支持视频直接输入）
        self.needs_frame_extraction = (self.client_type == "gpt4o")
        
        if self.client_type == "gemini":
            # 初始化Gemini客户端
            self.client = GeminiClient(
                api_key=client_config["api_key"],
                base_url=client_config["base_url"],
                config=client_config,
                temp_dir=config.temp_dir
            )
            self.frame_extractor = None  # Gemini不需要帧提取器
        else:  # gpt4o, gpt5, 或其他GPT模型
            # 初始化GPT客户端
            self.client = GPT4oClient(
                api_key=client_config["api_key"],
                base_url=client_config["base_url"],
                config=client_config,
                temp_dir=config.temp_dir
            )
            # 只有gpt4o需要帧提取器
            if self.needs_frame_extraction:
                self.frame_extractor = FrameExtractor(
                    sample_fps=config.sample_fps,
                    max_frames=config.max_frames,
                    resize_max_size=client_config["resize_max_size"],
                    image_format=client_config["image_format"],
                    image_quality=client_config["image_quality"]
                )
            else:
                self.frame_extractor = None
        
        # 初始化存储
        self.event_storage = EventStorage(config.events_dir, config.features_dir)
        
        # 初始化Text Embedding客户端（如果配置开启）
        self.use_text_embedding = config.use_text_embedding
        if self.use_text_embedding:
            self.embedding_client = EmbeddingClient.from_config(config)
        else:
            self.embedding_client = None
        
        # 加载prompt（根据客户端类型选择对应的prompt）
        self.system_message, self.event_prompt = config.get_prompt("event_annotation", self.client_type)
        
        # 交互目标过滤阈值
        self.min_interaction_duration = config.min_interaction_duration
        
        self.verbose = config.verbose
    
    def _calculate_total_duration(self, interaction_segments: list) -> float:
        """
        计算interaction_segments的总交互时长
        
        Args:
            interaction_segments: 交互时间段列表 [{"start_time": float, "end_time": float}, ...]
        
        Returns:
            总时长（秒）
        """
        total_duration = 0.0
        for segment in interaction_segments:
            duration = segment.get("end_time", 0.0) - segment.get("start_time", 0.0)
            total_duration += duration
        return total_duration
    
    def annotate_video(self, video_path: str, speech_segments_json: dict = None) -> List[str]:
        """对单个视频生成一个event（提取视频帧并调用GPT-4o进行多模态标注）"""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"处理视频: {video_path}")
            print(f"{'='*60}")
        
        # 1. 解析视频文件名
        video_info = parse_video_filename(video_path)
        if self.verbose:
            print(f"\n[1/3] 视频信息: DAY{video_info['day']}, {video_info['person']}")
        
        
        # 3. 调用GPT-4o生成caption和attributes（多模态：视频帧+文本prompt）
        event_id = generate_event_id(video_info["timestamp"])

        if self.verbose:
            print(f"\n[3/3] 调用GPT-4o生成caption和attributes（多模态输入：视频帧+文本）...")

        # 新增：统计API调用时间
        api_start_time = time.time()
        
        # 根据是否需要帧提取来调用不同的方法
        if self.needs_frame_extraction:
            # GPT-4o：使用帧提取
            if self.verbose:
                print(f"  使用帧提取模式（{self.client_type}）")
            frame_base64_list = self.frame_extractor.extract_frames(video_path)
            response = self.client.generate_json(
                prompt=self.event_prompt,
                images=frame_base64_list,
                system_message=self.system_message
            )
        else:
            # Gemini / GPT-5 / 其他支持视频的模型：直接使用视频
            if self.verbose:
                print(f"  使用视频直接输入模式（{self.client_type}）")
            response = self.client.generate_json_with_video(
                prompt=self.event_prompt,
                video_path=video_path,
                speech_segments_json=speech_segments_json,
                system_message=self.system_message
            )
        
        api_end_time = time.time()
        api_elapsed = api_end_time - api_start_time
        print(f"[API] {self.client_type.upper()}调用耗时: {api_elapsed:.2f} 秒")
        
        # 调试：打印API返回的完整结构
        if self.verbose:
            print(f"\n[DEBUG] API返回的JSON结构:")
            import json
            print(json.dumps(response, ensure_ascii=False, indent=2)[:500] + "...")

        caption = response.get("caption", "")
        
        # 根据不同模型的返回格式处理
        # GPT-4o返回attributes结构，Gemini返回interaction_target结构
        if "attributes" in response:
            # GPT-4o格式
            attributes = response.get("attributes", {})
            raw_interaction_object = attributes.get("interaction_object", [])
            
            # 过滤掉交互时长不足的目标（与Gemini格式保持一致）
            interaction_object = []
            for obj in raw_interaction_object:
                interaction_segments = obj.get("interaction_segments", [])
                total_duration = self._calculate_total_duration(interaction_segments)
                
                if total_duration < self.min_interaction_duration:
                    if self.verbose:
                        print(f"  [过滤] {obj.get('name', 'unknown')}: 交互时长 {total_duration:.1f}s < {self.min_interaction_duration}s")
                    continue
                
                interaction_object.append(obj)
            
            interaction_action = attributes.get("interaction_action", "unknown")
            interaction_language = attributes.get("interaction_language")
            interaction_state = attributes.get("interaction_state", "")
            interaction_location = attributes.get("interaction_location", "")
        else:
            # Gemini/GPT-5格式：需要从interaction_target转换
            interaction_target = response.get("interaction_target", [])
            interaction_location = response.get("interaction_location", "")
            interaction_language = response.get("interaction_language", [])
            
            # 转换interaction_target为interaction_object格式，同时过滤掉交互时长不足的目标
            interaction_object = []
            for target in interaction_target:
                # 计算总交互时长（基于interaction_segments）
                interaction_segments = target.get("interaction_segments", [])
                total_duration = self._calculate_total_duration(interaction_segments)
                
                # 过滤：只保留交互时长 >= min_interaction_duration 的目标
                if total_duration < self.min_interaction_duration:
                    if self.verbose:
                        print(f"  [过滤] {target.get('name', 'unknown')}: 交互时长 {total_duration:.1f}s < {self.min_interaction_duration}s")
                    continue
                
                # 构建interaction_object（保持与prompt要求的格式一致）
                obj = {
                    "name": target.get("name", ""),
                    "description": target.get("description", ""),
                    "action": target.get("action", ""),
                    "location": target.get("location", "")  # 保留location字段
                }
                # 保留first_appearance_time字段（如果存在）
                if "first_appearance_time" in target:
                    obj["first_appearance_time"] = target["first_appearance_time"]
                # 保留interaction_segments字段
                if interaction_segments:
                    obj["interaction_segments"] = interaction_segments
                interaction_object.append(obj)
            
            # 提取主要动作（取第一个object的action作为主要动作）
            if interaction_object and interaction_object[0].get("action"):
                interaction_action = interaction_object[0]["action"]
            else:
                interaction_action = "unknown"
            
            interaction_state = ""

        interaction_time = timestamp_to_datetime(
            video_info["timestamp"],
            base_date=self.config.base_date,
        )
        
        # 过滤interaction_language：移除text和description都为空的片段
        if interaction_language:
            filtered_interaction_language = []
            for seg in interaction_language:
                text = seg.get('text', '').strip()
                description = seg.get('description', '').strip() if seg.get('description') else ''
                # 只保留text或description至少有一个非空的片段
                if text or description:
                    filtered_interaction_language.append(seg)
            interaction_language = filtered_interaction_language

        event_data = {
            "event_id": event_id,
            "timestamp": video_info["timestamp"],
            "video_path": video_path,
            "caption": caption,
            "attributes": {
                "interaction_object": interaction_object,
                "interaction_action": interaction_action,
                "interaction_language": interaction_language,
                "interaction_state": interaction_state,
                "interaction_location": interaction_location,
                "interaction_time": interaction_time,
            },
            # relations保留为空列表，后续判断事件关系时再写入
            "relations": [],
            # 预留实体接口
            "linked_entities": {
                "persons": [],
                "objects": [],
            },
            # 特征路径统一存储在features对象中
            "features": {
                "caption_embedding_path": None,
                "vision_feature_path": None,
                "audio_feature_path": None,
            },
            "metadata": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "person_name": video_info["person"],
                "day": video_info["day"],
            },
        }

        # 提取caption embedding（如果配置开启）
        caption_embedding = None
        if self.use_text_embedding and self.embedding_client:
            if self.verbose:
                print(f"\n[Embedding] 提取caption embedding...")
            caption_embedding = self.embedding_client.get_embedding(caption)
            if self.verbose:
                print(f"  Shape: {caption_embedding.shape}")

        # 保存event（包含embedding如果提取了）
        self.event_storage.save_event(event_data, embedding=caption_embedding)

        if self.verbose:
            print(f"\n{'='*60}")
            print("完成！生成 1 个event")
            print(f"{'='*60}\n")

        return [event_id]
