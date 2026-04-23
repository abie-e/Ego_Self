"""
VoiceprintProcessor - 声纹注册与识别主流程

核心功能：
1. 为语音片段分配全局说话人标识
2. 混合多向量+EMA方式维护声纹库
3. 支持ASR speaker标签的批量匹配
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from voice.voice_embedder import VoiceEmbedder
from voice.voice_database import VoiceprintDatabase
from voice.voice_matcher import VoiceMatcher
import utils.audio_utils as audio_utils
import utils.data_utils as data_utils


class VoiceprintProcessor:
    """
    声纹处理器，负责event级别的声纹匹配与全局声纹库维护
    
    主流程：
    1. 从event JSON中提取speaker片段
    2. 合并同一speaker的多个片段
    3. 提取声纹embedding
    4. 与全局库进行多对多匹配
    5. 更新event JSON的speaker字段为voice_id
    6. 更新全局声纹库
    """

    def __init__(self, config, embedder: Optional[VoiceEmbedder] = None):
        """
        初始化声纹处理器
        
        Args:
            config: 配置对象
            embedder: 声纹提取器（可选，未提供则自动创建）
        """
        self.config = config
        
        # 初始化声纹提取器
        if embedder is None:
            model_params = config.get_voiceprint_model_params()
            self.embedder = VoiceEmbedder(
                model_path=config.voiceprint_model_path,
                model_config=model_params,
                device=config.voiceprint_device
            )
        else:
            self.embedder = embedder
        
        # 初始化全局声纹库
        self.database = VoiceprintDatabase(
            database_path=config.voice_database_path,
            embedding_dir=config.voiceprint_embedding_dir,
            reset=config.voiceprint_reset_database
        )
        
        # 初始化匹配器
        self.matcher = VoiceMatcher(
            match_threshold=config.voiceprint_match_threshold,
            top_k=config.voiceprint_top_k,
            max_history=config.voiceprint_max_history_features
        )
        
        # 确保目录存在
        os.makedirs(config.voiceprint_merged_wav_dir, exist_ok=True)
    
    def process(self, 
                event_json_path: str,
                save_merged_audio: bool = True,
                write_back: bool = False) -> dict:
        """
        处理单个event的声纹匹配
        
        Args:
            event_json_path: event JSON文件路径
            save_merged_audio: 是否保存合并后的音频文件
            write_back: 是否将更新后的数据写回event JSON文件
            
        Returns:
            处理结果字典：{
                "event_id": "...",
                "updated_segments": [...],  # 更新后的interaction_language
                "voice_mapping": {"person1": "voice_1", ...}  # speaker → voice_id映射
            }
        """
        # 1. 读取event JSON
        with open(event_json_path, 'r', encoding='utf-8') as f:
            event_data = json.load(f)
        
        # 直接访问字段，如果不存在则报错
        event_id = event_data['event_id']
        event_time = event_data['attributes']['interaction_time']
        
        # 2. 提取并分组speaker片段
        speaker_segments = data_utils.filter_and_group_segments(
            event_json_path=event_json_path,
            min_duration=self.config.voiceprint_min_segment_duration,
            include_null_speakers=True
        )
        
        if len(speaker_segments) == 0:
            print(f"⚠️  {event_id}: 没有符合条件的语音片段")
            return {
                "event_id": event_id,
                "updated_segments": event_data['attributes']['interaction_language'],
                "voice_mapping": {}
            }
        
        # 3. 获取音频文件路径，直接访问
        video_path = event_data['video_path']
        if not video_path or not os.path.exists(video_path):
            print(f"⚠️  {event_id}: 视频文件不存在: {video_path}")
            return {
                "event_id": event_id,
                "updated_segments": event_data['attributes']['interaction_language'],
                "voice_mapping": {}
            }
        
        # 确保临时目录存在
        os.makedirs(self.config.temp_dir, exist_ok=True)
        
        # 提取音频（如果是视频文件）
        if video_path.endswith('.mp4') or video_path.endswith('.avi'):
            # 从视频中提取音频到临时目录
            audio_filename = f"{event_id}.wav"
            audio_path = os.path.join(self.config.temp_dir, audio_filename)
            if not os.path.exists(audio_path):
                print(f"📂 从视频提取音频: {video_path} -> {audio_path}")
                audio_path = audio_utils.prepare_audio(video_path, self.config.temp_dir)
        else:
            # 已经是音频文件
            audio_path = video_path
        
        # 4. 逐个处理speaker：提取embedding、匹配、更新database
        voice_mapping = {}  # {event_speaker: voice_id}
        speaker_merged_paths = {}  # {speaker: merged_audio_path}
        
        for speaker, time_segments in speaker_segments.items():
            # 4.1 合并该speaker的所有片段（存储在temp目录）
            merged_audio_path = os.path.join(
                self.config.temp_dir,
                f"{event_id}_{speaker}.wav"
            )
            
            audio_utils.merge_audio_segments_torch(
                    audio_path=audio_path,
                time_segments=time_segments,
                output_path=merged_audio_path
            )
            speaker_merged_paths[speaker] = merged_audio_path
            
            # 4.2 提取embedding
            embedding = self.embedder.process(merged_audio_path)
            
            # 4.3 计算总时长
            total_duration = sum(end - start for start, end in time_segments)
            
            # 4.4 提取language_text、descriptions、actions和person对应关系
            language_texts = []
            descriptions = []
            actions = []  # 收集该speaker的action信息
            # 直接访问interaction_language
            interaction_language = event_data['attributes']['interaction_language']
            
            # 收集该speaker的所有description（用于匹配person）
            speaker_descriptions = []
            for seg in interaction_language:
                seg_speaker = seg['speaker']
                if seg_speaker == speaker:
                    language_texts.append(seg['text'])
                    desc = seg['description']
                    if desc and desc not in descriptions:
                        descriptions.append(desc)
                    if desc and desc != "I":  # 排除"I"，只保留person描述
                        speaker_descriptions.append(desc)
            language_text = ''.join(language_texts)
            
            # 提取interaction_objects，直接访问
            all_objects = event_data['attributes']['interaction_object']
            
            # 通过description匹配找到该speaker对应的person实体
            # related_person记录：这个voice（speaker）对应哪个person实体
            # 匹配逻辑：如果speaker的description与某个person的description相同，则建立关联
            person_objects = []
            for obj in all_objects:
                # 直接访问global_name，跳过None（tracking失败的对象）
                global_name = obj.get('global_name')
                if global_name is None or not global_name.startswith('person_'):
                    continue  # 只处理person类型
                obj_desc = obj['description']
                # 如果该person的描述出现在speaker的descriptions中，说明这个speaker就是这个person
                if obj_desc and obj_desc in speaker_descriptions:
                    person_objects.append(obj)
            
            # 如果没有匹配到任何person（description都是"I"），则使用speaker name作为标识
            # 这种情况通常是"我"自己说的话
            if not person_objects:
                person_objects.append({
                    'global_name': speaker,  # 使用speaker name（如SPEAKER_00）作为标识
                    'description': 'I',
                    'action': ''
                })
            
            # 提取action信息（从所有interaction_object中提取，因为通常是"我"在操作物体）
            for obj in all_objects:
                action = obj['action']
                if action and action not in actions:
                    actions.append(action)
            
            # 4.5 与当前database进行匹配（每次都重新获取最新的database）
            database_voices = self.database.get_all_voices()
            match_result = self.matcher.match_single_speaker_to_database(
                speaker_embedding=embedding,
                database_voices=database_voices
            )
            
            # 4.6 根据匹配结果更新database
            if match_result is None:
                # 无匹配：创建新voice
                register_scores = {}
                if database_voices:
                    for db_voice in database_voices:
                        similarity = self.matcher._compute_similarity_with_voice(embedding, db_voice)
                        register_scores[db_voice['voice_id']] = round(similarity, 4)
                
                voice_id = self.database.add_voice(
                    event_id=event_id,
                    event_speaker=speaker,
                    embedding=embedding,
                    duration=total_duration,
                    language_text=language_text,
                    descriptions=descriptions,
                    event_time=event_time,
                    register_scores=register_scores if register_scores else None,
                    interaction_objects=person_objects,  # 只传递person类型的实体
                    actions=actions  # 传递actions
                )
                voice_mapping[speaker] = voice_id
            else:
                # 匹配成功：更新已有voice
                voice_id, match_score = match_result
                self.database.update_voice(
                    voice_id=voice_id,
                    event_id=event_id,
                    event_speaker=speaker,
                    embedding=embedding,
                    duration=total_duration,
                    language_text=language_text,
                    descriptions=descriptions,
                    match_score=match_score,
                    ema_alpha=self.config.voiceprint_ema_alpha,
                    ema_update_threshold=self.config.voiceprint_ema_update_threshold,
                    max_history=self.config.voiceprint_max_history_features,
                    min_history_duration=self.config.voiceprint_min_history_duration,
                    min_history_match_score=self.config.voiceprint_min_history_match_score,
                    interaction_objects=person_objects,  # 只传递person类型的实体
                    actions=actions  # 传递actions
                )
                voice_mapping[speaker] = voice_id
        
        # 7. 更新event JSON：保留原始speaker，新增global_voice字段
        # 新逻辑：不替换speaker（保留ASR原始输出），而是增加global_voice字段
        updated_segments = []
        for seg in interaction_language:
            seg_copy = seg.copy()
            old_speaker = seg['speaker']  # 直接访问
            if old_speaker in voice_mapping:
                # 保留原始speaker，新增global_voice字段
                seg_copy['global_voice'] = voice_mapping[old_speaker]
            updated_segments.append(seg_copy)
        
        # 8. 可选：写回event JSON
        if write_back:
            event_data['attributes']['interaction_language'] = updated_segments
            with open(event_json_path, 'w', encoding='utf-8') as f:
                json.dump(event_data, f, ensure_ascii=False, indent=2)
        
        # 9. 可选：清理临时音频文件
        if not save_merged_audio:
            for merged_path in speaker_merged_paths.values():
                if os.path.exists(merged_path):
                    os.remove(merged_path)
        
        return {
            "event_id": event_id,
            "updated_segments": updated_segments,
            "voice_mapping": voice_mapping
        }


def main():
    """命令行入口"""
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from config.config import Config

    parser = argparse.ArgumentParser(description="声纹匹配系统")
    parser.add_argument("event_json_path", type=str, help="Event JSON文件路径")
    parser.add_argument("--reset_database", action="store_true", 
                       help="是否重置全局声纹库（覆盖config配置）")
    parser.add_argument("--write_back", action="store_true", 
                       help="是否写回event JSON文件（默认否）")
    parser.add_argument("--no_save_merged", action="store_true",
                       help="不保存合并后的音频（默认保存）")
    parser.add_argument("--config", type=str, 
        default="/data/xuyuan/Egolife_env/ego-things/personalization/configs/config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config.from_yaml(args.config)
    
    # 覆盖reset_database参数
    if args.reset_database:
        config.voiceprint_reset_database = True
    
    # 初始化处理器
    processor = VoiceprintProcessor(config)
    
    # 执行处理
    result = processor.process(
        event_json_path=args.event_json_path,
        save_merged_audio=not args.no_save_merged,
        write_back=args.write_back
    )
    
    print(f"✓ 处理完成")
    print(f"  Event ID: {result['event_id']}")
    print(f"  匹配结果: {result['voice_mapping']}")
    
    # 打印全局库统计
    database = VoiceprintDatabase(
        database_path=config.voice_database_path,
        embedding_dir=config.voiceprint_embedding_dir,
        reset=False
    )
    voices = database.get_all_voices()
    print(f"  全局库: {len(voices)} 个voice")


if __name__ == "__main__":
    main()
