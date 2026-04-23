"""
数据处理工具模块

提供segment过滤、合并、格式转换等通用功能
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip


def dict_to_oneline(data: Dict) -> str:
    """
    将字典转换为单行字符串（紧凑JSON格式，无空格）
    
    用于在entity存储中将descriptions和related_events等信息存为单行
    
    Args:
        data: 输入字典
    
    Returns:
        单行字符串表示
    """
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))


def list_to_oneline(data: List) -> str:
    """
    将列表转换为单行字符串（紧凑JSON格式，无空格）
    
    用于在JSON保存时将bbox、timestep、score等列表字段存为单行
    
    Args:
        data: 输入列表
    
    Returns:
        单行字符串表示
    """
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))


def save_json_compact(data, filepath: str, compact_fields: List[str] = None):
    """
    【已废弃】此函数已不再使用，所有代码已改用标准json.dump
    
    保存JSON文件，指定字段以单行形式存储
    
    原理：先用标准json.dump格式化，然后通过正则表达式压缩指定字段
    
    Args:
        data: 要保存的数据（dict或list）
        filepath: 保存路径
        compact_fields: 需要单行存储的字段名列表
    
    注意：此函数保留仅为向后兼容，新代码请直接使用：
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    """
    import warnings
    warnings.warn(
        "save_json_compact已废弃，请使用标准json.dump代替",
        DeprecationWarning,
        stacklevel=2
    )
    import re
    
    # 如果没有指定compact_fields，直接正常保存
    if not compact_fields:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return
    
    # 先用标准格式化生成JSON字符串
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    
    # 对每个compact_field进行处理
    for field in compact_fields:
        # 模式1: list[dict] - 每个dict压缩为一行
        # 匹配: "field": [\n    {\n      "key": "value",\n      ...\n    },\n    ...\n  ]
        # 策略：找到字段对应的数组，识别每个dict的边界，将其压缩为单行
        
        # 使用更精确的匹配：找到"field": [ ... ]整个块
        pattern = rf'"' + re.escape(field) + r'":\s*\[\s*\n(.*?)\n\s*\]'
        
        def compress_array(match):
            array_content = match.group(1)  # 数组内所有内容
            
            # 检查是否是dict数组（包含{字符）
            if '{' in array_content:
                # list[dict]：使用递归括号匹配提取每个完整dict
                dicts = []
                depth = 0
                current_dict = []
                
                for char in array_content:
                    if char == '{':
                        if depth == 0:
                            current_dict = ['{']
                        else:
                            current_dict.append(char)
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        current_dict.append('}')
                        if depth == 0:
                            # 完成一个dict的提取
                            dict_str = ''.join(current_dict)
                            # 解析并重新序列化为紧凑格式
                            dict_obj = json.loads(dict_str)
                            compact_dict_str = json.dumps(dict_obj, ensure_ascii=False, separators=(',', ':'))
                            dicts.append(compact_dict_str)
                            current_dict = []
                    elif depth > 0:
                        current_dict.append(char)
                
                # 获取原始缩进
                indent_match = re.search(r'\n(\s+)\{', array_content)
                indent = indent_match.group(1) if indent_match else '    '
                
                # 重新组装：每个dict一行
                formatted_dicts = [f'{indent}{d}' for d in dicts]
                new_array_content = ',\n'.join(formatted_dicts)
                
                return f'"{field}": [\n{new_array_content}\n{indent[:-2]}]'
            else:
                # list[primitive]或list[list]：整个数组压缩为一行
                # 移除换行和多余空格（保留字符串内的空格）
                lines = array_content.split('\n')
                cleaned_lines = [line.strip() for line in lines if line.strip()]
                compact_content = ''.join(cleaned_lines)
                return f'"{field}": [{compact_content}]'
        
        json_str = re.sub(pattern, compress_array, json_str, flags=re.MULTILINE | re.DOTALL)
    
    # 写入文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_str)



def extract_event_id(input_path: str) -> str:
    """
    从文件名提取event_id
    
    Args:
        input_path: 文件路径
        
    Returns:
        event_id（去除扩展名的文件名）
    """
    return os.path.splitext(os.path.basename(input_path))[0]


def default_output_path(input_path: str, output_dir: str = None, suffix: str = "_asr") -> str:
    """
    生成默认输出路径
    
    Args:
        input_path: 输入文件路径
        output_dir: 输出目录（默认为输入文件所在目录）
        suffix: 输出文件名后缀（默认"_asr"）
        
    Returns:
        输出文件路径
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f"{base}{suffix}.json")


def _get_media_duration(event_data: dict) -> Optional[float]:
    """
    获取event对应的视频/音频时长（秒）
    
    优先级：
    1. 从 video_path 读取视频时长（使用 moviepy）
    2. 如果 video_path 不存在或读取失败，返回 None
    
    Args:
        event_data: event JSON数据（字典）
    
    Returns:
        视频/音频时长（秒），读取失败则返回 None
    """
    video_path = event_data.get('video_path')
    
    if not video_path or not os.path.exists(video_path):
        return None
    
    # 使用 moviepy 读取视频时长
    video = VideoFileClip(video_path)
    duration = video.duration
    video.close()
    
    return duration


def filter_asr_segments(asr_result: Dict[str, Any], 
                        min_duration: float = 0.5,
                        filter_empty_text: bool = True) -> Dict[str, Any]:
    """
    过滤ASR结果中的语音片段（基于时长和文本内容）
    
    过滤规则：
    1. 片段时长 < min_duration：过滤
    2. 文本为空字符串（如果 filter_empty_text=True）：过滤
    
    Args:
        asr_result: ASR结果字典，格式：
            {
                "event_id": "xxx",
                "audio_path": "xxx", 
                "duration": 120.5,
                "language": "zh",
                "speech_segments": [
                    {"start_time": 0.0, "end_time": 5.2, "speaker": "xxx", "text": "xxx"},
                    ...
                ]
            }
        min_duration: 最小片段时长（秒）
        filter_empty_text: 是否过滤空文本片段
    
    Returns:
        过滤后的ASR结果（格式与输入相同，但speech_segments已过滤）
    """
    original_count = len(asr_result["speech_segments"])
    filtered_segments = []
    
    for seg in asr_result["speech_segments"]:
        # 计算片段时长
        duration = seg["end_time"] - seg["start_time"]
        
        # 过滤条件1：时长过短
        if duration < min_duration:
            continue
        
        # 过滤条件2：文本为空（如果启用）
        if filter_empty_text and seg.get("text", "").strip() == "":
            continue
        
        # 通过过滤，保留该片段
        filtered_segments.append(seg)
    
    # 构造过滤后的结果
    filtered_result = {
        **asr_result,
        "speech_segments": filtered_segments
    }
    
    # 返回结果和过滤统计
    filtered_count = original_count - len(filtered_segments)
    return filtered_result, filtered_count


def filter_and_group_segments(event_json_path: str, 
                              min_duration: float,
                              include_null_speakers: bool = True) -> Dict[str, List[tuple]]:
    """
    从event JSON文件中过滤并按speaker分组音频片段
    
    读取event JSON中的attributes.interaction_language，按以下规则过滤和分组：
    1. 将时间戳限制在 [0, 视频时长] 范围内（避免ASR给出超出视频实际时长的时间戳）
    2. 过滤掉duration < min_duration的片段
    3. speaker不为null：按speaker字段分组
    4. speaker为null：每个片段独立分配临时ID（如 "null_0", "null_1"）
    
    Args:
        event_json_path: event JSON文件路径
        min_duration: 最小时长阈值（秒）
        include_null_speakers: 是否包含speaker为null的片段（默认True）
    
    Returns:
        speaker_segments: {
            'person1': [(start1, end1), (start2, end2), ...],  # speaker不为null
            'null_0': [(start3, end3)],  # speaker为null的独立片段
            'null_1': [(start4, end4)],  # speaker为null的独立片段
            ...
        }
        键为speaker标识，值为时间段列表（秒为单位的浮点数）
    
    示例:
        segments = filter_and_group_segments('event.json', min_duration=1.0)
        # 返回: {
        #     'person1': [(1.5, 3.2), (10.0, 15.5)],
        #     'person2': [(5.0, 8.0)],
        #     'null_0': [(12.0, 13.5)]
        # }
    """
    # 1. 加载JSON文件
    with open(event_json_path, 'r', encoding='utf-8') as f:
        event_data = json.load(f)
    
    # 2. 获取视频/音频时长（用于限制时间戳范围）
    video_duration = _get_media_duration(event_data)
    
    # 3. 提取segments列表（兼容两种结构）
    # 新结构: attributes.interaction_language (list)
    # 旧结构: interaction_language.segments (list)
    attributes = event_data.get('attributes', {})
    interaction_language = attributes.get('interaction_language', [])
    
    # 如果attributes.interaction_language不存在，尝试旧结构
    if not interaction_language:
        interaction_language_obj = event_data.get('interaction_language', {})
        interaction_language = interaction_language_obj.get('segments', [])
    
    # 4. 过滤并分组
    speaker_segments = {}
    null_counter = 0  # 用于为null speaker分配唯一ID
    
    for seg in interaction_language:
        # 获取时间范围（兼容两种字段名）
        start_raw = seg.get('start_time') or seg.get('start', 0)
        end_raw = seg.get('end_time') or seg.get('end', 0)
        
        # 限制时间戳在 [0, video_duration] 范围内
        # 原因：ASR 可能给出超出视频实际时长的时间戳（如视频30.04秒，但ASR给出32.06秒）
        if video_duration is not None:
            start = max(0, min(start_raw, video_duration))
            end = max(0, min(end_raw, video_duration))
        else:
            start = start_raw
            end = end_raw
        
        # 计算时长并过滤
        duration = end - start
        if duration < min_duration:
            continue
        
        # 获取speaker字段
        speaker = seg.get('speaker')
        
        if speaker is None:
            # speaker为null：每个片段分配唯一临时ID
            if include_null_speakers:
                temp_id = f"null_{null_counter}"
                speaker_segments[temp_id] = [(start, end)]
                null_counter += 1
            # 否则跳过null speaker
        else:
            # speaker不为null：按speaker分组
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((start, end))
    
    return speaker_segments


def get_audio_path_from_event(event_json_path: str, 
                               audio_root: str) -> str:
    """
    根据event JSON文件路径推导出对应的音频文件路径
    
    假设目录结构：
        events/DAY1/DAY1_11110000_evt.json  (event JSON)
        audios/DAY1/DAY1_11110000.wav       (对应音频)
    
    Args:
        event_json_path: event JSON文件路径
            例: /path/to/events/DAY1/DAY1_11110000_evt.json
        audio_root: 音频文件根目录
            例: /path/to/audios
    
    Returns:
        audio_path: 对应的音频文件路径
            例: /path/to/audios/DAY1/DAY1_11110000.wav
    
    文件命名规则:
        - JSON: {event_id}_evt.json
        - 音频: {event_id}.wav
    
    示例:
        audio_path = get_audio_path_from_event(
            '/data/events/DAY1/DAY1_11110000_evt.json',
            '/data/audios'
        )
        # 返回: '/data/audios/DAY1/DAY1_11110000.wav'
    """
    # 1. 提取JSON文件名（去除路径）
    json_filename = os.path.basename(event_json_path)
    # 例: DAY1_11110000_evt.json
    
    # 2. 提取event_id（去除_evt.json后缀）
    event_id = json_filename.replace('_evt.json', '')
    # 例: DAY1_11110000
    
    # 3. 提取日期目录（JSON文件所在的目录名）
    day_dir = os.path.basename(os.path.dirname(event_json_path))
    # 例: DAY1
    
    # 4. 构造音频文件路径
    audio_path = os.path.join(audio_root, day_dir, f'{event_id}.wav')
    # 例: /data/audios/DAY1/DAY1_11110000.wav
    
    return audio_path
