"""
音频处理工具模块

提供音频提取、裁剪、拼接等通用功能
"""

import os
from pathlib import Path
from typing import List, Dict
try:
    # moviepy 2.x
    from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips
except ImportError:
    # moviepy 1.x (fallback)
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips


def extract_audio_from_video(video_path: str, output_audio_path: str) -> str:
    """
    从视频文件中提取音频并保存为wav格式
    
    Args:
        video_path: 视频文件路径
        output_audio_path: 输出音频文件路径
        
    Returns:
        输出音频文件路径
    """
    video = VideoFileClip(video_path)
    audio = video.audio
    # moviepy 2.x 移除了 verbose 和 logger 参数
    audio.write_audiofile(output_audio_path, codec='pcm_s16le', logger=None)
    video.close()
    audio.close()
    return output_audio_path


def prepare_audio(input_path: str, temp_dir: str) -> str:
    """
    准备音频文件：如果是视频，提取音频到临时目录；如果是音频，直接返回路径
    
    Args:
        input_path: 输入文件路径（视频或音频）
        temp_dir: 临时目录
        
    Returns:
        音频文件路径
    """
    if input_path.endswith('.mp4'):
        os.makedirs(temp_dir, exist_ok=True)
        temp_audio = os.path.join(temp_dir, f"{os.path.basename(input_path)}.wav")
        extract_audio_from_video(input_path, temp_audio)
        return temp_audio
    return input_path


def merge_audio_segments_torch(audio_path: str, 
                               time_segments: List[tuple], 
                               output_path: str) -> str:
    """
    从音频文件中提取多个时间段并合并成一个音频文件
    
    使用torchaudio实现，高效且保持音频质量
    
    Args:
        audio_path: 原始音频文件路径
        time_segments: [(start_sec, end_sec), ...] 时间段列表（秒）
                      例如：[(1.0, 3.0), (10.0, 15.0)] 表示提取1-3秒和10-15秒
        output_path: 输出音频文件路径
        
    Returns:
        output_path: 输出文件路径
    
    示例:
        # 提取1-3秒和10-15秒的音频，合并为7秒的新音频
        merge_audio_segments_torch(
            'original.wav',
            [(1.0, 3.0), (10.0, 15.0)],
            'merged.wav'
        )
    
    注意:
        - 输出音频的采样率与输入保持一致
        - 如果输入是多声道，输出也保持多声道
        - 时间段按给定顺序拼接
    """
    import torch
    import soundfile as sf
    
    # 1. 加载完整音频（使用soundfile避免torchaudio版本问题）
    data, sr = sf.read(audio_path, dtype='float32')
    # 转换为torch tensor并确保shape为 (channels, samples)
    wav = torch.from_numpy(data.T) if data.ndim > 1 else torch.from_numpy(data).unsqueeze(0)
    
    # 2. 提取各个时间段的音频片段
    segments = []
    for start_sec, end_sec in time_segments:
        # 转换为采样点索引
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        
        # 提取片段（保持所有声道）
        segment = wav[:, start_sample:end_sample]
        segments.append(segment)
    
    # 3. 沿时间轴拼接所有片段
    merged_wav = torch.cat(segments, dim=1)
    
    # 4. 保存合并后的音频
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 转回numpy并保存（soundfile）
    merged_np = merged_wav.numpy().T if merged_wav.shape[0] > 1 else merged_wav.squeeze(0).numpy()
    sf.write(output_path, merged_np, sr)
    
    return output_path