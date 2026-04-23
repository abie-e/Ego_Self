"""
Voice模块 - 语音转文字、说话人分割、声纹识别

提供以下功能：
1. VoiceEmbedder: 声纹特征提取
"""

from .voice_embedder import VoiceEmbedder

__all__ = [
    'VoiceEmbedder',
]

