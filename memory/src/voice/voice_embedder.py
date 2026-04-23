"""
VoiceEmbedder - 声纹特征提取模块

基于3D-Speaker的ERes2NetV2模型，提取音频片段的声纹特征(embedding)
"""

import os
import sys

# 添加父目录到sys.path，支持单独运行此文件
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torchaudio
import numpy as np
from pathlib import Path

# 尝试相对导入（作为包导入时），失败则使用绝对导入（直接运行时）
try:
    from .speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2
    from .speakerlab.process.processor import FBank
except ImportError:
    from voice.speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2
    from voice.speakerlab.process.processor import FBank


class VoiceEmbedder:
    """
    声纹特征提取器
    
    使用ERes2NetV2模型从音频片段中提取192维声纹特征向量
    
    支持的模型配置:
        - pretrained_eres2netv2.ckpt: baseWidth=26, scale=2, expansion=2
        - pretrained_eres2netv2w24s4ep4.ckpt: baseWidth=24, scale=4, expansion=4
    
    使用示例:
        embedder = VoiceEmbedder(
            model_path='pretrained_eres2netv2.ckpt',
            model_config={'baseWidth': 26, 'scale': 2, 'expansion': 2},
            device='cuda'
        )
        embedding = embedder.process('audio.wav')  # shape: (192,)
    """
    
    def __init__(self, 
                 model_path: str,
                 model_config: dict,
                 device: str = 'cuda'):
        """
        初始化声纹提取器
        
        Args:
            model_path: 预训练模型checkpoint路径
            model_config: 模型结构配置，包含以下键：
                - baseWidth: int, 基础宽度参数（26或24）
                - scale: int, Res2Net的scale参数（2或4）
                - expansion: int, 通道扩展系数（2或4）
            device: 计算设备，'cuda' 或 'cpu'
        
        示例:
            # 加载默认模型
            embedder = VoiceEmbedder(
                model_path='/path/to/pretrained_eres2netv2.ckpt',
                model_config={'baseWidth': 26, 'scale': 2, 'expansion': 2}
            )
            
            # 加载w24s4ep4模型
            embedder = VoiceEmbedder(
                model_path='/path/to/pretrained_eres2netv2w24s4ep4.ckpt',
                model_config={'baseWidth': 24, 'scale': 4, 'expansion': 4}
            )
        """
        # 设置设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 初始化FBank特征提取器（与3D-Speaker保持一致）
        self.feature_extractor = FBank(
            n_mels=80,              # 80维Mel滤波器组
            sample_rate=16000,      # 16kHz采样率
            mean_nor=True           # 启用均值归一化
        )
        
        # 加载模型
        self.model = self._load_model(model_path, model_config)
    
    def _load_model(self, model_path: str, config: dict):
        """
        加载ERes2NetV2模型
        
        Args:
            model_path: checkpoint文件路径
            config: 模型配置参数
        
        Returns:
            model: 加载好的模型（已设为eval模式）
        """
        # 1. 实例化ERes2NetV2模型
        model = ERes2NetV2(
            feat_dim=80,                      # FBank特征维度
            embedding_size=192,               # 输出embedding维度
            baseWidth=config['baseWidth'],    # 基础宽度
            scale=config['scale'],            # Res2Net scale
            expansion=config['expansion'],    # 通道扩展系数
            pooling_func='TSTP',              # 时域统计池化
            two_emb_layer=False               # 单层embedding输出
        )
        
        # 2. 加载预训练权重
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        # 3. 移至指定设备并设为评估模式
        model.to(self.device)
        model.eval()
        
        return model
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """
        L2归一化embedding向量
        
        将embedding归一化为单位长度，用于余弦相似度计算
        参考m3-agent的实现，3D-Speaker虽然不显式归一化但使用余弦相似度时结果等价
        
        Args:
            embedding: 原始embedding向量，shape=(D,)
        
        Returns:
            normalized_embedding: 归一化后的embedding，shape=(D,)，L2范数为1
        """
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    
    def _load_wav(self, audio_path: str, target_sr: int = 16000):
        """
        加载并预处理音频文件
        
        处理步骤:
        1. 加载音频文件
        2. 重采样到目标采样率（如需要）
        3. 转换为单声道（如为多声道）
        
        Args:
            audio_path: 音频文件路径
            target_sr: 目标采样率，默认16000Hz
        
        Returns:
            wav: 预处理后的音频张量，shape=(1, samples)
        """
        # 1. 加载音频文件
        wav, sr = torchaudio.load(audio_path)
        
        # 2. 重采样到目标采样率（如果当前采样率不匹配）
        if sr != target_sr:
            # 使用sox_effects进行高质量重采样
            wav, sr = torchaudio.sox_effects.apply_effects_tensor(
                wav, sr, effects=[['rate', str(target_sr)]]
            )
        
        # 3. 转换为单声道（如果是多声道音频）
        if wav.shape[0] > 1:
            # 只保留第一个声道
            wav = wav[0, :].unsqueeze(0)
        
        return wav
    
    def process(self, audio_path: str) -> np.ndarray:
        """
        从音频文件中提取声纹embedding
        
        完整流程:
        1. 加载音频（自动处理采样率和声道）
        2. 提取FBank特征（80维，均值归一化）
        3. 模型前向传播（ERes2NetV2）
        4. 输出192维embedding向量
        
        Args:
            audio_path: 音频文件路径（.wav格式）
        
        Returns:
            embedding: 声纹特征向量，numpy数组，shape=(192,)
        
        示例:
            embedder = VoiceEmbedder(...)
            embedding = embedder.process('speaker1.wav')
            print(embedding.shape)  # (192,)
            
            # 计算两个音频的相似度
            emb1 = embedder.process('audio1.wav')
            emb2 = embedder.process('audio2.wav')
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        """
        # 1. 加载并预处理音频
        wav = self._load_wav(audio_path)  # shape: (1, samples)
        
        # 2. 提取FBank特征
        # FBank会自动处理：提取80维Mel特征 + 均值归一化
        feat = self.feature_extractor(wav)  # shape: (T, 80)
        # T是时间帧数，取决于音频长度（帧移10ms，帧长25ms）
        
        # 3. 增加batch维度并移至计算设备
        feat = feat.unsqueeze(0).to(self.device)  # shape: (1, T, 80)
        
        # 4. 模型推理（禁用梯度计算以节省内存）
        with torch.no_grad():
            embedding = self.model(feat)  # shape: (1, 192)
        
        # 5. 转换为numpy数组并移除batch维度
        # 注意：先转为float32，因为numpy不支持BFloat16
        embedding = embedding.detach().squeeze(0).float().cpu().numpy()  # shape: (192,)
        
        # 6. L2归一化（参考m3-agent实现，确保余弦相似度计算的稳定性）
        embedding = self._normalize(embedding)
        
        # 7. 检测 NaN：音频太短或静音时 pooling layer 可能产生 NaN
        if np.isnan(embedding).any():
            raise ValueError(f"❌ Embedding 包含 NaN！可能原因：音频太短（< 100ms）或静音。音频路径: {audio_path}")
        
        return embedding