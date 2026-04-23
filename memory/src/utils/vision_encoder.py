"""
视觉特征编码统一接口

支持的模型：
- CLIP vision encoder
- DINOv2 vision encoder
"""

import numpy as np
from typing import List, Union
import torch
from pathlib import Path
from PIL import Image


class VisionEncoder:
    """视觉编码器统一接口"""
    
    def __init__(self, model_name: str, config: dict):
        """
        初始化视觉编码器
        
        Args:
            model_name: 模型名称 ("clip")
            config: 模型配置字典
        """
        self.model_name = model_name
        self.config = config
        self.model = None
        self.preprocess = None
        self.device = config.get('device', 'cuda')
        self.use_fp16 = config.get('use_fp16', False)
        
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if self.model_name == "clip":
            self._load_clip()
        elif self.model_name == "dinov2":
            self._load_dinov2()
        else:
            raise ValueError(f"Unsupported vision model: {self.model_name}")
    
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
    
    def _load_dinov2(self):
        """加载DINOv2模型"""
        model_name = self.config['model_name']
        use_local = self.config.get('use_local', False)
        
        if use_local:
            # 从本地加载
            local_dir = self.config.get('local_dir', None)
            if local_dir:
                model_path = Path(local_dir) / self.config['model_path']
                self.model = torch.hub.load(str(model_path), model_name, source='local')
            else:
                self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        else:
            # 从网络下载
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # DINOv2的预处理
        from torchvision import transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def encode(self, images: Union[np.ndarray, List[np.ndarray], Image.Image, List[Image.Image]]) -> np.ndarray:
        """
        编码图像
        
        Args:
            images: 图像或图像列表
                - numpy数组: (H, W, 3) BGR格式
                - PIL Image: RGB格式
        
        Returns:
            features: shape (N, D) 特征矩阵，如果输入单个图像则为 (1, D)
        """
        # 统一转换为列表
        if isinstance(images, (np.ndarray, Image.Image)):
            images = [images]
        
        if self.model_name == "clip":
            return self._encode_clip(images)
        elif self.model_name == "dinov2":
            return self._encode_dinov2(images)
    
    def _encode_clip(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """使用CLIP编码图像"""
        import clip
        
        # 预处理图像
        processed_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                # numpy BGR -> PIL RGB
                img = Image.fromarray(img[..., ::-1])
            processed_images.append(self.preprocess(img))
        
        # Stack and move to device
        image_batch = torch.stack(processed_images).to(self.device)
        
        # 使用fp16加速
        if self.use_fp16:
            image_batch = image_batch.half()
        
        # Encode
        with torch.no_grad():
            image_features = self.model.encode_image(image_batch)
            
            # 转回fp32进行归一化
            if self.use_fp16:
                image_features = image_features.float()
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def _encode_dinov2(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """使用DINOv2编码图像"""
        # 预处理图像
        processed_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                # numpy BGR -> PIL RGB
                img = Image.fromarray(img[..., ::-1])
            processed_images.append(self.preprocess(img))
        
        # Stack and move to device
        image_batch = torch.stack(processed_images).to(self.device)
        
        # 使用fp16加速
        if self.use_fp16:
            image_batch = image_batch.half()
            self.model = self.model.half()
        
        # Encode
        with torch.no_grad():
            image_features = self.model(image_batch)
            
            # 转回fp32进行归一化
            if self.use_fp16:
                image_features = image_features.float()
            
            # DINOv2输出已经是归一化的，但为了保险起见还是归一化一次
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()


# 全局缓存，避免重复加载模型
_vision_encoder_cache = {}


def get_vision_encoder(model_name: str, config: dict) -> VisionEncoder:
    """
    获取视觉编码器（带缓存）
    
    Args:
        model_name: 模型名称
        config: 模型配置
    
    Returns:
        VisionEncoder实例
    """
    cache_key = model_name
    if cache_key not in _vision_encoder_cache:
        _vision_encoder_cache[cache_key] = VisionEncoder(model_name, config)
    return _vision_encoder_cache[cache_key]


def encode_vision(images: Union[np.ndarray, List[np.ndarray], Image.Image, List[Image.Image]], 
                  model_name: str, config: dict) -> np.ndarray:
    """
    编码图像的便捷函数
    
    Args:
        images: 图像或图像列表
        model_name: 模型名称 ("clip")
        config: 模型配置
    
    Returns:
        features: shape (N, D) 特征矩阵
    """
    encoder = get_vision_encoder(model_name, config)
    return encoder.encode(images)

