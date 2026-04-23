"""
媒体处理工具

提供第一视角视频在MVP阶段所需的最小能力：
1. 读取视频并按设定帧率采样
2. 对采样帧进行等比例缩放
3. 将帧转换为base64字符串，便于直接送入多模态模型
"""

from __future__ import annotations

import base64
import os
import tempfile
from io import BytesIO
from typing import List, Dict, Optional

import cv2
import numpy as np
from PIL import Image


class FrameExtractor:
    """视频帧提取器，负责从视频中按需采样"""

    def __init__(
        self,
        sample_fps: float = 1.0,
        resize_max_size: int = 1024,
        image_format: str = "JPEG",
        image_quality: int = 85,
    ) -> None:
        """
        初始化采样配置

        Args:
            sample_fps: 采样帧率（帧/秒）
            resize_max_size: 输出帧的最长边像素
            image_format: 保存图像的格式
            image_quality: JPEG质量（1-100）
        """

        self.sample_fps = sample_fps
        self.resize_max_size = resize_max_size
        self.image_format = image_format
        self.image_quality = image_quality

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """保持宽高比对图像等比例缩放"""

        height, width = image.shape[:2]

        if max(height, width) <= self.resize_max_size:
            return image

        if height > width:
            new_height = self.resize_max_size
            new_width = int(width * (self.resize_max_size / height))
        else:
            new_width = self.resize_max_size
            new_height = int(height * (self.resize_max_size / width))

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def extract_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        """
        按设定帧率对视频进行采样

        Returns:
            列表，元素包含：
            {
                "frame_index": 序号（按采样顺序递增）,
                "timestamp_sec": 距视频开始的秒数,
                "image": 采样后的BGR图像
            }
        """

        cap = cv2.VideoCapture(video_path)

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / self.sample_fps) if self.sample_fps > 0 else 1

        frames: List[Dict[str, object]] = []
        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_interval == 0 or frame_count % frame_interval == 0:
                resized_frame = self._resize_image(frame)
                frames.append(
                    {
                        "frame_index": extracted_count,
                        "timestamp_sec": frame_count / video_fps if video_fps else 0.0,
                        "image": resized_frame,
                    }
                )
                extracted_count += 1

                if max_frames is not None and extracted_count >= max_frames:
                    break

            frame_count += 1

        cap.release()
        return frames

    def frame_to_base64(self, frame: np.ndarray) -> str:
        """将BGR图像转换为base64字符串（不带前缀）"""

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        buffered = BytesIO()
        pil_image.save(buffered, format=self.image_format, quality=self.image_quality)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


