"""
GPT client.

Wraps OpenAI's chat.completions endpoint for GPT-4o-style multimodal calls
and the Whisper transcription endpoint.
"""

import json
import time
import base64
import tempfile
import os
import subprocess
from typing import List, Optional, Dict, Any

import cv2
from openai import OpenAI


class GPT4oClient:
    """
    GPT-4o API客户端
    
    支持多模态输入（文本+图像）和音频转录
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        config: Dict[str, Any],
        temp_dir: str = None
    ):
        """
        初始化GPT客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            config: 配置字典，包含model, max_retries, timeout等参数
            temp_dir: 临时文件目录（可选）
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=config["timeout"])
        
        self.model = config["model"]
        self.max_retries = config["max_retries"]
        self.timeout = config["timeout"]
        self.temperature = config["temperature"] if "temperature" in config else 0.0
        self.max_tokens = config["max_tokens"] if "max_tokens" in config else 10000
        
        # 图像处理参数
        self.resize_max_size = config["resize_max_size"]
        self.image_format = config["image_format"]
        self.image_quality = config["image_quality"]
        
        # 视频处理参数（用于支持视频输入的模型，如 gpt-4o, gpt-5）
        self.sample_fps = config["sample_fps"]
        self.video_max_size = config["video_max_size"] if "video_max_size" in config else None
        
        # 临时文件目录
        self.temp_dir = temp_dir
        
        # kwargs传递给API的额外参数
        self.kwargs = config["kwargs"]
    
    def generate_text(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system_message: Optional[str] = None
    ) -> str:
        """
        生成文本响应（支持多模态：文本+图像）

        Args:
            prompt: 提示词
            images: base64编码的图像列表（可选）
            system_message: 系统消息（可选）

        Returns:
            生成的文本
        """
        # 构建messages：先添加图像，最后添加文本prompt
        content = []
        
        # 添加所有图像（作为image_url）
        if images:
            for img_base64 in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
        
        # 添加文本prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # 构建完整messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": content})
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    **self.kwargs
                )
                
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise Exception(f"GPT API调用失败（{self.max_retries}次重试后）: {e}")

    def generate_json(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system_message: Optional[str] = None
    ) -> dict:
        """
        生成JSON响应（支持多模态：文本+图像）

        Args:
            prompt: 提示词
            images: base64编码的图像列表（可选）
            system_message: 系统消息（可选）

        Returns:
            解析后的JSON字典
        """
        response = self.generate_text(prompt, images, system_message)

        # 尝试提取JSON（可能包含markdown代码块）
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试从代码块中提取
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                raise
    
    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        kwargs_transcribe: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        使用Whisper模型将音频转为文字（segment级别的时间戳）
        
        Args:
            audio_path: 音频文件路径
            language: 音频语言（可选，如'zh'、'en'）
            prompt: 提示词，用于引导转录风格（可选）
            kwargs_transcribe: 传递给transcribe API的额外参数
        
        Returns:
            包含转录文本和时间戳信息的字典：
            {
                "text": "完整的转录文本",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 3.5,
                        "text": "第一句话",
                        "speaker": None
                    },
                    ...
                ],
                "language": "zh",
                "duration": 120.5
            }
        """
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                with open(audio_path, "rb") as audio_file:
                    # 构建请求参数
                    kwargs = {
                        "model": self.model,
                        "file": audio_file,
                    }
                    
                    if language:
                        kwargs["language"] = language
                    
                    if prompt:
                        kwargs["prompt"] = prompt
                    
                    # 合并额外的transcribe参数
                    if kwargs_transcribe:
                        kwargs.update(kwargs_transcribe)
                    
                    # 调用Whisper API
                    response = self.client.audio.transcriptions.create(**kwargs)
                print(f"response: {response}")
                # 解析响应 - verbose_json格式包含详细的segments信息
                result = {
                    "text": response.text,
                    "segments": [],
                    "language": getattr(response, "language", None),
                    "duration": getattr(response, "duration", None)
                }
                
                # 提取segments
                if hasattr(response, "segments") and response.segments:
                    for seg in response.segments:
                        result["segments"].append({
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text.strip(),
                            "speaker": getattr(seg, "speaker", None)
                        })
                
                return result
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Whisper API调用失败，{wait_time}秒后重试... 错误: {e}")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Whisper API调用失败（{self.max_retries}次重试后）: {e}")
    
    def _compress_video(self, video_path: str) -> str:
        """
        压缩视频：等比例缩放到指定尺寸，降低fps，保留音频，不修改原视频
        
        Args:
            video_path: 原视频路径
            
        Returns:
            压缩后的视频临时文件路径
        """
        # 获取原始视频大小
        original_size = os.path.getsize(video_path)
        original_size_mb = original_size / (1024 * 1024)
        
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        
        # 获取原始尺寸和fps
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()
        
        # 计算新尺寸（保持宽高比）
        max_dim = max(original_width, original_height)
        if max_dim <= self.video_max_size:
            # 无需缩放尺寸，但仍需降低fps
            new_width = original_width
            new_height = original_height
        else:
            scale = self.video_max_size / max_dim
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # 确保尺寸为偶数（H.264要求）
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
        
        print(f"[视频压缩] 原始: {original_width}x{original_height} @ {original_fps:.1f}fps, {original_size_mb:.2f}MB")
        print(f"[视频压缩] 目标: {new_width}x{new_height} @ {self.sample_fps}fps")
        
        # 创建临时输出文件（使用指定的temp_dir）
        if self.temp_dir:
            os.makedirs(self.temp_dir, exist_ok=True)
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4', dir=self.temp_dir)
        else:
            temp_fdemp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)
        
        # 使用ffmpeg压缩（保留音频，降低fps）
        # 注意：使用mpeg4编码器代替libx264，因为libx264可能未编译到ffmpeg中
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'scale={new_width}:{new_height}:force_original_aspect_ratio=decrease,pad={new_width}:{new_height}:-1:-1:color=black',  # 视频缩放+填充
            '-r', str(self.sample_fps),  # 降低fps（视频时长不变）
            '-c:v', 'mpeg4',  # 视频编码器（使用更通用的mpeg4）
            '-q:v', '5',  # 视频质量（2-31，越小质量越好）
            '-c:a', 'aac',  # 音频编码器
            '-b:a', '128k',  # 音频比特率
            '-y',  # 覆盖输出文件
            temp_path
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            
            # 获取压缩后的视频大小
            compressed_size = os.path.getsize(temp_path)
            compressed_size_mb = compressed_size / (1024 * 1024)
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            print(f"[视频压缩] 完成: {compressed_size_mb:.2f}MB (压缩率: {compression_ratio:.1f}%)")
            
        except subprocess.CalledProcessError as e:
            # 如果ffmpeg失败，删除临时文件并抛出异常
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise RuntimeError(f"Video compression failed: {e.stderr}")
        
        return temp_path

    def generate_json_with_video(
        self,
        prompt: str,
        video_path: str,
        audio_path: Optional[str] = None,
        speech_segments_json: Optional[str] = None,
        system_message: Optional[str] = None
    ) -> dict:
        """
        使用视频+音频生成JSON响应（支持视频输入的GPT模型专用方法）
        
        Args:
            prompt: 提示词（包含{speech_segments_json}占位符）
            video_path: 视频文件路径
            audio_path: 音频文件路径（可选）
            speech_segments_json: 语音片段JSON字符串（替换prompt中的占位符）
            system_message: 系统消息（可选）
            
        Returns:
            解析后的JSON字典
        """
        # 压缩视频
        compressed_video_path = self._compress_video(video_path)
        temp_video_created = (compressed_video_path != video_path)
        
        try:
            # 读取视频文件并转为base64
            with open(compressed_video_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode("utf-8")
            
            # 替换prompt中的占位符
            if speech_segments_json and "{speech_segments_json}" in prompt:
                prompt = prompt.replace("{speech_segments_json}", speech_segments_json)
            
            # 构建content（包含视频和音频）
            content = [
                {
                    "type": "input_video",
                    "input_video": {
                        "data": video_data,
                        "format": "mp4"
                    }
                }
            ]
            
            # 添加音频（如果提供）
            if audio_path:
                with open(audio_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode("utf-8")
                
                audio_format = audio_path.split(".")[-1].lower()
                if audio_format == "wav":
                    audio_format = "wav"
                elif audio_format in ["m4a", "aac"]:
                    audio_format = "aac"
                else:
                    audio_format = "mp3"  # 默认
                
                content.append({
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_data,
                        "format": audio_format
                    }
                })
            
            # 添加文本prompt
            content.append({
                "type": "text",
                "text": prompt
            })
            
            # 构建messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": content})
            
            # 调用GPT API（带重试）
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                        **self.kwargs
                    )
                    
                    # 解析JSON响应
                    response_text = response.choices[0].message.content.strip()
                    
                    # 提取JSON（可能包含markdown代码块）
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        if "```json" in response_text:
                            json_str = response_text.split("```json")[1].split("```")[0].strip()
                            return json.loads(json_str)
                        elif "```" in response_text:
                            json_str = response_text.split("```")[1].split("```")[0].strip()
                            return json.loads(json_str)
                        else:
                            raise
                
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                    else:
                        raise Exception(f"GPT视频标注调用失败（{self.max_retries}次重试后）: {e}")
        
        finally:
            # 清理临时视频文件
            if temp_video_created and os.path.exists(compressed_video_path):
                os.unlink(compressed_video_path)
