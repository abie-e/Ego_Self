"""
ASR（Automatic Speech Recognition）模块

支持三种实现方案：
1. GPT-4o: 使用Whisper API进行语音转文字（speaker=null）
2. Gemini: 使用Gemini 2.5 Pro进行语音转文字+说话人分割
3. WhisperX: 使用本地WhisperX进行语音转文字+说话人分割（faster-whisper + pyannote）

输出统一数据结构（结构1）：
{
    "event_id": "xxx",           # 事件ID
    "audio_path": "xxx",         # 音频文件路径
    "duration": 120.5,           # 音频时长（秒）
    "language": "zh",            # 语音语言
    "speech_segments": [         # 语音片段列表
        {
            "start_time": 0.0,   # 开始时间（秒）
            "end_time": 5.2,     # 结束时间（秒）
            "speaker": "SPEAKER_00" or null,  # 说话人（WhisperX/Gemini有值，GPT-4o为null）
            "text": "说话内容"
        },
        ...
    ]
}
"""

import os
import json
import sys
import time
import argparse
from typing import Dict, Any, Optional

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.gpt_client import GPT4oClient
from api.gemini_client import GeminiClient
from config import Config
from event.event_storage import EventStorage
import utils.audio_utils as audio_utils
import utils.data_utils as data_utils

import torch 
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class ASRProcessor:
    """ASR处理器基类"""
    
    def __init__(self, config: Config):
        """
        初始化ASR处理器
        
        Args:
            config: 配置对象
        """
        pass
    
    
    def process(
        self,
        audio_path: str,
        event_id: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理音频文件，返回统一的ASR结果
        
        Args:
            audio_path: 音频文件路径
            event_id: 事件ID（可选）
            language: 音频语言（可选，如'zh'、'en'）
        
        Returns:
            统一格式的ASR结果字典（结构1）
        """
        raise NotImplementedError


class GPT4oASR(ASRProcessor):
    """
    GPT-4o ASR实现
    使用Whisper API进行语音转文字，speaker设为null
    """
    
    def __init__(self, config: Config):
        """初始化GPT-4o ASR"""
        # 调用父类初始化（获取min_segment_duration）
        super().__init__(config)
        
        # 获取ASR配置（包括API配置和模型参数）
        asr_config = config.get_asr_config("gpt4o")
        
        self.client = GPT4oClient(
            api_key=asr_config["api_key"],
            base_url=asr_config["base_url"],
            config=asr_config
        )
        # 保存kwargs配置，用于传递给transcribe_audio
        self.kwargs_transcribe = asr_config["kwargs"]
    
    def process(
        self,
        audio_path: str,
        event_id: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        使用GPT-4o Whisper API处理音频
        
        Args:
            audio_path: 音频文件路径
            event_id: 事件ID（可选）
            language: 音频语言（可选）
        
        Returns:
            统一格式的ASR结果（speaker为null）
        """
        # 如果没有提供event_id，从文件名提取
        if not event_id:
            event_id = os.path.splitext(os.path.basename(audio_path))[0]
        
        # 调用Whisper API
        whisper_result = self.client.transcribe_audio(
            audio_path=audio_path,
            language=language,
            kwargs_transcribe=self.kwargs_transcribe
        )
        
        # 转换为统一格式
        result = {
            "event_id": event_id,
            "audio_path": audio_path,
            "duration": whisper_result["duration"],
            "language": whisper_result["language"],
            "speech_segments": []
        }
        # 转换segments格式，speaker设为null
        for seg in whisper_result["segments"]:
            result["speech_segments"].append({
                "start_time": seg["start"],
                "end_time": seg["end"],
                "speaker": seg["speaker"] if "speaker" in seg else None,  # GPT-4o不提供说话人信息
                "text": seg["text"]
            })
        
        return result


class GeminiASR(ASRProcessor):
    """
    Gemini ASR实现
    使用Gemini 2.5 Pro进行语音转文字+说话人分割
    """
    
    def __init__(self, config: Config):
        """初始化Gemini ASR"""
        # 调用父类初始化（获取min_segment_duration）
        super().__init__(config)
        
        # 获取ASR配置（包括API配置和模型参数）
        asr_config = config.get_asr_config("gemini")
        
        self.client = GeminiClient(
            api_key=asr_config["api_key"],
            base_url=asr_config["base_url"],
            config=asr_config,
            temp_dir=config.temp_dir
        )
        
        # 加载prompt
        self.system_message, self.user_prompt = config.get_prompt("voice_diarization", "gemini")
    
    def process(
        self,
        audio_path: str,
        event_id: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        使用Gemini API处理音频（ASR+说话人分割）
        
        Args:
            audio_path: 音频文件路径
            event_id: 事件ID（可选）
            language: 音频语言（可选，目前未使用）
        
        Returns:
            统一格式的ASR结果（包含说话人信息）
        """
        # 如果没有提供event_id，从文件名提取
        if not event_id:
            event_id = os.path.splitext(os.path.basename(audio_path))[0]
        
        # 调用Gemini API进行ASR+说话人分割
        gemini_result = self.client.transcribe_audio(
            audio_path=audio_path,
            language=language,
            system_message=self.system_message,
            user_prompt=self.user_prompt
        )
        
        # 转换为统一格式
        result = {
            "event_id": event_id,
            "audio_path": audio_path,
            "duration": None,  # 从最后一个segment推断
            "language": language,
            "speech_segments": []
        }
        
        # Gemini返回格式：[{"start_time": 5.0, "end_time": 8.5, "speaker": "[SPEAKER 1]", "text": "..."}]
        # 直接使用返回的格式，因为已经包含了 speaker 和正确的时间戳
        for seg in gemini_result:
            result["speech_segments"].append({
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "speaker": seg["speaker"],  # 直接使用 Gemini 返回的 speaker 标识
                "text": seg["text"]
            })
        
        # 推断时长
        if result["speech_segments"]:
            result["duration"] = result["speech_segments"][-1]["end_time"]
        
        return result


class WhisperXASR(ASRProcessor):
    """
    WhisperX ASR实现
    使用本地WhisperX进行语音转文字+说话人分割
    
    特点：
    - 使用faster-whisper进行快速转录（70x实时速度）
    - 使用wav2vec2进行单词级时间戳对齐
    - 使用pyannote-audio进行说话人分割
    - 完全本地运行，无需API调用
    """
    
    def __init__(self, config: Config):
        """初始化WhisperX ASR"""
        # 调用父类初始化（获取min_segment_duration和min_word_count）
        super().__init__(config)
        
        # 延迟导入whisperx（只在使用WhisperX时才导入）
        try:
            import whisperx
            from whisperx.diarize import DiarizationPipeline
            self.whisperx = whisperx
            self.DiarizationPipeline = DiarizationPipeline
        except ImportError as e:
            raise ImportError(
                "WhisperX is not installed. Please install it with: "
                "pip install git+https://github.com/m-bain/whisperX.git"
            ) from e
        
        # 获取ASR配置
        asr_config = config.get_asr_config("whisperx")
        
        # 设备和计算类型配置
        self.device = asr_config["device"]
        self.compute_type = asr_config["compute_type"]
        
        # 模型配置
        self.model_name = asr_config["model"]
        self.batch_size = asr_config["batch_size"]
        self.download_root = asr_config.get("download_root", None)
        
        # 本地模型路径配置（优先使用本地路径，避免联网下载）
        self.whisper_model_path = asr_config.get("whisper_model_path", None)
        self.diarization_model_path = asr_config.get("diarization_model_path", None)
        
        # 对齐（Alignment）配置
        self.enable_alignment = asr_config.get("enable_alignment", False)
        
        # 说话人分割配置
        self.enable_diarization = asr_config.get("enable_diarization", False)
        self.hf_token = asr_config.get("hf_token", None)
        self.min_speakers = asr_config.get("min_speakers", None)
        self.max_speakers = asr_config.get("max_speakers", None)
        
        # 设置模型下载路径环境变量（如果指定了）
        if self.download_root:
            os.environ['HF_HOME'] = self.download_root
            os.environ['TRANSFORMERS_CACHE'] = os.path.join(self.download_root, 'transformers')
        
        # 加载Whisper模型（用于转录）
        # 如果指定了本地路径，直接使用本地模型，否则从HuggingFace在线下载
        if self.whisper_model_path:
            # 使用本地模型
            self.whisper_model = whisperx.load_model(
                self.whisper_model_path,
                self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
                local_files_only=True
            )
        else:
            # 从 Hugging Face 在线下载模型
            self.whisper_model = whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
                local_files_only=False
            )
        
        # 对齐模型会根据语言动态加载（仅在启用alignment时）
        self.align_model = None
        self.align_metadata = None
        self.current_language = None
        
        # 说话人分割模型（仅在启用时加载）
        self.diarize_model = None
        if self.enable_diarization:
            # 使用在线下载模型（从HuggingFace）
            # 如果指定了本地路径且包含config.yaml，则使用本地模型
            if self.diarization_model_path and os.path.exists(os.path.join(self.diarization_model_path, "config.yaml")):
                diarization_model = os.path.join(self.diarization_model_path, "config.yaml")
            else:
                # 使用在线模型，直接指定HuggingFace模型ID
                # 使用 3.0 版本，这是一个稳定且兼容的版本
                diarization_model = "pyannote/speaker-diarization-3.0"

            self.diarize_model = self.DiarizationPipeline(
                model_name=diarization_model,
                use_auth_token=self.hf_token,
                device=self.device
            )
    
    def process(
        self,
        audio_path: str,
        event_id: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        使用WhisperX处理音频（ASR+对齐+说话人分割）
        
        Args:
            audio_path: 音频文件路径
            event_id: 事件ID（可选）
            language: 音频语言（可选，如'zh'、'en'）
        
        Returns:
            统一格式的ASR结果（包含说话人信息）
        """
        # 如果没有提供event_id，从文件名提取
        if not event_id:
            event_id = os.path.splitext(os.path.basename(audio_path))[0]
        
        # 1. 加载音频
        audio = self.whisperx.load_audio(audio_path)
        
        # 2. 转录（Transcribe）- 使用faster-whisper进行批量推理
        # 注意: 如果遇到BFloat16错误，设置vad_filter=False禁用pyannote VAD
        transcribe_result = self.whisper_model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=language
            # vad_filter=False  # 禁用VAD以避免pyannote的BFloat16问题
        )
        detected_language = transcribe_result["language"] if "language" in transcribe_result else language
        
        # 3. 对齐（Align）- 获取单词级精确时间戳（可选）
        if detected_language != self.current_language:
            # 动态加载对应语言的对齐模型
            self.align_model, self.align_metadata = self.whisperx.load_align_model(
                language_code=detected_language,
                device=self.device
            )
            self.current_language = detected_language
        
        aligned_result = self.whisperx.align(
            transcribe_result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            self.device,
            return_char_alignments=False
        )
        
        # 4. 说话人分割（Diarize）- 使用pyannote-audio（可选）
        if self.enable_diarization and self.diarize_model:
            # 获取说话人分割结果
            diarize_segments = self.diarize_model(
                audio_path,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )
            # 将说话人标签分配给转录文本
            final_result = self.whisperx.assign_word_speakers(
                diarize_segments,
                aligned_result
            )
        else:
            final_result = aligned_result
        
        # 5. 转换为统一格式并直接返回
        result = self._convert_to_standard_format(
            final_result,
            event_id,
            audio_path,
            detected_language
        )
        
        return result
    
    def _convert_to_standard_format(
        self,
        whisperx_result: Dict,
        event_id: str,
        audio_path: str,
        language: str
    ) -> Dict:
        """
        将WhisperX输出转换为统一格式
        
        WhisperX输出格式：
        {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.2,
                    "text": "说话内容",
                    "speaker": "SPEAKER_00"  # 如果启用了diarization
                }
            ]
        }
        
        统一格式：
        {
            "event_id": "xxx",
            "audio_path": "xxx",
            "duration": 120.5,
            "language": "zh",
            "speech_segments": [
                {
                    "start_time": 0.0,
                    "end_time": 5.2,
                    "speaker": "SPEAKER_00" or None,
                    "text": "说话内容"
                }
            ]
        }
        """
        segments = []
        for seg in whisperx_result["segments"]:
            segments.append({
                "start_time": seg["start"],
                "end_time": seg["end"],
                "speaker": seg["speaker"] if "speaker" in seg else None,  # WhisperX提供SPEAKER_XX格式
                "text": seg["text"]
            })
        
        # 推断音频时长
        duration = None
        if segments:
            duration = segments[-1]["end_time"]
        
        return {
            "event_id": event_id,
            "audio_path": audio_path,
            "duration": duration,
            "language": language,
            "speech_segments": segments
        }
    


def create_asr_processor(config: Config, asr_type: Optional[str] = None) -> ASRProcessor:
    """
    工厂函数：根据配置创建ASR处理器
    
    Args:
        config: 配置对象
        asr_type: ASR类型（可选，默认从config读取）
                  可选值: "gpt4o", "gemini", "whisperx"
    
    Returns:
        ASR处理器实例
    """
    # 如果未指定类型，从配置读取
    if asr_type is None:
        asr_type = config.asr_client
    
    # 根据类型创建对应的ASR处理器
    if asr_type == "whisperx":
        return WhisperXASR(config)
    elif asr_type == "gemini":
        return GeminiASR(config)
    elif asr_type == "gpt4o":
        return GPT4oASR(config)
    else:
        raise ValueError(f"Unknown ASR type: {asr_type}. Supported types: gpt4o, gemini, whisperx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR语音识别脚本 - 支持video/audio/json输入")
    parser.add_argument(
        "input",
        type=str,
        help="输入文件路径：video(.mp4) / audio(.mp3/.wav) / json(.json)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/data/xuyuan/Egolife_env/ego-things/personalization/configs/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出JSON路径（可选，默认与输入文件同目录）"
    )
    parser.add_argument(
        "--asr_type",
        type=str,
        default=None,
        choices=["gpt4o", "gemini", "whisperx"],
        help="ASR类型（可选，默认从配置文件读取）"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="音频语言（可选，如'zh'、'en'）"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config.from_yaml(args.config)
    
    # 判断输入类型
    if args.input.endswith('.json'):
        # JSON路径：直接加载已有ASR结果
        with open(args.input, 'r', encoding='utf-8') as f:
            result = json.load(f)
        output_path = args.output or args.input
    else:
        # 视频或音频路径：需要运行ASR
        audio_path = audio_utils.prepare_audio(args.input, config.temp_dir)
        event_id = data_utils.extract_event_id(args.input)
        
        # 创建ASR处理器并处理
        asr = create_asr_processor(config, asr_type=args.asr_type)
        result = asr.process(
            audio_path=audio_path,
            event_id=event_id,
            language=args.language
        )
        
        # 确定输出路径
        output_path = args.output or data_utils.default_output_path(args.input, config.voiceprint_asr_dir)
    
    # 过滤ASR结果（基于时长和文本内容）
    result, filtered_count = data_utils.filter_asr_segments(
        result, 
        min_duration=config.min_segment_duration,
        filter_empty_text=True
    )
    
    # 保存结果（使用自定义格式化，将speech_segments压缩为单行）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    event_storage = EventStorage(config.events_dir, config.features_dir)
    json_str = event_storage._custom_json_format(result)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    
    print(f"✓ ASR结果已保存: {output_path}")
    print(f"  识别到 {len(result['speech_segments'])} 个语音片段（已过滤 {filtered_count} 个短片段/空文本）")
    if "duration" in result and result["duration"]:
        print(f"  时长: {result['duration']:.2f}秒")

