"""
Grounded-SAM-2 封装类

提供物体检测和追踪的统一接口，支持：
1. Box Prompt模式：使用bbox进行追踪（主要模式）
2. 初始帧检测：在多帧中检测物体，选择最高置信度
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple

# 添加Grounded-SAM-2子模块路径
GROUNDED_SAM2_PATH = os.path.join(
    os.path.dirname(__file__), 
    "../../submodules/Grounded-SAM-2"
)
sys.path.insert(0, GROUNDED_SAM2_PATH)

# 导入sam2以触发Hydra初始化
import sam2
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class GroundedSAM2Wrapper:
    """封装Grounded-SAM-2的检测和追踪功能"""
    
    def __init__(self, config: dict):
        """
        初始化模型
        
        Args:
            config: 配置字典，包含grounding_dino和sam2配置
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 配置自动精度
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # 初始化Grounding DINO
        # 支持本地模型路径或Hugging Face model_id
        model_id = config["grounding_dino"]["model_id"]
        
        # 如果是相对路径，转换为绝对路径（相对于项目根目录）
        # 使用realpath规范化路径，避免符号链接和相对路径（如../）导致的问题
        if not model_id.startswith("IDEA-Research/") and not os.path.isabs(model_id):
            current_file = os.path.realpath(__file__)  # 获取当前文件的真实绝对路径
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # 项目根目录
            model_id = os.path.join(base_dir, model_id)
        
        # 加载模型
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device)
        
        # 初始化SAM 2
        # 获取绝对路径（相对于项目根目录）
        # 使用realpath规范化路径，避免符号链接和相对路径（如../）导致的问题
        current_file = os.path.realpath(__file__)  # 获取当前文件的真实绝对路径
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # 项目根目录
        sam2_checkpoint = os.path.join(base_dir, config["sam2"]["checkpoint"])
        
        # 确保checkpoint存在
        if not os.path.exists(sam2_checkpoint):
            raise FileNotFoundError(f"SAM2 checkpoint未找到: {sam2_checkpoint}")
        
        # model_cfg应该是相对于sam2配置目录的路径名（hydra格式）
        # 例如: "sam2.1/sam2.1_hiera_l" 而不是完整路径
        model_cfg = config["sam2"]["model_cfg"]
        
        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
        
        # 检测阈值
        self.box_threshold = config["grounding_dino"]["box_threshold"]
        self.text_threshold = config["grounding_dino"]["text_threshold"]
    
    def detect_objects_in_frame(
        self, 
        image: np.ndarray, 
        text_prompt: str
    ) -> Dict:
        """
        在单帧中检测物体
        
        Args:
            image: BGR图像 (H, W, 3)
            text_prompt: 物体描述文本（需要小写并以.结尾）
            
        Returns:
            {
                "boxes": ndarray (N, 4), # [x1, y1, x2, y2]格式
                "scores": ndarray (N,),
                "labels": list[str]
            }
        """
        # 转换为RGB并创建PIL Image
        image_rgb = image[:, :, ::-1]  # BGR to RGB
        pil_image = Image.fromarray(image_rgb)
        
        # 确保文本格式正确
        if not text_prompt.endswith('.'):
            text_prompt = text_prompt + '.'
        text_prompt = text_prompt.lower()
        
        # 运行Grounding DINO
        inputs = self.processor(images=pil_image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        
        # 后处理
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]]
        )
        
        if len(results) == 0 or len(results[0]["boxes"]) == 0:
            return {
                "boxes": np.array([]),
                "scores": np.array([]),
                "labels": []
            }
        
        return {
            "boxes": results[0]["boxes"].cpu().numpy(),
            "scores": results[0]["scores"].cpu().numpy(),
            "labels": results[0]["labels"]
        }
    
    def detect_objects_in_frames_batch(
        self,
        images: List[np.ndarray],
        text_prompt: str,
        batch_size: int = 8
    ) -> List[Dict]:
        """
        批量在多帧中检测物体（使用同一个文本prompt）
        
        Args:
            images: BGR图像列表，每个为 (H, W, 3)
            text_prompt: 物体描述文本（所有图像使用相同prompt）
            batch_size: 批处理大小，避免GPU内存溢出
            
        Returns:
            检测结果列表，每个元素为:
            {
                "boxes": ndarray (N, 4), # [x1, y1, x2, y2]格式
                "scores": ndarray (N,),
                "labels": list[str]
            }
        """
        if not images:
            return []
        
        # 确保文本格式正确
        if not text_prompt.endswith('.'):
            text_prompt = text_prompt + '.'
        text_prompt = text_prompt.lower()
        
        all_results = []
        
        # 分批处理图像
        for batch_start in range(0, len(images), batch_size):
            batch_images = images[batch_start:batch_start + batch_size]
            
            # 转换为PIL Image列表
            pil_images = []
            target_sizes = []
            for img in batch_images:
                img_rgb = img[:, :, ::-1]  # BGR to RGB
                pil_img = Image.fromarray(img_rgb)
                pil_images.append(pil_img)
                target_sizes.append(pil_img.size[::-1])  # (height, width)
            
            # 批量处理：为每张图像复制文本prompt
            # 修复：Grounding DINO需要为每张图像提供独立的文本输入
            text_prompts = [text_prompt] * len(pil_images)
            
            inputs = self.processor(
                images=pil_images, 
                text=text_prompts,  # 使用文本列表而不是单个文本
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)
            
            # 后处理：返回每张图像的结果
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes
            )
            
            # 转换格式
            for result in results:
                if len(result["boxes"]) == 0:
                    all_results.append({
                        "boxes": np.array([]),
                        "scores": np.array([]),
                        "labels": []
                    })
                else:
                    all_results.append({
                        "boxes": result["boxes"].cpu().numpy(),
                        "scores": result["scores"].cpu().numpy(),
                        "labels": result["labels"]
                    })
        
        return all_results
    
    def detect_initial_frames(
        self, 
        frames: List[Dict], 
        text_prompt: str
    ) -> List[Dict]:
        """
        在多帧中检测物体，返回每帧的检测结果
        
        Args:
            frames: 帧列表，每个元素为 {"image": ndarray, "timestamp_sec": float, "frame_index": int}
            text_prompt: 物体描述文本
            
        Returns:
            detections: 检测结果列表，每个元素为
            {
                "frame_idx": int,
                "timestamp_sec": float,
                "boxes": ndarray (N, 4),
                "scores": ndarray (N,),
                "labels": list[str]
            }
        """
        detections = []
        
        for frame_data in frames:
            detection = self.detect_objects_in_frame(
                frame_data["image"], 
                text_prompt
            )
            
            detections.append({
                "frame_idx": frame_data["frame_index"],
                "timestamp_sec": frame_data["timestamp_sec"],
                "boxes": detection["boxes"],
                "scores": detection["scores"],
                "labels": detection["labels"]
            })
        
        return detections
    
    def track_object_in_video(
        self,
        video_dir: str,
        frame_names: List[str],
        init_frame_idx: int,
        init_box: np.ndarray,
        obj_id: int = 1
    ) -> Dict[int, Dict]:
        """
        使用box prompt追踪视频中的物体
        
        Args:
            video_dir: 视频帧所在目录
            frame_names: 帧文件名列表（已排序）
            init_frame_idx: 初始帧索引
            init_box: 初始bbox [x1, y1, x2, y2]
            obj_id: 物体ID（默认1）
            
        Returns:
            tracking_results: {
                frame_idx: {
                    "box": ndarray (4,),  # [x1, y1, x2, y2]
                    "mask": ndarray (H, W),  # bool mask
                    "score": float
                },
                ...
            }
        """
        # 初始化video predictor
        inference_state = self.video_predictor.init_state(video_path=video_dir)
        
        # 在初始帧添加box prompt
        _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=init_frame_idx,
            obj_id=obj_id,
            box=init_box,
        )
        
        # 在整个视频中传播
        tracking_results = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
            # 找到对应的object
            obj_idx = list(out_obj_ids).index(obj_id) if obj_id in out_obj_ids else 0
            
            # 获取mask
            mask = (out_mask_logits[obj_idx] > 0.0).cpu().numpy().squeeze()
            
            # 从mask计算bbox
            if mask.any():
                ys, xs = np.where(mask)
                x1, y1 = xs.min(), ys.min()
                x2, y2 = xs.max(), ys.max()
                box = np.array([x1, y1, x2, y2], dtype=np.float32)
                
                # 计算score（基于mask的置信度）
                score = float(out_mask_logits[obj_idx].sigmoid().mean().cpu().numpy())
            else:
                box = np.array([0, 0, 0, 0], dtype=np.float32)
                score = 0.0
            
            tracking_results[out_frame_idx] = {
                "box": box,
                "mask": mask,
                "score": score
            }
        
        return tracking_results

