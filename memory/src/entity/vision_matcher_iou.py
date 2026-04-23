"""
视觉匹配器 - IoU方法

使用entity的timestep，均匀采样N帧，用entity描述检测bbox，计算与new_object的IoU作为相似度
"""

import os
import sys
import cv2
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import tempfile
import shutil

# 添加src目录到路径
src_dir = os.path.dirname(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from entity.grounded_sam2_wrapper import GroundedSAM2Wrapper
from utils.path_utils import resolve_path_template, build_event_entity_path
from utils.bbox_utils import calculate_iou, xyxy_to_xywh


# IoU计算函数已移至utils.bbox_utils模块


class VisionMatcherIoU:
    """视觉匹配器 - IoU方法"""
    
    def __init__(self, config):
        """
        初始化IoU-based视觉匹配器
        
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 初始化Grounding DINO（传入entity_tracking配置）
        self.grounded_sam2 = GroundedSAM2Wrapper(config._config["entity_tracking"])
        
        # 从原始配置字典获取vision2vision配置参数
        matching_config = config._config["entity"]["global_matching"]["vision2vision"]
        self.num_sample_frames = matching_config.get("num_sample_frames", 3)
        
        # 获取entity tracking结果目录
        self.entities_tracking_dir = self._resolve_path(
            config.entity_tracking["output"]["entities_dir"]
        )
        
        print(f"[VisionMatcherIoU] 初始化完成")
    
    def _resolve_path(self, path_template: str) -> str:
        """解析路径模板中的变量"""
        return resolve_path_template(path_template, self.config)
    
    def _load_entity_tracking(self, event_id: str, object_name: str) -> Dict:
        """
        加载entity tracking结果（事件级JSON格式）
        
        Args:
            event_id: event ID
            object_name: object名称
        
        Returns:
            tracking数据字典，包含timestep、bbox、video_path等
        """
        # entities_tracking_dir 已经指向 .../entities/event，不需要再加 "event"
        base_dir = self.entities_tracking_dir
        
        # 加载事件级JSON
        event_path = build_event_entity_path(base_dir, event_id)
        if not os.path.exists(event_path):
            return None
        
        with open(event_path, 'r', encoding='utf-8') as f:
            event_data = json.load(f)
        
        # 从objects列表中找到匹配的object
        for obj in event_data.get("objects", []):
            if obj["object_name"] == object_name:
                return {
                    "event_id": event_data["event_id"],
                    "object_name": obj["object_name"],
                    "object_info": obj["object_info"],
                    "video_path": event_data["video_path"],
                    "bbox": obj["bbox"],
                    "timestep": obj["timestep"],
                    "score": obj["score"],
                    "initial_detections": obj.get("initial_detections", []),
                    "image_paths": obj.get("image_paths", []),
                    "metadata": obj.get("metadata", {})
                }
        
        return None  # 未找到匹配的object
    
    def _extract_frame_at_timestamp(self, video_path: str, timestamp_sec: float) -> np.ndarray:
        """
        从视频中提取指定时间戳的帧
        
        Args:
            video_path: 视频路径
            timestamp_sec: 时间戳（秒）
        
        Returns:
            frame图像 (BGR格式)
        """
        cap = cv2.VideoCapture(video_path)
        
        # 设置到指定时间戳
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None
    
    def _extract_frames_batch(
        self, 
        video_timestamps: Dict[str, List[float]]
    ) -> Dict[Tuple[str, float], np.ndarray]:
        """
        批量从多个视频中提取帧，减少IO次数
        
        Args:
            video_timestamps: {video_path: [timestamp1, timestamp2, ...]}
        
        Returns:
            {(video_path, timestamp): frame} 帧缓存字典
        """
        frame_cache = {}
        
        # 按video_path分组处理
        for video_path, timestamps in video_timestamps.items():
            if not timestamps:
                continue
            
            cap = cv2.VideoCapture(video_path)
            
            # 按时间戳排序，提高提取效率
            sorted_timestamps = sorted(set(timestamps))
            
            for ts in sorted_timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                ret, frame = cap.read()
                if ret:
                    frame_cache[(video_path, ts)] = frame
            
            cap.release()
        
        return frame_cache
    
    def match(
        self,
        new_object_data: Dict,
        candidate_entities: List[str],
        global_entities: Dict,
        return_vis_data: bool = False
    ) -> Tuple[Dict[str, float], Optional[Dict]]:
        """
        使用IoU方法计算视觉相似度（时序对齐版本）
        
        关键改进：使用new_object的timestep进行采样，确保时序对齐
        - 从new_object的video和timestep中采样N帧
        - 在这些帧上用各个候选entity的描述进行检测
        - 计算检测bbox与new_object bbox的IoU（时序对齐）
        
        Args:
            new_object_data: 新object的数据，包含event_id、name等
            candidate_entities: 候选实体名称列表
            global_entities: 全局实体字典
            return_vis_data: 是否返回可视化数据
        
        Returns:
            (similarities, vis_data):
                - similarities: {global_name: iou_score}
                - vis_data: 可视化数据（如果return_vis_data=True）
        """
        if not candidate_entities:
            return ({}, None) if return_vis_data else {}
        
        # ========== 阶段0：加载new_object的tracking数据（时序对齐的关键）==========
        new_event_id = new_object_data.get("event_id")
        new_object_name = new_object_data.get("name")
        
        new_object_tracking = self._load_entity_tracking(new_event_id, new_object_name)
        if not new_object_tracking:
            result = {name: 0.0 for name in candidate_entities}
            return (result, None) if return_vis_data else result
        
        new_timesteps = new_object_tracking.get("timestep", [])
        new_bboxes = new_object_tracking.get("bbox", [])
        new_video_path = new_object_tracking.get("video_path")
        
        if not new_timesteps or not new_bboxes or not new_video_path:
            result = {name: 0.0 for name in candidate_entities}
            return (result, None) if return_vis_data else result
        
        # 从new_object的timestep中均匀采样N帧（时序对齐的基准）
        new_step = max(1, len(new_timesteps) // self.num_sample_frames)
        new_sampled_indices = list(range(0, len(new_timesteps), new_step))[:self.num_sample_frames]
        
        # ========== 阶段1：为每个候选entity准备检测任务 ==========
        # 所有任务都使用new_object的video和timestep
        entity_tasks = {}
        
        for global_name in candidate_entities:
            entity = global_entities[global_name]
            
            # 获取实体的描述
            descriptions = entity.get("descriptions", [])
            if not descriptions:
                entity_tasks[global_name] = []
                continue
            
            desc_obj = json.loads(descriptions[-1]) if isinstance(descriptions[-1], str) else descriptions[-1]
            desc_text = desc_obj["text"]
            
            # 创建检测任务列表（使用new_object的timestep）
            tasks = []
            for idx in new_sampled_indices:
                tasks.append({
                    "video_path": new_video_path,  # 使用new_object的video
                    "timestamp": new_timesteps[idx],  # 使用new_object的timestep
                    "desc_text": desc_text,  # 使用全局entity的描述进行检测
                    "new_object_bbox": new_bboxes[idx]  # new_object在该帧的bbox（用于计算IoU）
                })
            
            entity_tasks[global_name] = tasks
        
        # 如果没有任何任务，直接返回
        if not any(entity_tasks.values()):
            result = {name: 0.0 for name in candidate_entities}
            return (result, None) if return_vis_data else result
        
        # 统计任务数量
        total_tasks = sum(len(tasks) for tasks in entity_tasks.values())
        print(f"    采样帧数: {total_tasks} 帧（来自 {len(candidate_entities)} 个候选实体）")
        
        # ========== 阶段2：批量提取视频帧 ==========
        # 收集所有需要的（video_path, timestamp）
        video_timestamps = defaultdict(list)
        for tasks in entity_tasks.values():
            for task in tasks:
                video_timestamps[task["video_path"]].append(task["timestamp"])
        
        # 批量提取帧
        frame_cache = self._extract_frames_batch(video_timestamps)
        
        # ========== 阶段3：批量检测 ==========
        # 按描述文本分组：{desc_text: [(frame, video_path, timestamp), ...]}
        desc_groups = defaultdict(list)
        missing_frames = 0
        for tasks in entity_tasks.values():
            for task in tasks:
                frame_key = (task["video_path"], task["timestamp"])
                if frame_key in frame_cache:
                    desc_groups[task["desc_text"]].append({
                        "frame": frame_cache[frame_key],
                        "video_path": task["video_path"],
                        "timestamp": task["timestamp"]
                    })
                else:
                    missing_frames += 1
        
        # 对每个描述文本，批量检测其所有帧
        detection_cache = {}  # {(desc_text, video_path, timestamp): detection_result}
        
        for desc_text, frame_list in desc_groups.items():
            frames = [item["frame"] for item in frame_list]
            
            # 批量检测
            detection_results = self.grounded_sam2.detect_objects_in_frames_batch(
                images=frames,
                text_prompt=desc_text,
                batch_size=8
            )
            
            # 存入缓存
            for item, result in zip(frame_list, detection_results):
                key = (desc_text, item["video_path"], item["timestamp"])
                detection_cache[key] = result
        
        # ========== 阶段4：计算IoU相似度 ==========
        similarities = {}
        vis_data = {} if return_vis_data else None
        
        for global_name in candidate_entities:
            tasks = entity_tasks.get(global_name, [])
            if not tasks:
                similarities[global_name] = 0.0
                continue
            
            all_ious = []
            
            # 可视化数据收集
            if return_vis_data:
                vis_entity_data = {
                    "frames": [],
                    "entity_bboxes": [],
                    "detected_bboxes": [],
                    "ious": [],
                    "timestamps": [],
                    "desc_text": tasks[0]["desc_text"] if tasks else ""
                }
            
            for task in tasks:
                # 从缓存中获取检测结果
                cache_key = (task["desc_text"], task["video_path"], task["timestamp"])
                detection_result = detection_cache.get(cache_key)
                
                if not detection_result:
                    continue
                
                detected_boxes = detection_result.get("boxes", [])
                new_object_bbox = task["new_object_bbox"]  # new_object在该帧的bbox
                
                if len(detected_boxes) == 0:
                    continue
                
                # 计算IoU（取最大值）
                # 注意：new_object_bbox是xywh格式，detected_box是xyxy格式，需要转换
                max_iou = 0.0
                best_detected_box = None
                for detected_box in detected_boxes:
                    detected_box_xywh = xyxy_to_xywh(detected_box)
                    iou = calculate_iou(new_object_bbox, detected_box_xywh, format="xywh")
                    if iou > max_iou:
                        max_iou = iou
                        best_detected_box = detected_box  # 记录最佳匹配的检测框
                
                if max_iou > 0:
                    all_ious.append(max_iou)
                
                # 收集可视化数据
                if return_vis_data:
                    frame_key = (task["video_path"], task["timestamp"])
                    if frame_key in frame_cache:
                        vis_entity_data["frames"].append(frame_cache[frame_key])
                        vis_entity_data["entity_bboxes"].append(new_object_bbox)  # 存储new_object的bbox
                        # 只存储最佳匹配的检测框（如果有的话）
                        vis_entity_data["detected_bboxes"].append([best_detected_box] if best_detected_box is not None else [])
                        vis_entity_data["ious"].append(max_iou)
                        vis_entity_data["timestamps"].append(task["timestamp"])
            
            # 计算平均IoU
            if all_ious:
                avg_iou = float(np.mean(all_ious))
                similarities[global_name] = avg_iou
            else:
                similarities[global_name] = 0.0
            
            # 保存可视化数据
            if return_vis_data and vis_entity_data["frames"]:
                vis_data[global_name] = vis_entity_data
        
        if return_vis_data:
            return similarities, vis_data
        else:
            return similarities

