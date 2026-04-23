"""物体追踪：统一提取视频帧，对每个物体进行检测和追踪"""

import os
import sys
import json
import argparse
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Optional

import cv2
import numpy as np

src_dir = os.path.dirname(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from entity.grounded_sam2_wrapper import GroundedSAM2Wrapper
from utils.media_utils import FrameExtractor
from utils.bbox_visualizer import BBoxVisualizer
from utils.path_utils import resolve_path_template, build_crop_dir, ensure_dir, get_day_prefix
from utils.bbox_utils import xyxy_to_xywh, xywh_to_xyxy
# 不再需要save_json_compact，直接使用标准json


class ObjectTracker:
    """物体追踪器"""
    
    def __init__(self, config):
        self.config = config
        self.tracking_config = config.entity_tracking
        
        # 初始化Grounded-SAM-2和帧提取器
        self.sam2_wrapper = GroundedSAM2Wrapper({
            "grounding_dino": self.tracking_config["grounding_dino"],
            "sam2": self.tracking_config["sam2"]
        })
        
        video_config = self.tracking_config["video_processing"]
        self.frame_extractor = FrameExtractor(
            sample_fps=video_config["sample_fps"],
            resize_max_size=video_config["resize_max_size"]
        )
        
        # 创建输出目录
        self.entities_dir = self._resolve_path(self.tracking_config["output"]["entities_dir"])
        self.crops_dir = self._resolve_path(self.tracking_config["output"]["crops_dir"])
        self.vis_dir = self._resolve_path(self.tracking_config["output"]["vis_dir"])
        ensure_dir(self.entities_dir)
        ensure_dir(self.crops_dir)
        ensure_dir(self.vis_dir)
    
    def _resolve_path(self, path_template: str) -> str:
        return resolve_path_template(path_template, self.config)
    
    def process_event(self, event_json_path: str, save_entity: bool = True,
                     update_event: bool = False, visualize: bool = False) -> List[str]:
        """处理event：统一提取视频帧，对每个物体进行追踪"""
        with open(event_json_path, 'r', encoding='utf-8') as f:
            event_data = json.load(f)
        
        event_id = event_data["event_id"]
        video_path = event_data["video_path"]
        interaction_objects = event_data["attributes"]["interaction_object"]  # 直接访问
        
        if not interaction_objects:
            return []
        
        print(f"[Tracking] {event_id}: {len(interaction_objects)}个物体")
        
        # 获取视频信息并统一提取所有帧
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        shared_temp_dir = tempfile.mkdtemp()
        print(f"  提取视频帧到: {shared_temp_dir}")
        
        all_frames = self.frame_extractor.extract_frames(video_path)
        for frame_data in all_frames:
            frame_filename = f"{frame_data['frame_index']:06d}.jpg"
            cv2.imwrite(os.path.join(shared_temp_dir, frame_filename), frame_data["image"])
        
        resize_scale = all_frames[0]["image"].shape[0] / original_height if all_frames else 1.0
        print(f"  总帧数: {len(all_frames)}, 缩放比例: {resize_scale:.3f}")
        
        # 初始化共享的SAM2 inference state
        shared_inference_state = None
        if all_frames:
            shared_inference_state = self.sam2_wrapper.video_predictor.init_state(video_path=shared_temp_dir)
        
        # 处理每个物体
        entity_paths, updated_objects, all_objects_data = [], [], []
        
        for obj_info in interaction_objects:
            object_data = self._process_object(
                video_path, event_id, obj_info, all_frames,
                shared_inference_state, resize_scale, video_fps, visualize
            )
            
            if object_data is None:
                print(f"  [Track失败] {obj_info['name']}")
                updated_objects.append(obj_info)
                continue
            
            print(f"  [Track成功] {obj_info['name']}: {len(object_data['timestep'])}帧")
            all_objects_data.append(object_data)
            
            if visualize:
                self._visualize_tracking(event_id, obj_info["name"], video_path, object_data)
            
            updated_obj = obj_info.copy()
            updated_obj.update({
                "bbox": object_data["bbox"],
                "timestep": object_data["timestep"],
                "score": object_data["score"]
            })
            updated_objects.append(updated_obj)
        
        # 保存并清理
        if save_entity and all_objects_data:
            entity_paths.append(self._save_event_entities(event_id, video_path, all_objects_data))
        
        shutil.rmtree(shared_temp_dir)
        
        if update_event:
            event_data["attributes"]["interaction_object"] = updated_objects
            # 使用标准json.dump保存event数据
            with open(event_json_path, 'w', encoding='utf-8') as f:
                json.dump(event_data, f, indent=2, ensure_ascii=False)
        
        return entity_paths
    
    def _process_object(self, video_path: str, event_id: str, obj_info: Dict,
                       shared_frames: List[Dict], shared_inference_state,
                       shared_resize_scale: float, video_fps: float, visualize: bool = False) -> Optional[Dict]:
        """处理单个物体：检测初始帧并追踪所有segment"""
        obj_name = obj_info["name"]
        description = obj_info["description"]
        segments = obj_info["interaction_segments"]  # 直接访问
        
        if not segments:
            return None
        
        all_initial_detections, all_tracking_results, all_frames_data = [], [], {}
        
        # 处理每个segment：检测初始帧 -> 追踪整个segment
        for seg_idx, segment in enumerate(segments):
            start_time, end_time = segment["start_time"], segment["end_time"]
            print(f"    时段{seg_idx+1}: {start_time:.1f}s-{end_time:.1f}s")
            
            # 筛选segment的帧
            segment_frames = [f for f in shared_frames if start_time <= f["timestamp_sec"] <= end_time]
            if not segment_frames:
                print(f"      [警告] 没有提取到帧")
                continue
            print(f"      提取帧数: {len(segment_frames)}")
            
            # 批量检测初始n帧
            n_frames = min(self.tracking_config["initial_detection"]["n_frames"], len(segment_frames))
            initial_frames = segment_frames[:n_frames]
            print(f"      初始检测: {n_frames}帧")
            
            initial_images = [f["image"] for f in initial_frames]
            batch_detections = self.sam2_wrapper.detect_objects_in_frames_batch(
                images=initial_images, text_prompt=description, batch_size=8
            )
            
            # 转换为统一格式
            detections = [
                {"frame_idx": initial_frames[i]["frame_index"],
                 "timestamp_sec": initial_frames[i]["timestamp_sec"],
                 "boxes": det["boxes"], "scores": det["scores"], "labels": det["labels"]}
                for i, det in enumerate(batch_detections)
            ]
            
            # 选择最佳初始帧
            best_detection = self._select_best_initial_frame(detections)
            if best_detection is None:
                print(f"      [警告] 未检测到物体")
                if seg_idx < len(segments) - 1:
                    self.sam2_wrapper.video_predictor.reset_state(shared_inference_state)
                continue
            print(f"      最佳帧: idx={best_detection['frame_idx']}, score={best_detection['best_score']:.3f}")
            
            # 保存初始检测结果
            for i, d in enumerate(detections):
                if len(d["boxes"]) > 0:
                    bbox_xywh = xyxy_to_xywh(d["boxes"][0].tolist())
                    all_initial_detections.append({
                        "frame_idx": d["frame_idx"], "timestamp_sec": d["timestamp_sec"],
                        "bbox": bbox_xywh, "score": float(d["scores"][0])
                    })
                    all_frames_data[d["timestamp_sec"]] = {
                        "image": initial_frames[i]["image"], "bbox": bbox_xywh
                    }
            
            # 找到初始帧在segment中的相对位置
            global_init_frame_idx = best_detection["frame_idx"]
            relative_init_frame_idx = next(
                (i for i, f in enumerate(segment_frames) if f["frame_index"] == global_init_frame_idx), None
            )
            if relative_init_frame_idx is None:
                print(f"      [错误] 初始帧{global_init_frame_idx}不在segment中")
                if seg_idx < len(segments) - 1:
                    self.sam2_wrapper.video_predictor.reset_state(shared_inference_state)
                continue
            
            # SAM2追踪
            print(f"      追踪...（全局idx={global_init_frame_idx}）")
            tracking_results = self._track_segment(
                shared_inference_state, segment_frames, relative_init_frame_idx,
                best_detection["best_box"], seg_idx + 1
            )
            print(f"      追踪帧数: {len(tracking_results)}")
            
            # 保存追踪结果（缩放回原始尺寸）
            for relative_idx, result in tracking_results.items():
                timestamp = segment_frames[relative_idx]["timestamp_sec"]
                bbox_scaled = [x / shared_resize_scale for x in result["box"]]
                all_tracking_results.append({
                    "frame_idx": segment_frames[relative_idx]["frame_index"],
                    "timestamp_sec": timestamp, "bbox": bbox_scaled, "score": result["score"]
                })
            
            # 重置state供下一个segment使用
            if seg_idx < len(segments) - 1:
                self.sam2_wrapper.video_predictor.reset_state(shared_inference_state)
        
        # 处理完当前object后reset
        self.sam2_wrapper.video_predictor.reset_state(shared_inference_state)
        
        if not all_tracking_results:
            return None
        
        # 按timestamp排序，构建bbox/timestep/score列表
        all_tracking_results.sort(key=lambda x: x["timestamp_sec"])
        bbox_list, timestep_list, score_list = [], [], []
        for result in all_tracking_results:
            x1, y1, x2, y2 = result["bbox"]
            bbox_list.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            timestep_list.append(float(result["timestamp_sec"]))
            score_list.append(float(result["score"]))
        
        # 选择top-k crops保存
        max_crops = self.tracking_config["initial_detection"]["max_crop_images"]
        sorted_detections = sorted(all_initial_detections, key=lambda x: x["score"], reverse=True)
        
        # 不足则从tracking补充
        if len(sorted_detections) < max_crops:
            initial_ts = {d["timestamp_sec"] for d in sorted_detections}
            additional = sorted([r for r in all_tracking_results if r["timestamp_sec"] not in initial_ts],
                              key=lambda x: x["score"], reverse=True)[:max_crops - len(sorted_detections)]
            sorted_detections.extend([{**f, "from_tracking": True} for f in additional])
        
        sorted_detections = sorted_detections[:max_crops]
        image_paths = self._save_crops(video_path, event_id, obj_name, sorted_detections, 
                                      all_frames_data, video_fps)
        print(f"    保存了 {len(image_paths)} 张crops")
        
        return {
            "object_info": obj_info, "bbox": bbox_list, "timestep": timestep_list,
            "score": score_list, "initial_detections": all_initial_detections,
            "image_paths": image_paths
        }
    
    def _save_crops(self, video_path: str, event_id: str, obj_name: str,
                   detections: List[Dict], frames_data: Dict, video_fps: float) -> List[str]:
        """保存crop图片"""
        image_paths = []
        cap = cv2.VideoCapture(video_path)
        
        for det in detections:
            timestamp = det["timestamp_sec"]
            x1, y1, x2, y2 = [int(v) for v in xywh_to_xyxy(det["bbox"])]
            
            # 获取帧图像：优先使用缓存
            if not det.get("from_tracking", False) and timestamp in frames_data:
                frame_img = frames_data[timestamp]["image"]
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * video_fps))
                ret, frame_img = cap.read()
                if not ret:
                    continue
            
            # 裁剪并保存
            # 检查bbox坐标是否有效
            if x1 >= x2 or y1 >= y2:
                continue  # 跳过无效的bbox

            # 裁剪图像
            crop = frame_img[y1:y2, x1:x2]

            # 检查crop是否为空（避免cv2.imwrite报错）
            if crop.size == 0:
                continue  # 跳过空图像

            crop_dir = build_crop_dir(self.crops_dir, event_id, obj_name)
            ensure_dir(crop_dir)
            crop_path = os.path.join(crop_dir, f"frame{det['frame_idx']}.jpg")
            cv2.imwrite(crop_path, crop)
            image_paths.append(crop_path)
        
        cap.release()
        return image_paths
    
    def _track_segment(self, inference_state, segment_frames: List[Dict],
                      init_frame_idx: int, init_box: np.ndarray, obj_id: int) -> Dict[int, Dict]:
        """使用SAM2追踪segment"""
        global_frame_idx = segment_frames[init_frame_idx]["frame_index"]
        
        # 添加初始box prompt
        self.sam2_wrapper.video_predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=global_frame_idx,
            obj_id=obj_id, box=init_box
        )
        
        # 传播追踪
        all_results = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_wrapper.video_predictor.propagate_in_video(inference_state):
            if obj_id not in out_obj_ids:
                continue
            
            obj_idx = list(out_obj_ids).index(obj_id)
            mask = (out_mask_logits[obj_idx] > 0.0).cpu().numpy().squeeze()
            
            # 从mask计算bbox和score
            if mask.any():
                ys, xs = np.where(mask)
                box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
                score = float(out_mask_logits[obj_idx].sigmoid().mean().cpu().numpy())
            else:
                box = np.array([0, 0, 0, 0], dtype=np.float32)
                score = 0.0
            
            all_results[out_frame_idx] = {"box": box, "mask": mask, "score": score}
        
        # 映射回segment的相对索引
        segment_indices = {f["frame_index"]: idx for idx, f in enumerate(segment_frames)}
        return {segment_indices[gidx]: result for gidx, result in all_results.items() 
                if gidx in segment_indices}
    
    def _select_best_initial_frame(self, detections: List[Dict]) -> Optional[Dict]:
        """从检测结果中选择得分最高的帧"""
        best_frame, best_score = None, 0.0
        
        for det in detections:
            if len(det["boxes"]) == 0:
                continue
            max_idx = det["scores"].argmax()
            score = float(det["scores"][max_idx])
            if score > best_score:
                best_score = score
                best_frame = {
                    "frame_idx": det["frame_idx"],
                    "timestamp_sec": det["timestamp_sec"],
                    "best_box": det["boxes"][max_idx],
                    "best_score": score
                }
        return best_frame
    
    def _save_event_entities(self, event_id: str, video_path: str, all_objects_data: List[Dict]) -> str:
        """保存事件级entity数据"""
        objects = [{
            "object_name": obj_data["object_info"]["name"],
            "object_info": obj_data["object_info"],
            "bbox": obj_data["bbox"],
            "timestep": obj_data["timestep"],
            "score": obj_data["score"],
            "initial_detections": obj_data["initial_detections"],
            "image_paths": obj_data["image_paths"],
            "metadata": {"num_frames": len(obj_data["bbox"])}
        } for obj_data in all_objects_data]
        
        event_data = {
            "event_id": event_id, "video_path": video_path, "objects": objects,
            "metadata": {"created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "num_objects": len(objects)}
        }
        
        filepath = os.path.join(self.entities_dir, get_day_prefix(event_id), f"{event_id}.json")
        ensure_dir(os.path.dirname(filepath))
        # 使用标准json.dump保存entity数据
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(event_data, f, indent=2, ensure_ascii=False)
        return filepath
    
    def _visualize_tracking(self, event_id: str, obj_name: str,
                           video_path: str, object_data: Dict) -> None:
        """可视化追踪结果"""
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # 构建可视化字典：timestamp -> frame_idx, bbox [x,y,w,h] -> [x1,y1,x2,y2]
        tracking_dict = {
            int(ts * video_fps): {
                "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                "score": score, "timestamp": ts
            }
            for bbox, ts, score in zip(object_data["bbox"], 
                                       object_data["timestep"], 
                                       object_data["score"])
        }
        
        output_path = os.path.join(self.vis_dir, f"{event_id}_{obj_name}_tracking.mp4")
        BBoxVisualizer.visualize_tracking_results(
            video_path=video_path, tracking_results=tracking_dict,
            output_path=output_path, label=obj_name, fps=30.0, show_score=True
        )


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="物体追踪")
    parser.add_argument("event_json", help="Event JSON文件路径")
    parser.add_argument("--update_event", action="store_true", help="更新event JSON")
    parser.add_argument("--visualize", action="store_true", help="生成可视化")
    parser.add_argument("--config", 
                       default="/data/xuyuan/Egolife_env/ego-things/personalization/configs/config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    args.update_event = True
    
    from config import Config
    config = Config.from_yaml(args.config)
    tracker = ObjectTracker(config)
    
    entity_paths = tracker.process_event(
        event_json_path=args.event_json, save_entity=True,
        update_event=args.update_event, visualize=args.visualize
    )
    
    if entity_paths:
        print("生成的entity文件:")
        for path in entity_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()

