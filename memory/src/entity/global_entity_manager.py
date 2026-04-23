"""
全局实体管理器：维护全局实体库，实现跨event的实体匹配

功能：
1. 文本匹配：使用Sentence-BERT计算描述相似度
2. 视觉匹配：使用CLIP计算crop图像相似度
3. 综合决策：加权融合文本和视觉相似度
4. 实体管理：创建、更新、保存全局实体
"""

import os
import sys
import json
import argparse
import cv2
from typing import Dict, List, Optional, Tuple
from glob import glob

# 添加src目录到路径
src_dir = os.path.dirname(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from entity.entity_storage import EntityStorage
from entity.text_matcher import TextMatcher
from entity.vision_matcher_iou import VisionMatcherIoU
#from entity.vision_matcher_vl import VisionMatcherVL
from utils.path_utils import resolve_path_template, build_event_entity_path
from utils.match_visualizer import MatchVisualizer


class GlobalEntityManager:
    """全局实体管理器：协调文本匹配、视觉匹配、实体更新"""
    
    def __init__(self, config, visualize: bool = False):
        """
        初始化全局实体管理器
        
        Args:
            config: 配置对象
            visualize: 是否开启可视化
        """
        self.config = config
        self.visualize = visualize
        
        # 解析路径配置（使用Config类展开后的属性）
        global_entities_path = self._resolve_path(
            config.global_entities_path
        )
        self.crops_base_dir = self._resolve_path(
            config.entity_tracking["output"]["crops_dir"]
        )
        
        # 初始化实体存储
        self.entity_storage = EntityStorage(
            entities_path=global_entities_path,
            max_descriptions=config.entity_global_matching_max_history_events,
            max_images=10
        )
        
        # 初始化匹配器
        self.text_matcher = TextMatcher(config)
        
        # 从原始配置字典读取global_matching配置（因为Config类没有完全展开嵌套配置）
        matching_config = config._config["entity"]["global_matching"]
        
        # 根据配置选择vision matcher
        vision_method = matching_config["vision2vision"].get("method", "vl_model")
        
        if vision_method == "iou":
            self.vision_matcher = VisionMatcherIoU(config)
        elif vision_method == "vl_model":
            self.vision_matcher = VisionMatcherVL(config)
        else:
            raise ValueError(f"未知的vision匹配方法: {vision_method}")
        
        self.vision_method = vision_method
        
        # 获取匹配配置
        self.match_threshold = matching_config["match_threshold"]
        
        # 初始化可视化器
        if self.visualize:
            vis_dir = "/data/xuyuan/Egolife_env/ego-things/personalization/data/vis/match"
            self.match_visualizer = MatchVisualizer(vis_dir)
            print(f"[GlobalEntityManager] 可视化已开启，输出目录: {vis_dir}")
        
        # 获取entity tracking目录（用于加载new object的tracking数据）
        self.entities_tracking_dir = self._resolve_path(
            config.entity_tracking["output"]["entities_dir"]
        )
        
        print(f"[GlobalEntityManager] 初始化: 实体数={len(self.entity_storage.entities)}, "
              f"方法={vision_method}, 阈值={self.match_threshold}")
    
    def _resolve_path(self, path_template: str) -> str:
        """解析路径模板中的变量"""
        return resolve_path_template(path_template, self.config)
    
    def _load_object_tracking(self, event_id: str, object_name: str) -> Optional[Dict]:
        """
        加载object的tracking数据（事件级JSON格式）
        
        Args:
            event_id: event ID
            object_name: object名称
        
        Returns:
            tracking数据字典
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
    
    def _visualize_entity_match_new(
        self,
        obj_info: Dict,
        obj_crop_paths: List[str],
        global_name: Optional[str],
        is_match: bool,
        similarity_scores: Dict[str, float]
    ):
        """
        新的实体匹配可视化方法（为多张crop图像生成可视化）
        
        Args:
            obj_info: event object信息（包含name, description, action, event_id等）
            obj_crop_paths: event object的crop图像路径列表
            global_name: 匹配到的全局实体名称（None表示新实体）
            is_match: 是否匹配成功
            similarity_scores: 相似度分数字典
        """
        event_id = obj_info["event_id"]
        event_entity_name = obj_info["name"]
        event_entity_desc = obj_info["description"]
        
        if not obj_crop_paths:
            return
        
        # 准备全局实体信息（如果有匹配）
        global_entity_desc = None
        global_entity_frames = []  # 存储多张全局实体图像
        global_entity_bboxes = []
        
        if global_name and global_name in self.entity_storage.entities:
            global_entity = self.entity_storage.entities[global_name]
            
            # 获取全局实体的描述（最新的一条）
            descriptions = global_entity.get("descriptions", [])
            if descriptions:
                last_desc = descriptions[-1]
                if isinstance(last_desc, str):
                    last_desc = json.loads(last_desc)
                global_entity_desc = last_desc.get("text", "")
            
            # 获取全局实体的多张代表图像（最多5张）
            representative_images = global_entity.get("representative_images", [])
            for global_img_path in representative_images[:5]:
                if os.path.exists(global_img_path):
                    global_frame = cv2.imread(global_img_path)
                    if global_frame is not None:
                        global_entity_frames.append(global_frame)
                        gh, gw = global_frame.shape[:2]
                        global_entity_bboxes.append([0, 0, gw, gh])  # xywh格式
        
        # 为每张event entity的crop图像生成可视化（最多5张）
        num_frames_to_vis = min(5, len(obj_crop_paths))
        
        for frame_idx in range(num_frames_to_vis):
            # 读取event entity的crop图像
            event_entity_frame = cv2.imread(obj_crop_paths[frame_idx])
            if event_entity_frame is None:
                continue
            
            # event entity的bbox（crop图像就是bbox区域）
            h, w = event_entity_frame.shape[:2]
            event_entity_bbox = [0, 0, w, h]  # xywh格式
            
            # 选择对应的全局实体图像（循环使用）
            global_entity_frame = None
            global_entity_bbox = None
            if global_entity_frames:
                global_idx = frame_idx % len(global_entity_frames)
                global_entity_frame = global_entity_frames[global_idx]
                global_entity_bbox = global_entity_bboxes[global_idx]
            
            # 调用可视化器
            self.match_visualizer.visualize_entity_match(
                event_id=event_id,
                event_entity_name=event_entity_name,
                event_entity_desc=event_entity_desc,
                event_entity_frame=event_entity_frame,
                event_entity_bbox=event_entity_bbox,
                global_entity_name=global_name,
                global_entity_desc=global_entity_desc,
                global_entity_frame=global_entity_frame,
                global_entity_bbox=global_entity_bbox,
                is_match=is_match,
                similarity_scores=similarity_scores,
                frame_idx=frame_idx
            )
    
    def match_object(
        self,
        obj_info: Dict,
        obj_crop_paths: List[str]
    ) -> Tuple[Optional[str], Dict]:
        """
        匹配单个object到全局实体
        
        Args:
            obj_info: object信息，包含name, description等
            obj_crop_paths: crop图像路径列表
        
        Returns:
            (global_name, match_info):
                - global_name: 匹配到的全局实体名（None表示未匹配）
                - match_info: 匹配信息字典
        """
        description = obj_info["description"]
        global_entities = self.entity_storage.entities
        
        print(f"\n[匹配开始] Object: {obj_info['name']}")
        print(f"  描述: {description}")
        print(f"  全局实体数量: {len(global_entities)}")
        
        # 如果全局库为空，直接返回未匹配
        if not global_entities:
            print(f"  结果: 全局库为空，创建新实体")
            return None, {
                "is_new": True,
                "vision_similarity": 0.0,
                "final_score": 0.0,
                "unmatched_candidates": {}
            }
        
        # 阶段1：文本匹配，获取Top-K候选
        print(f"\n  [阶段1] 文本匹配（阈值: {self.text_matcher.threshold}）")
        text_candidates = self.text_matcher.match(description, global_entities)
        print(f"  找到 {len(text_candidates)} 个文本候选")
        
        if not text_candidates:
            print(f"  结果: 无文本候选（所有候选相似度 < {self.text_matcher.threshold}），创建新实体")
            return None, {
                "is_new": True,
                "vision_similarity": 0.0,
                "final_score": 0.0,
                "unmatched_candidates": {}
            }
        
        # 打印文本候选详情
        for name, sim in text_candidates:
            print(f"    - {name}: 文本相似度 = {sim:.3f}")
        
        # 阶段2：视觉匹配候选实体
        print(f"\n  [阶段2] 视觉匹配（IoU方法）")
        candidate_names = [name for name, _ in text_candidates]
        
        if isinstance(self.vision_matcher, VisionMatcherIoU):
            # IoU方法：需要new_object_data（包含event_id、name等）和global_entities
            # 如果开启可视化，获取vis_data
            if self.visualize:
                vision_similarities, vis_data = self.vision_matcher.match(
                    new_object_data=obj_info,
                    candidate_entities=candidate_names,
                    global_entities=global_entities,
                    return_vis_data=True
                )
            else:
                vision_similarities = self.vision_matcher.match(
                    new_object_data=obj_info,
                    candidate_entities=candidate_names,
                    global_entities=global_entities,
                    return_vis_data=False
                )
                vis_data = None
        else:  # VisionMatcherVL
            # VL方法：需要crop图像路径和global_entities
            vision_similarities = self.vision_matcher.match(
                new_crop_paths=obj_crop_paths,
                candidate_entities=candidate_names,
                global_entities=global_entities
            )
            vis_data = None
        
        # 阶段3：直接使用vision相似度进行决策（不做加权融合）
        print(f"\n  [阶段3] 最终决策（匹配阈值: {self.match_threshold}）")
        best_match = None
        best_vision_sim = 0.0
        best_text_sim = 0.0

        # 记录所有候选的分数（text_similarity和vision_similarity）
        all_candidates_scores = {}

        # 将text_candidates转为字典方便查询
        text_sim_dict = {name: sim for name, sim in text_candidates}

        for global_name, text_sim in text_candidates:
            vision_sim = vision_similarities.get(global_name, 0.0)
            all_candidates_scores[global_name] = {
                "text_similarity": float(text_sim),
                "vision_similarity": float(vision_sim)
            }
            print(f"    - {global_name}: 文本={text_sim:.3f}, IoU={vision_sim:.3f}")
            
            # 直接使用vision相似度作为匹配依据
            if vision_sim > best_vision_sim:
                best_vision_sim = vision_sim
                best_text_sim = text_sim
                best_match = global_name

        # 构建unmatched_candidates（未匹配到的所有候选实体及其分数，除了best_match）
        unmatched_candidates = {}
        for global_name, scores in all_candidates_scores.items():
            if global_name != best_match:
                unmatched_candidates[global_name] = {
                    "text_similarity": scores["text_similarity"],
                    "vision_similarity": scores["vision_similarity"]
                }

        # 判断是否超过阈值
        print(f"\n  最佳匹配: {best_match}, 文本={best_text_sim:.3f}, IoU={best_vision_sim:.3f}")
        
        is_match = best_vision_sim >= self.match_threshold
        global_name_result = best_match if is_match else None
        
        # 新的可视化调用（使用时序对齐的vis_data）
        if self.visualize and vis_data and best_match and best_match in vis_data:
            print(f"\n  [可视化] 保存到: {self.match_visualizer.vis_dir}")
            # 获取匹配实体的描述
            global_entity_desc = None
            if global_name_result and global_name_result in global_entities:
                descriptions = global_entities[global_name_result].get("descriptions", [])
                if descriptions:
                    last_desc = descriptions[-1]
                    if isinstance(last_desc, str):
                        last_desc = json.loads(last_desc)
                    global_entity_desc = last_desc.get("text", "")
            
            # 使用新的时序对齐可视化方法
            self.match_visualizer.visualize_aligned_match(
                event_id=obj_info["event_id"],
                event_entity_name=obj_info["name"],
                event_entity_desc=obj_info["description"],
                global_entity_name=global_name_result,
                global_entity_desc=global_entity_desc,
                vis_data=vis_data[best_match],  # 使用best_match的vis_data
                is_match=is_match,
                text_similarity=float(best_text_sim) if best_text_sim > 0 else 0.0
            )
        
        if is_match:
            print(f"  结果: ✓ 匹配成功 -> {best_match}")   
            return best_match, {
                "is_new": False,
                "text_similarity": float(best_text_sim),
                "vision_similarity": float(best_vision_sim),
                "final_score": float(best_vision_sim),
                "unmatched_candidates": unmatched_candidates
            }
        else:
            print(f"  结果: ✗ IoU未达到阈值（{best_vision_sim:.3f} < {self.match_threshold}），创建新实体")
            # 全部候选都未match上（此时best_match为None）
            return None, {
                "is_new": True,
                "text_similarity": float(best_text_sim) if best_text_sim > 0 else 0.0,
                "vision_similarity": float(best_vision_sim),
                "final_score": float(best_vision_sim),
                "unmatched_candidates": all_candidates_scores  # 此时所有候选都为未匹配
            }
    
    def process_event(
        self,
        event_json_path: str,
        output_path: Optional[str] = None,
        crops_base_dir: Optional[str] = None
    ) -> str:
        """
        处理单个event，为所有objects分配global_name
        
        Args:
            event_json_path: event JSON文件路径
            output_path: 输出路径（None表示覆盖原文件）
            crops_base_dir: crops基础目录（None表示使用默认配置）
        
        Returns:
            output_path: 输出文件路径
        """
        # 读取event JSON
        with open(event_json_path, 'r', encoding='utf-8') as f:
            event_data = json.load(f)
        
        event_id = event_data["event_id"]
        video_path = event_data["video_path"]  # 直接访问
        interaction_objects = event_data["attributes"]["interaction_object"]  # 直接访问
        
        if not interaction_objects:
            return event_json_path
        
        print(f"[处理Event] {event_id}: {len(interaction_objects)}个物体")
        
        # 确定crops目录
        if crops_base_dir is None:
            crops_base_dir = self.crops_base_dir
        
        # 处理每个object
        for obj_info in interaction_objects:
            obj_name = obj_info["name"]
            
            # 查找该object的crops
            obj_crops_dir = os.path.join(crops_base_dir, event_id, obj_name)
            
            if not os.path.exists(obj_crops_dir):
                obj_info["global_name"] = None
                obj_info["match_info"] = {"error": "no_crops"}
                continue
            
            # 获取所有crop图像路径
            obj_crop_paths = sorted(glob(os.path.join(obj_crops_dir, "*.jpg")))
            
            if not obj_crop_paths:
                obj_info["global_name"] = None
                obj_info["match_info"] = {"error": "no_crops"}
                continue
            
            # 添加event_id到obj_info（用于VisionMatcherIoU）
            obj_info["event_id"] = event_id
            
            # 匹配object
            global_name, match_info = self.match_object(obj_info, obj_crop_paths)
            
            # 输出匹配结果
            if match_info["is_new"]:
                # 创建新实体
                global_name = self.entity_storage.create_entity(
                    obj_info=obj_info,
                    event_id=event_id,
                    obj_crops_paths=obj_crop_paths,
                    match_score=match_info,
                    unmatched_candidates=match_info.get("unmatched_candidates", {})
                )
            else:
                # 更新已有实体
                self.entity_storage.update_entity(
                    global_name=global_name,
                    obj_info=obj_info,
                    event_id=event_id,
                    obj_crops_paths=obj_crop_paths,
                    match_score=match_info
                )
            
            # 更新event JSON
            obj_info["global_name"] = global_name
            obj_info["match_info"] = match_info
        
        # 保存全局实体库
        self.entity_storage.save()
        
        # 保存修改后的event JSON
        if output_path is None:
            output_path = event_json_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(event_data, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def process_events_dir(
        self,
        events_dir: str,
        pattern: str = "*.json"
    ):
        """
        批量处理目录下的所有event JSON
        
        Args:
            events_dir: events目录路径
            pattern: 文件匹配模式
        """
        event_files = sorted(glob(os.path.join(events_dir, pattern)))
        
        print(f"[批量处理] {len(event_files)}个events")
        
        for i, event_file in enumerate(event_files, 1):
            self.process_event(event_file)
        
        print(f"[批量处理完成] 全局实体数: {len(self.entity_storage.entities)}")
    
    def reset_global(self):
        """重置全局实体库"""
        self.entity_storage.reset()
        print("[GlobalEntityManager] 全局实体库已重置")
    
    @property
    def global_entities(self) -> Dict:
        """获取全局实体字典"""
        return self.entity_storage.entities


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="全局实体管理器：为event中的objects分配global_name"
    )
    parser.add_argument(
        "event_json",
        help="单个event JSON文件路径"
    )
    parser.add_argument(
        "--config",
        default='/data/xuyuan/Egolife_env/ego-things/personalization/configs/config.yaml',
        help="配置文件路径"
    )
    parser.add_argument(
        "--events_dir",
        help="批量处理：events目录路径"
    )
    parser.add_argument(
        "--output",
        help="输出路径（仅单个event模式，默认覆盖原文件）"
    )
    parser.add_argument(
        "--reset_global",
        action="store_true",
        help="重置全局实体库"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="开启可视化，保存匹配过程中的bbox和IoU"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    from config import Config
    config = Config.from_yaml(args.config)
    
    # 初始化管理器
    manager = GlobalEntityManager(config, visualize=args.visualize)
    
    # 执行操作
    if args.reset_global:
        manager.reset_global()
    elif args.event_json:
        manager.process_event(args.event_json, args.output)
    elif args.events_dir:
        manager.process_events_dir(args.events_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

