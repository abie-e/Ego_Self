"""
BBox可视化工具

提供物体检测和追踪结果的可视化功能
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


class BBoxVisualizer:
    """可视化bbox检测和追踪结果"""
    
    # 颜色配置（BGR格式）
    COLORS = [
        (255, 0, 0),    # 蓝色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 品红
        (0, 255, 255),  # 黄色
    ]
    
    @staticmethod
    def draw_bbox_on_frame(
        frame: np.ndarray,
        bbox: List[float],
        label: Optional[str] = None,
        score: Optional[float] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        在单帧上绘制bbox
        
        Args:
            frame: BGR图像
            bbox: [x1, y1, x2, y2] 或 [x, y, w, h]格式
            label: 标签文本
            score: 置信度分数
            color: bbox颜色（BGR）
            thickness: 线条粗细
            
        Returns:
            绘制后的图像
        """
        frame = frame.copy()
        
        # 检查bbox格式，如果是[x, y, w, h]转换为[x1, y1, x2, y2]
        if len(bbox) == 4:
            x1, y1 = int(bbox[0]), int(bbox[1])
            if bbox[2] < frame.shape[1] * 0.5:  # 如果第三个值较小，认为是w
                x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
            else:  # 否则认为是x2
                x2, y2 = int(bbox[2]), int(bbox[3])
        else:
            return frame
        
        # 绘制矩形
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # 绘制标签
        if label or score is not None:
            text_parts = []
            if label:
                text_parts.append(label)
            if score is not None:
                text_parts.append(f"{score:.2f}")
            text = " ".join(text_parts)
            
            # 计算文本大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness=1
            )
            
            # 绘制文本背景
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # 绘制文本
            cv2.putText(
                frame,
                text,
                (x1, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        return frame
    
    @staticmethod
    def visualize_tracking_results(
        video_path: str,
        tracking_results: Dict[int, Dict],
        output_path: str,
        label: str = "object",
        fps: float = 30.0,
        show_score: bool = True
    ) -> None:
        """
        创建带bbox标注的追踪视频
        
        Args:
            video_path: 原始视频路径
            tracking_results: 追踪结果字典 {frame_idx: {"bbox": [...], "score": ...}}
            output_path: 输出视频路径
            label: 物体标签
            fps: 输出视频帧率
            show_score: 是否显示置信度分数
        """
        # 打开原始视频
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 如果该帧有追踪结果，绘制bbox
            if frame_idx in tracking_results:
                result = tracking_results[frame_idx]
                bbox = result["bbox"]
                score = result.get("score", None) if show_score else None
                
                frame = BBoxVisualizer.draw_bbox_on_frame(
                    frame=frame,
                    bbox=bbox,
                    label=label,
                    score=score,
                    color=(0, 255, 0),
                    thickness=2
                )
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        print(f"  [可视化] 保存追踪视频: {output_path}")
    
    @staticmethod
    def save_detection_grid(
        detections: List[Dict],
        frames: List[Dict],
        output_path: str,
        max_frames: int = 10
    ) -> None:
        """
        保存检测结果的网格可视化
        
        Args:
            detections: 检测结果列表
            frames: 帧数据列表
            output_path: 输出图片路径
            max_frames: 最大显示帧数
        """
        # 筛选有检测结果的帧
        valid_detections = [d for d in detections if len(d["boxes"]) > 0]
        if len(valid_detections) == 0:
            print("  [警告] 没有有效的检测结果")
            return
        
        # 限制帧数
        valid_detections = valid_detections[:max_frames]
        
        # 计算网格布局
        n = len(valid_detections)
        cols = min(5, n)
        rows = (n + cols - 1) // cols
        
        # 获取单帧大小
        sample_frame = frames[0]["image"]
        h, w = sample_frame.shape[:2]
        
        # 创建网格图像
        grid_h = h * rows + 10 * (rows - 1)
        grid_w = w * cols + 10 * (cols - 1)
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
        
        # 填充每个格子
        for idx, detection in enumerate(valid_detections):
            row = idx // cols
            col = idx % cols
            
            # 获取帧
            frame_idx = detection["frame_idx"]
            frame = frames[frame_idx]["image"].copy()
            
            # 绘制bbox
            if len(detection["boxes"]) > 0:
                bbox = detection["boxes"][0]
                score = float(detection["scores"][0])
                frame = BBoxVisualizer.draw_bbox_on_frame(
                    frame=frame,
                    bbox=bbox,
                    score=score,
                    color=(0, 255, 0),
                    thickness=2
                )
            
            # 放置到网格中
            y1 = row * (h + 10)
            x1 = col * (w + 10)
            grid[y1:y1+h, x1:x1+w] = frame
        
        # 保存
        cv2.imwrite(output_path, grid)
        print(f"  [可视化] 保存检测网格: {output_path}")
    
    @staticmethod
    def visualize_entity_data(
        entity_json_path: str,
        video_path: str,
        output_dir: str
    ) -> None:
        """
        从entity JSON可视化追踪结果
        
        Args:
            entity_json_path: entity JSON文件路径
            video_path: 原始视频路径
            output_dir: 输出目录
        """
        import json
        
        # 读取entity数据
        with open(entity_json_path, 'r', encoding='utf-8') as f:
            entity_data = json.load(f)
        
        event_id = entity_data["event_id"]
        obj_name = entity_data["object_name"]
        tracking_results = entity_data["tracking_results"]
        
        # 转换为所需格式
        tracking_dict = {
            r["frame_idx"]: {
                "bbox": r["bbox"],
                "score": r["score"]
            }
            for r in tracking_results
        }
        
        # 生成视频
        output_path = os.path.join(output_dir, f"{event_id}_{obj_name}_tracking.mp4")
        BBoxVisualizer.visualize_tracking_results(
            video_path=video_path,
            tracking_results=tracking_dict,
            output_path=output_path,
            label=obj_name,
            fps=30.0,
            show_score=True
        )









