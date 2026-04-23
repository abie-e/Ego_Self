"""
匹配过程可视化工具

用于可视化global entity matching过程中的entity对比
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


class MatchVisualizer:
    """匹配过程可视化器：绘制Event Entity和Global Entity的对比"""
    
    def __init__(self, vis_dir: str):
        """
        初始化可视化器
        
        Args:
            vis_dir: 可视化结果输出目录
        """
        self.vis_dir = vis_dir
        os.makedirs(vis_dir, exist_ok=True)
    
    def visualize_aligned_match(
        self,
        event_id: str,
        event_entity_name: str,
        event_entity_desc: str,
        global_entity_name: Optional[str],
        global_entity_desc: Optional[str],
        vis_data: Dict,
        is_match: bool,
        text_similarity: float
    ):
        """
        可视化时序对齐的匹配结果（在同一帧上画两个bbox）
        
        这个方法用于可视化vision_matcher返回的时序对齐数据：
        - 在new_object的视频帧上画两个bbox
        - 绿色bbox: new_object的tracking bbox
        - 蓝色bbox: 用global_entity描述检测出的bbox
        
        Args:
            event_id: 事件ID
            event_entity_name: event实体名称
            event_entity_desc: event实体描述
            global_entity_name: 全局实体名称（None表示新实体）
            global_entity_desc: 全局实体描述
            vis_data: vision_matcher返回的可视化数据，包含：
                     - frames: 视频帧列表
                     - entity_bboxes: new_object的bbox列表(xywh格式)
                     - detected_bboxes: 检测出的bbox列表(xyxy格式)
                     - ious: IoU值列表
                     - timestamps: 时间戳列表
            is_match: 是否匹配成功
            text_similarity: 文本相似度
        """
        if not vis_data or not vis_data.get("frames"):
            return
        
        frames = vis_data["frames"]
        entity_bboxes = vis_data["entity_bboxes"]  # new_object的bbox (xywh)
        detected_bboxes = vis_data["detected_bboxes"]  # 检测出的bbox (xyxy)
        ious = vis_data["ious"]
        timestamps = vis_data.get("timestamps", [])
        
        # 为每一帧生成可视化
        for frame_idx, (frame, new_obj_bbox, detected_boxes, iou) in enumerate(
            zip(frames, entity_bboxes, detected_bboxes, ious)
        ):
            self._visualize_single_frame_aligned(
                event_id=event_id,
                event_entity_name=event_entity_name,
                event_entity_desc=event_entity_desc,
                global_entity_name=global_entity_name if is_match else "new",
                global_entity_desc=global_entity_desc if global_entity_desc else "NEW ENTITY",
                frame=frame.copy(),
                new_obj_bbox=new_obj_bbox,
                detected_boxes=detected_boxes,
                iou=iou,
                is_match=is_match,
                text_similarity=text_similarity,
                frame_idx=frame_idx,
                timestamp=timestamps[frame_idx] if frame_idx < len(timestamps) else 0
            )
    
    def _visualize_single_frame_aligned(
        self,
        event_id: str,
        event_entity_name: str,
        event_entity_desc: str,
        global_entity_name: str,
        global_entity_desc: str,
        frame: np.ndarray,
        new_obj_bbox: List[float],
        detected_boxes: List[List[float]],
        iou: float,
        is_match: bool,
        text_similarity: float,
        frame_idx: int,
        timestamp: float
    ):
        """
        在单帧上绘制两个bbox的对比可视化
        
        Args:
            frame: 视频帧
            new_obj_bbox: new_object的bbox (xywh格式)
            detected_boxes: 检测出的bbox列表 (xyxy格式)
            iou: IoU值
            其他参数: 用于显示和保存
        """
        # ===== 在原始帧上绘制两个bbox =====
        h, w = frame.shape[:2]
        
        # 绘制new_object的bbox（绿色）
        x, y, bw, bh = new_obj_bbox
        x1, y1, x2, y2 = int(x), int(y), int(x+bw), int(y+bh)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 绿色，粗线
        cv2.putText(frame, "Event Entity", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # 显示bbox坐标
        bbox_text = f"({x1},{y1},{x2},{y2})"
        cv2.putText(frame, bbox_text, (x1, y2+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制检测出的global entity bbox（蓝色）
        if detected_boxes and len(detected_boxes) > 0:
            # 只绘制最佳匹配的bbox（第一个）
            det_box = detected_boxes[0]
            dx1, dy1, dx2, dy2 = [int(v) for v in det_box]
            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (255, 0, 0), 3)  # 蓝色，粗线
            cv2.putText(frame, "Global Entity (Detected)", (dx1, dy1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            # 显示bbox坐标和IoU
            det_bbox_text = f"({dx1},{dy1},{dx2},{dy2})"
            cv2.putText(frame, det_bbox_text, (dx1, dy2+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"IoU: {iou:.3f}", (dx1, dy2+55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # ===== 在图像下方添加信息和图例 =====
        # 计算扩展画布大小（下方留出空间显示信息）
        info_height = 300
        canvas_h = h + info_height
        canvas = np.ones((canvas_h, w, 3), dtype=np.uint8) * 40  # 深灰色背景
        canvas[:h, :w] = frame
        
        # 绘制分隔线
        cv2.line(canvas, (0, h), (w, h), (255, 255, 255), 2)
        
        # 在下方显示信息
        info_y = h + 40
        line_height = 45
        
        # 描述信息
        self._put_text_wrapped(canvas, f"Event Entity: {event_entity_desc}", 
                              (20, info_y), 0.8, (255, 255, 255), 2, max_width=w-40)
        info_y += line_height
        
        self._put_text_wrapped(canvas, f"Global Entity: {global_entity_desc}", 
                              (20, info_y), 0.8, (255, 255, 255), 2, max_width=w-40)
        info_y += line_height
        
        # 相似度信息
        match_text = "MATCH" if is_match else "NO MATCH"
        match_color = (0, 255, 0) if is_match else (0, 0, 255)
        cv2.putText(canvas, f"Status: {match_text}", (20, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, match_color, 2)
        info_y += line_height
        
        cv2.putText(canvas, f"Text Sim: {text_similarity:.3f} | IoU: {iou:.3f}", 
                   (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 绘制图例（右下角）
        legend_x = w - 350
        legend_y = h + 40
        cv2.rectangle(canvas, (legend_x-10, legend_y-30), (w-10, h+info_height-10), (80, 80, 80), 2)
        cv2.putText(canvas, "Legend:", (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        legend_y += 40
        cv2.rectangle(canvas, (legend_x, legend_y-15), (legend_x+25, legend_y+10), (0, 255, 0), -1)
        cv2.putText(canvas, "Event Entity", (legend_x+35, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        legend_y += 40
        cv2.rectangle(canvas, (legend_x, legend_y-15), (legend_x+25, legend_y+10), (255, 0, 0), -1)
        cv2.putText(canvas, "Global Entity", (legend_x+35, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 保存可视化结果
        self._save_aligned_visualization(
            canvas=canvas,
            event_id=event_id,
            event_entity_name=event_entity_name,
            global_entity_name=global_entity_name,
            is_match=is_match,
            iou=iou,
            frame_idx=frame_idx,
            new_obj_bbox=new_obj_bbox,
            detected_boxes=detected_boxes
        )
    
    def _put_text_wrapped(self, img, text, pos, font_scale, color, thickness, max_width):
        """在图像上绘制自动换行的文本（简化版）"""
        if len(text) * font_scale * 15 < max_width:
            # 文本不长，直接绘制
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        else:
            # 文本过长，截断并加省略号
            cv2.putText(img, text[:int(max_width / (font_scale * 15))] + "...", 
                       pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    def _save_aligned_visualization(
        self,
        canvas: np.ndarray,
        event_id: str,
        event_entity_name: str,
        global_entity_name: str,
        is_match: bool,
        iou: float,
        frame_idx: int,
        new_obj_bbox: List[float],
        detected_boxes: List[List[float]]
    ):
        """保存时序对齐的可视化结果"""
        # 构建输出路径
        match_status = "match" if is_match else "no_match"
        entity_folder = f"{event_entity_name}_{global_entity_name}"
        sim_folder = f"{global_entity_name}_{match_status}_iou_{iou:.3f}"
        
        output_dir = os.path.join(
            self.vis_dir,
            event_id,
            entity_folder,
            sim_folder
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图像
        img_filename = f"frame_{frame_idx}.png"
        img_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(img_path, canvas)
        
        # 保存bbox和IoU信息到txt文件
        txt_filename = "bbox_info.txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        mode = 'w' if frame_idx == 0 else 'a'
        with open(txt_path, mode, encoding='utf-8') as f:
            if frame_idx == 0:
                f.write("# Bbox and IoU Information (Temporal Aligned)\n")
                f.write("# Format: frame_idx | event_bbox(x1,y1,x2,y2) | detected_bbox(x1,y1,x2,y2) | IoU\n")
                f.write("-" * 80 + "\n")
            
            # 转换bbox为x1,y1,x2,y2格式
            x, y, bw, bh = new_obj_bbox
            event_bbox_str = f"({int(x)},{int(y)},{int(x+bw)},{int(y+bh)})"
            
            if detected_boxes and len(detected_boxes) > 0:
                dx1, dy1, dx2, dy2 = [int(v) for v in detected_boxes[0]]
                detected_bbox_str = f"({dx1},{dy1},{dx2},{dy2})"
            else:
                detected_bbox_str = "None"
            
            # 写入当前帧信息
            line = f"Frame {frame_idx}: Event={event_bbox_str} | Detected={detected_bbox_str} | IoU={iou:.4f}\n"
            f.write(line)
        
        print(f"    [可视化已保存] {img_path}")
    
    def visualize_entity_match(
        self,
        event_id: str,
        event_entity_name: str,
        event_entity_desc: str,
        event_entity_frame: np.ndarray,
        event_entity_bbox: List[float],
        global_entity_name: Optional[str],
        global_entity_desc: Optional[str],
        global_entity_frame: Optional[np.ndarray],
        global_entity_bbox: Optional[List[float]],
        is_match: bool,
        similarity_scores: Dict[str, float],
        frame_idx: int = 0
    ):
        """
        可视化Event Entity与Global Entity的匹配对比
        
        Args:
            event_id: 事件ID
            event_entity_name: event实体名称(如human1, object1)
            event_entity_desc: event实体描述
            event_entity_frame: event实体的帧图像
            event_entity_bbox: event实体的bbox (xywh格式)
            global_entity_name: 全局实体名称(如person_001)，None表示新实体
            global_entity_desc: 全局实体描述
            global_entity_frame: 全局实体的代表图像
            global_entity_bbox: 全局实体的bbox (xywh格式)
            is_match: 是否匹配成功
            similarity_scores: 相似度分数 {"text": 0.5, "vision": 0.8, "final": 0.8}
            frame_idx: 帧索引
        """
        # 创建左右分栏的可视化图像
        img_height = 800
        img_width = 1600
        canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 40  # 深灰色背景
        
        # 定义分栏区域
        left_x, left_y, left_w, left_h = 50, 150, 700, 600  # Event Entity区域
        right_x, right_y, right_w, right_h = 850, 150, 700, 600  # Global Entity区域
        
        # 获取IoU值（即vision_similarity）
        iou_value = similarity_scores.get("vision_similarity", 0.0)
        
        # ===== 绘制Event Entity (左侧) =====
        # 调整并绘制event entity的帧图像
        event_frame_resized = self._resize_and_pad(event_entity_frame, left_w, left_h)
        # 在调整后的图像上绘制bbox（不显示IoU，因为IoU是两个bbox之间的）
        event_frame_with_bbox = self._draw_bbox_on_image(
            event_frame_resized,
            event_entity_bbox,
            label="Event Entity",
            color=(0, 255, 0),  # 绿色
            original_frame_size=event_entity_frame.shape[:2],
            iou=None  # Event Entity不显示IoU
        )
        canvas[left_y:left_y+left_h, left_x:left_x+left_w] = event_frame_with_bbox
        
        # ===== 绘制Global Entity (右侧) =====
        if global_entity_name and global_entity_frame is not None:
            # 有匹配的全局实体，显示IoU
            global_frame_resized = self._resize_and_pad(global_entity_frame, right_w, right_h)
            global_frame_with_bbox = self._draw_bbox_on_image(
                global_frame_resized,
                global_entity_bbox,
                label="Global Entity",
                color=(255, 0, 0),  # 蓝色
                original_frame_size=global_entity_frame.shape[:2],
                iou=iou_value  # 显示IoU值
            )
            canvas[right_y:right_y+right_h, right_x:right_x+right_w] = global_frame_with_bbox
        else:
            # 新实体，显示"NEW"
            self._draw_new_entity_placeholder(canvas, right_x, right_y, right_w, right_h)
        
        # ===== 添加标题和描述文本 =====
        # 标题
        match_status = "MATCH" if is_match else "NO MATCH"
        status_color = (0, 255, 0) if is_match else (0, 0, 255)  # 绿色=匹配，红色=不匹配
        self._draw_text(
            canvas,
            f"Entity Matching: {match_status}",
            (img_width // 2 - 200, 30),
            font_scale=1.2,
            color=status_color,
            thickness=3
        )
        
        # Event Entity描述（左侧标题）
        self._draw_text(
            canvas,
            f"Event: {event_entity_name}",
            (left_x, left_y - 60),
            font_scale=0.9,
            color=(255, 255, 255),
            thickness=2
        )
        self._draw_multiline_text(
            canvas,
            event_entity_desc,
            (left_x, left_y - 35),
            font_scale=0.6,
            color=(200, 200, 200),
            max_width=left_w,
            line_height=25
        )
        
        # Global Entity描述（右侧标题）
        if global_entity_name:
            self._draw_text(
                canvas,
                f"Global: {global_entity_name}",
                (right_x, right_y - 60),
                font_scale=0.9,
                color=(255, 255, 255),
                thickness=2
            )
            if global_entity_desc:
                self._draw_multiline_text(
                    canvas,
                    global_entity_desc,
                    (right_x, right_y - 35),
                    font_scale=0.6,
                    color=(200, 200, 200),
                    max_width=right_w,
                    line_height=25
                )
        else:
            self._draw_text(
                canvas,
                "NEW ENTITY",
                (right_x, right_y - 60),
                font_scale=0.9,
                color=(0, 255, 255),  # 黄色
                thickness=2
            )
        
        # ===== 添加相似度分数 =====
        self._draw_similarity_scores(canvas, similarity_scores, img_width // 2 - 200, 80)
        
        # ===== 添加图例 =====
        self._draw_legend_large(canvas, 50, img_height - 100)
        
        # ===== 保存图像和bbox信息 =====
        self._save_entity_match_visualization(
            canvas,
            event_id,
            event_entity_name,
            global_entity_name if global_entity_name else "new",
            is_match,
            similarity_scores.get("final", 0.0),
            frame_idx,
            event_bbox=event_entity_bbox,
            global_bbox=global_entity_bbox,
            iou=iou_value
        )
    
    def _resize_and_pad(self, frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """调整图像大小并padding，保持宽高比"""
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # 创建padding后的图像
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def _draw_bbox_on_image(
        self,
        frame: np.ndarray,
        bbox: List[float],
        label: str,
        color: Tuple[int, int, int],
        original_frame_size: Tuple[int, int],
        iou: Optional[float] = None
    ) -> np.ndarray:
        """在调整后的图像上绘制bbox和坐标信息（需要根据缩放比例调整坐标）"""
        result = frame.copy()
        
        # 计算缩放比例
        orig_h, orig_w = original_frame_size
        curr_h, curr_w = frame.shape[:2]
        scale_x = curr_w / orig_w
        scale_y = curr_h / orig_h
        
        # 转换bbox坐标
        x, y, w, h = bbox
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)
        
        # 绘制bbox矩形
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
        
        # 绘制标签（bbox上方）
        cv2.putText(
            result,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )
        
        # 绘制坐标信息（bbox下方）
        coord_text = f"({x1}, {y1}, {x2}, {y2})"
        cv2.putText(
            result,
            coord_text,
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )
        
        # 如果有IoU信息，也显示在bbox下方
        if iou is not None:
            iou_text = f"IoU: {iou:.3f}"
            cv2.putText(
                result,
                iou_text,
                (x1, y2 + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )
        
        return result
    
    def _draw_new_entity_placeholder(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int
    ):
        """绘制新实体的占位符"""
        # 绘制虚线框
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (100, 100, 100), 2, cv2.LINE_AA)
        
        # 绘制"NEW"文字
        text = "NEW ENTITY"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(
            canvas,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 255),  # 黄色
            thickness,
            cv2.LINE_AA
        )
    
    def _draw_text(
        self,
        canvas: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 0.8,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2
    ):
        """绘制单行文本"""
        cv2.putText(
            canvas,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
    
    def _draw_multiline_text(
        self,
        canvas: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 0.6,
        color: Tuple[int, int, int] = (255, 255, 255),
        max_width: int = 700,
        line_height: int = 25
    ):
        """绘制多行文本（自动换行）"""
        x, y = position
        words = text.split()
        lines = []
        current_line = []
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            text_size = cv2.getTextSize(test_line, font, font_scale, 1)[0]
            if text_size[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # 绘制每一行
        for i, line in enumerate(lines[:2]):  # 最多显示2行
            cv2.putText(
                canvas,
                line,
                (x, y + i * line_height),
                font,
                font_scale,
                color,
                1,
                cv2.LINE_AA
            )
    
    def _draw_similarity_scores(
        self,
        canvas: np.ndarray,
        scores: Dict[str, float],
        x: int,
        y: int
    ):
        """绘制相似度分数"""
        text_sim = scores.get("text_similarity", 0.0)
        vision_sim = scores.get("vision_similarity", 0.0)
        final_sim = scores.get("final_score", 0.0)
        
        score_text = f"Scores - Text: {text_sim:.3f}  Vision: {vision_sim:.3f}  Final: {final_sim:.3f}"
        self._draw_text(
            canvas,
            score_text,
            (x, y),
            font_scale=0.8,
            color=(255, 255, 0),  # 黄色
            thickness=2
        )
    
    def _draw_legend_large(self, canvas: np.ndarray, x: int, y: int):
        """绘制大号图例"""
        legends = [
            ("Event Entity bbox", (0, 255, 0)),     # 绿色
            ("Global Entity bbox", (255, 0, 0)),    # 蓝色
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        box_size = 30
        spacing = 250
        
        for i, (label, color) in enumerate(legends):
            curr_x = x + i * spacing
            
            # 绘制色块
            cv2.rectangle(
                canvas,
                (curr_x, y),
                (curr_x + box_size, y + box_size),
                color,
                -1
            )
            cv2.rectangle(
                canvas,
                (curr_x, y),
                (curr_x + box_size, y + box_size),
                (255, 255, 255),
                2
            )
            
            # 绘制文本
            cv2.putText(
                canvas,
                label,
                (curr_x + box_size + 10, y + box_size - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )
    
    def _save_entity_match_visualization(
        self,
        canvas: np.ndarray,
        event_id: str,
        event_entity_name: str,
        global_entity_name: str,
        is_match: bool,
        similarity: float,
        frame_idx: int,
        event_bbox: List[float],
        global_bbox: Optional[List[float]],
        iou: float
    ):
        """保存实体匹配可视化结果（包括图像和txt信息）"""
        # 构建输出路径
        match_status = "match" if is_match else "no_match"
        entity_folder = f"{event_entity_name}_{global_entity_name}"
        sim_folder = f"{global_entity_name}_{match_status}_sim_{similarity:.3f}"
        
        output_dir = os.path.join(
            self.vis_dir,
            event_id,
            entity_folder,
            sim_folder
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图像
        img_filename = f"frame_{frame_idx}.png"
        img_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(img_path, canvas)
        
        # 保存bbox和IoU信息到txt文件（追加模式）
        txt_filename = "bbox_info.txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        # 如果是第一帧，创建新文件并写入表头
        mode = 'w' if frame_idx == 0 else 'a'
        with open(txt_path, mode, encoding='utf-8') as f:
            if frame_idx == 0:
                f.write("# Bbox and IoU Information\n")
                f.write("# Format: frame_idx | event_bbox(x1,y1,x2,y2) | global_bbox(x1,y1,x2,y2) | IoU\n")
                f.write("-" * 80 + "\n")
            
            # 转换bbox为x1,y1,x2,y2格式
            event_x1, event_y1, event_w, event_h = event_bbox
            event_x2 = event_x1 + event_w
            event_y2 = event_y1 + event_h
            
            if global_bbox is not None:
                global_x1, global_y1, global_w, global_h = global_bbox
                global_x2 = global_x1 + global_w
                global_y2 = global_y1 + global_h
                global_bbox_str = f"({global_x1:.0f},{global_y1:.0f},{global_x2:.0f},{global_y2:.0f})"
            else:
                global_bbox_str = "None"
            
            # 写入当前帧信息
            line = f"Frame {frame_idx}: Event=({event_x1:.0f},{event_y1:.0f},{event_x2:.0f},{event_y2:.0f}) | Global={global_bbox_str} | IoU={iou:.4f}\n"
            f.write(line)
        
        print(f"  [可视化已保存] {img_path}")







