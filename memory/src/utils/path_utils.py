"""
路径处理工具函数

提供统一的路径解析、构建和管理功能
"""

import os
from typing import Optional


def resolve_path_template(path_template: str, config) -> str:
    """
    解析路径模板中的变量
    
    Args:
        path_template: 路径模板字符串，可包含${var}变量
        config: 配置对象
    
    Returns:
        解析后的路径
    
    示例：
        resolve_path_template("${paths.entities_dir}/event", config)
        -> "/data/xuyuan/.../entities/event"
    """
    resolved = path_template
    
    # 构建替换字典
    replacements = {}
    
    # 添加常用配置变量
    if hasattr(config, 'entities_dir'):
        replacements["${paths.entities_dir}"] = config.entities_dir
    if hasattr(config, 'output_root'):
        replacements["${paths.output_root}"] = config.output_root
    if hasattr(config, 'data_root'):
        replacements["${paths.data_root}"] = config.data_root
    
    # 执行替换
    for var, value in replacements.items():
        if var in resolved:
            resolved = resolved.replace(var, value)
    
    return resolved


def get_day_prefix(event_id: str) -> str:
    """
    从event_id提取day前缀
    
    Args:
        event_id: 如 "DAY1_11094208_evt"
    
    Returns:
        day前缀，如 "DAY1"
    """
    return event_id.split('_')[0]


def build_event_entity_path(
    base_dir: str,
    event_id: str,
    ext: str = ".json"
) -> str:
    """
    构建事件级entity文件路径（新格式，一个事件一个文件）
    
    Args:
        base_dir: 基础目录
        event_id: event ID
        ext: 文件扩展名
    
    Returns:
        完整路径: {base_dir}/{DAY}/{event_id}{ext}
    """
    day_prefix = get_day_prefix(event_id)
    filename = f"{event_id}{ext}"
    return os.path.join(base_dir, day_prefix, filename)


def build_crop_dir(base_crops_dir: str, event_id: str, object_name: str) -> str:
    """
    构建crop图像目录路径
    
    Args:
        base_crops_dir: crops基础目录
        event_id: event ID
        object_name: object名称
    
    Returns:
        完整路径: {base_crops_dir}/{event_id}/{object_name}/
    """
    return os.path.join(base_crops_dir, event_id, object_name)


def ensure_dir(directory: str) -> None:
    """
    确保目录存在，不存在则创建
    
    Args:
        directory: 目录路径
    """
    os.makedirs(directory, exist_ok=True)



