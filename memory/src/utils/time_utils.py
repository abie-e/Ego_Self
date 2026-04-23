"""
时间戳解析工具

功能：
1. 从视频文件名解析时间信息
2. 生成event_id
3. 时间戳转换为可读格式
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Optional


def parse_video_filename(filename: str) -> Dict:
    """
    从视频文件名解析时间信息
    
    Args:
        filename: 视频文件名，如 "DAY1_A1_JAKE_11094208.mp4"
    
    Returns:
        时间信息字典:
        {
            "day": 1,
            "person": "A1_JAKE",
            "hour": 11,
            "minute": 9,
            "second": 42,
            "centisecond": 8,
            "timestamp": "DAY1_11094208"
        }
    """
    # 匹配格式: DAY{X}_{PERSON}_{HHMMSSMS}.mp4
    pattern = r"(DAY\d+)_([A-Z0-9_]+)_(\d{8})\.mp4"
    match = re.search(pattern, filename)
    
    if not match:
        raise ValueError(f"无法解析视频文件名: {filename}")
    
    day_str = match.group(1)  # "DAY1"
    person = match.group(2)    # "A1_JAKE"
    time_str = match.group(3)  # "11094208"
    
    # 解析day
    day = int(day_str.replace("DAY", ""))
    
    # 解析时间: HHMMSSMS
    hour = int(time_str[0:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])
    centisecond = int(time_str[6:8])  # 百分之一秒
    
    timestamp = f"{day_str}_{time_str}"
    
    return {
        "day": day,
        "person": person,
        "hour": hour,
        "minute": minute,
        "second": second,
        "centisecond": centisecond,
        "timestamp": timestamp,
        "filename": filename
    }


def generate_event_id(timestamp: str, sequence_index: Optional[int] = None) -> str:
    """
    生成event_id

    Args:
        timestamp: 时间戳，如 "DAY1_11094208"
        sequence_index: 可选的序号；为None时表示“一个视频一个event”

    Returns:
        event_id，如 "DAY1_11094208_evt" 或 "DAY1_11094208_evt_0005"
    """

    if sequence_index is None:
        return f"{timestamp}_evt"
    return f"{timestamp}_evt_{sequence_index:04d}"


def timestamp_to_datetime(timestamp: str, base_date: str = "2024-01-01") -> str:
    """
    将timestamp转为可读的datetime字符串
    
    Args:
        timestamp: 时间戳，如 "DAY1_11094208"
        base_date: 基准日期字符串，格式 "YYYY-MM-DD"
    
    Returns:
        可读时间字符串，如 "2024-01-01 11:09:42.08"
    """
    # 解析timestamp
    parts = timestamp.split("_")
    day_str = parts[0]  # "DAY1"
    time_str = parts[1]  # "11094208"
    
    day = int(day_str.replace("DAY", ""))
    hour = int(time_str[0:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])
    centisecond = int(time_str[6:8])
    
    # 计算实际日期
    base = datetime.strptime(base_date, "%Y-%m-%d")
    actual_date = base + timedelta(days=day - 1)  # DAY1对应第0天
    
    # 添加时间
    dt = actual_date.replace(hour=hour, minute=minute, second=second, microsecond=centisecond * 10000)
    
    # 格式化输出
    return dt.strftime("%Y-%m-%d %H:%M:%S") + f".{centisecond:02d}"


def timestamp_to_seconds(timestamp: str) -> float:
    """
    将timestamp转为从DAY1开始的总秒数（用于时间范围比较）
    
    Args:
        timestamp: 时间戳，如 "DAY1_11094208"
    
    Returns:
        总秒数（浮点数）
    """
    parts = timestamp.split("_")
    day_str = parts[0]
    time_str = parts[1]
    
    day = int(day_str.replace("DAY", ""))
    hour = int(time_str[0:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])
    centisecond = int(time_str[6:8])
    
    # 计算总秒数
    total_seconds = (day - 1) * 24 * 3600  # 天数的秒数
    total_seconds += hour * 3600
    total_seconds += minute * 60
    total_seconds += second
    total_seconds += centisecond / 100.0  # 百分之一秒转为秒
    
    return total_seconds


# 测试代码
if __name__ == "__main__":
    print("=== Time Utils Test ===\n")
    
    # 测试1: 解析文件名
    filename = "DAY1_A1_JAKE_11094208.mp4"
    info = parse_video_filename(filename)
    print(f"Filename: {filename}")
    print(f"Parsed: {info}")
    assert info["day"] == 1
    assert info["person"] == "A1_JAKE"
    assert info["hour"] == 11
    assert info["minute"] == 9
    assert info["second"] == 42
    assert info["centisecond"] == 8
    print("✓ Parse filename test passed\n")
    
    # 测试2: 生成event_id
    timestamp = "DAY1_11094208"
    event_id = generate_event_id(timestamp, 5)
    print(f"Timestamp: {timestamp}, Frame: 5")
    print(f"Event ID: {event_id}")
    assert event_id == "DAY1_11094208_evt_0005"
    print("✓ Generate event_id test passed\n")

    # 测试2.1: 无序号（单视频单event）
    no_index_event = generate_event_id(timestamp)
    print(f"Timestamp: {timestamp}, No index")
    print(f"Event ID: {no_index_event}")
    assert no_index_event == "DAY1_11094208_evt"
    print("✓ Generate event_id without index passed\n")
    
    # 测试3: 时间戳转datetime
    readable_time = timestamp_to_datetime(timestamp)
    print(f"Timestamp: {timestamp}")
    print(f"Readable: {readable_time}")
    assert "2024-01-01 11:09:42.08" == readable_time
    print("✓ Timestamp to datetime test passed\n")
    
    # 测试4: 时间戳转秒数
    seconds = timestamp_to_seconds(timestamp)
    print(f"Timestamp: {timestamp}")
    print(f"Total seconds from DAY1 00:00:00: {seconds}")
    expected = 11 * 3600 + 9 * 60 + 42 + 0.08
    assert abs(seconds - expected) < 0.01
    print("✓ Timestamp to seconds test passed\n")
    
    # 测试5: 跨天时间戳
    timestamp2 = "DAY2_12000000"
    seconds2 = timestamp_to_seconds(timestamp2)
    readable_time2 = timestamp_to_datetime(timestamp2)
    print(f"Timestamp: {timestamp2}")
    print(f"Readable: {readable_time2}")
    print(f"Total seconds: {seconds2}")
    expected2 = 24 * 3600 + 12 * 3600
    assert abs(seconds2 - expected2) < 0.01
    print("✓ Cross-day test passed\n")
    
    print("=== All tests passed! ===")


