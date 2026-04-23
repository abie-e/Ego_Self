"""
Event Relation Processor - 事件关系判断处理器

功能：
1. 查找历史Events（时间窗口内）
2. 判断关系（调用Gemini，包含3类：causal_same_strong, same_activity_non_causal, no_relationship）
3. 独立存储所有判断结果到data/relationships/（包括no_relationship）
4. 只有causal_same_strong和same_activity_non_causal才更新Event的relations字段（双向更新）

Usage:
    python -m src.event.event_relation_processor --config configs/config.yaml --event DAY1_11113000_evt
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional

from config import Config
from api.gemini_client import GeminiClient
from .event_storage import EventStorage

# 添加src目录到路径
src_dir = os.path.dirname(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 不再需要save_json_compact，直接使用标准json


class EventRelationProcessor:
    """Event关系判断处理器"""
    
    def __init__(self, config: Config):
        """
        初始化Event关系判断处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.event_storage = EventStorage(config.events_dir, config.features_dir)
        
        # 初始化Gemini客户端（用于关系判断，使用可配置的模型）
        relation_config = config.get_relation_config()
        self.gemini_client = GeminiClient(
            api_key=relation_config["api_key"],
            base_url=relation_config["base_url"],
            config=relation_config,
            temp_dir=config.temp_dir
        )
        
        # 加载prompt（自动使用配置的client）
        self.system_message, self.relation_prompt = config.get_prompt("event_relation")
        
        # 关系判断配置
        self.window_size = config.relation_window_size
        self.time_threshold = config.relation_time_threshold
        self.verbose = config.verbose
        
        # 创建relationships目录
        self.relationships_dir = os.path.join(config.output_root, "relationships")
        os.makedirs(self.relationships_dir, exist_ok=True)
    
    def process_event(self, event_id: str, day: int) -> Dict:
        """
        处理单个Event的关系判断
        
        Args:
            event_id: Event ID
            day: 天数
        
        Returns:
            包含关系判断结果的字典
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"处理Event关系: {event_id}")
            print(f"{'='*60}")
        
        # 1. 加载当前Event
        current_event = self.event_storage.load_event(event_id, day)
        current_time = current_event["attributes"]["interaction_time"]
        
        if self.verbose:
            print(f"\n当前事件时间: {current_time}")
        
        # 2. 查找历史Events（时间窗口内）
        historical_events = self.find_historical_events(current_time, day)
        
        if not historical_events:
            if self.verbose:
                print("没有找到历史事件，跳过关系判断")
            # 仍然保存一个空的关系判断文件
            self._save_relation_judgment(event_id, current_event, [])
            return {"event_id": event_id, "relations": []}
        
        if self.verbose:
            print(f"找到 {len(historical_events)} 个历史事件")
        
        # 3. 判断所有关系（调用Gemini，包括no_relationship）
        all_judgments = self.judge_relations(current_event, historical_events)
        
        # 4. 保存完整的关系判断结果（包括no_relationship）
        self._save_relation_judgment(event_id, current_event, all_judgments)
        
        # 5. 过滤出有效的因果/活动关系（排除no_relationship）
        causal_activity_relations = [
            rel for rel in all_judgments 
            if rel["type"] not in ["no_relationship", "no_causal", "no_temporal"]
        ]
        
        # 6. 为所有历史Event添加时序关系（独立于LLM判断，独立存储）
        current_time_str = current_event["attributes"]["interaction_time"]
        temporal_relations = []
        
        for hist_event in historical_events:
            hist_time_str = hist_event["attributes"]["interaction_time"]
            time_diff = self._calculate_time_diff(hist_time_str, current_time_str)
            
            # 只为时间差在阈值内的事件添加时序关系
            if 0 < time_diff <= self.time_threshold:
                # 时序关系独立存储，不附加在因果关系上
                temporal_relations.append({
                    "related_event_id": hist_event["event_id"],
                    "type": "temporal_after",
                    "time_diff_seconds": time_diff,
                    "reason": f"{hist_time_str} -> {current_time_str}"
                })
        
        # 7. 合并所有有效关系（因果/活动 + 纯时序）
        all_valid_relations = causal_activity_relations + temporal_relations
        
        # 8. 更新当前Event的relations字段（直接覆盖）
        # 注意：只更新当前事件，不更新历史事件的relations
        # 每个事件的relations只在该事件被处理时确定
        if all_valid_relations:
            current_event["relations"] = all_valid_relations
            self.event_storage.save_event(current_event)
        
        if self.verbose:
            print(f"\n完成！总判断数: {len(all_judgments)}, 因果/活动关系: {len(causal_activity_relations)}, 纯时序关系: {len(temporal_relations)}")
            print(f"{'='*60}\n")
        
        return {
            "event_id": event_id,
            "all_judgments": all_judgments,
            "causal_activity_relations": causal_activity_relations,
            "temporal_relations": temporal_relations,
            "all_valid_relations": all_valid_relations
        }
    
    def find_historical_events(self, current_time: str, day: int) -> List[Dict]:
        """
        查找历史Events（时间窗口内的前N个events）
        
        Args:
            current_time: 当前Event的时间字符串
            day: 天数
        
        Returns:
            历史Events列表（按时间排序）
        """
        current_dt = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S.%f")
        
        # 扫描同一天的所有Events
        day_dir = os.path.join(self.config.events_dir, f"DAY{day}")
        if not os.path.exists(day_dir):
            return []
        
        # 收集所有在当前Event之前的Events
        historical_events = []
        for filename in os.listdir(day_dir):
            if not filename.endswith(".json"):
                continue
            
            event_data = self.event_storage.load_event(
                filename.replace(".json", ""), day
            )
            event_time = event_data["attributes"]["interaction_time"]
            event_dt = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S.%f")
            
            # 只保留在当前Event之前的Events
            if event_dt < current_dt:
                historical_events.append({
                    "event": event_data,
                    "time": event_dt
                })
        
        # 按时间排序，取最近的N个
        historical_events.sort(key=lambda x: x["time"], reverse=True)
        historical_events = historical_events[:self.window_size]
        
        # 返回Event数据（按时间从早到晚排序）
        return [item["event"] for item in reversed(historical_events)]
    
    def judge_relations(
        self, 
        current_event: Dict, 
        historical_events: List[Dict]
    ) -> List[Dict]:
        """
        判断所有关系（调用Gemini）
        返回所有判断结果，包括no_relationship
        
        Args:
            current_event: 当前Event数据
            historical_events: 历史Events列表
        
        Returns:
            关系列表（包含no_relationship）
        """
        if self.verbose:
            print(f"\n调用Gemini判断因果关系...")
        
        # 构建prompt
        prompt = self._build_causal_prompt(current_event, historical_events)
        
        # 调用Gemini
        response = self.gemini_client.generate_json(
            prompt=prompt,
            system_message=self.system_message
        )
        
        # 解析结果（期望返回一个数组）
        if isinstance(response, list):
            relations = []
            for item in response:
                relations.append({
                    "type": item["relation_type"],
                    "related_event_id": item["historical_event_id"],
                    "reason": item["reason"]
                })
            return relations
        else:
            return []
    
    def _save_relation_judgment(
        self,
        event_id: str,
        current_event: Dict,
        all_judgments: List[Dict]
    ):
        """
        保存完整的关系判断结果到data/relationships/
        
        Args:
            event_id: Event ID
            current_event: 当前Event数据
            all_judgments: 所有判断结果（包括no_relationship）
        """
        # 构建判断结果JSON
        judgment_data = {
            "event_id": event_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_event": {
                "event_id": current_event["event_id"],
                "time": current_event["attributes"]["interaction_time"],
                "caption": current_event["caption"],
                "location": current_event["attributes"]["interaction_location"]
            },
            "judgments": all_judgments,
            "summary": {
                "total_judgments": len(all_judgments),
                "causal_same_strong": len([j for j in all_judgments if j["type"] == "causal_same_strong"]),
                "same_activity_non_causal": len([j for j in all_judgments if j["type"] == "same_activity_non_causal"]),
                "no_relationship": len([j for j in all_judgments if j["type"] == "no_relationship"])
            }
        }
        
        # 保存到文件（标准格式）
        output_path = os.path.join(self.relationships_dir, f"{event_id}.json")
        # 使用标准json.dump保存判断结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(judgment_data, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"保存关系判断结果到: {output_path}")
    
    def _build_causal_prompt(
        self, 
        current_event: Dict, 
        historical_events: List[Dict]
    ) -> str:
        """
        构建关系判断的prompt
        
        Args:
            current_event: 当前Event数据
            historical_events: 历史Events列表
        
        Returns:
            Prompt字符串
        """
        # 提取当前Event信息
        current_attrs = current_event["attributes"]
        current_info = {
            "current_event_id": current_event["event_id"],
            "current_time": current_attrs["interaction_time"],
            "current_caption": current_event["caption"],
            "current_location": current_attrs["interaction_location"]
        }
        
        # 构建历史Events信息（只包含必要字段）
        hist_events_text = ""
        for idx, hist_event in enumerate(historical_events, 1):
            hist_attrs = hist_event["attributes"]
            hist_events_text += f"""### Event {idx}
**Event ID**: {hist_event["event_id"]}
**Time**: {hist_attrs["interaction_time"]}
**Caption**: {hist_event["caption"]}
**Location**: {hist_attrs["interaction_location"]}

"""
        
        current_info["historical_events"] = hist_events_text.strip()
        
        # 填充prompt模板
        return self.relation_prompt.format(**current_info)
    
    def _calculate_time_diff(self, time1: str, time2: str) -> float:
        """
        计算两个时间的差值（秒）
        
        Args:
            time1: 时间字符串1
            time2: 时间字符串2
        
        Returns:
            时间差（秒）
        """
        dt1 = datetime.strptime(time1, "%Y-%m-%d %H:%M:%S.%f")
        dt2 = datetime.strptime(time2, "%Y-%m-%d %H:%M:%S.%f")
        return (dt2 - dt1).total_seconds()


def main():
    """测试Event关系处理器"""
    parser = argparse.ArgumentParser(description="测试Event关系处理")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--event", type=str, default="DAY1_11113000_evt", help="Event ID (例如: DAY1_11113000_evt)")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config(args.config)
    if args.verbose:
        config.verbose = True
    
    # 从Event ID中提取DAY信息
    if not args.event.startswith("DAY"):
        print(f"错误：Event ID格式不正确，应以DAY开头: {args.event}")
        sys.exit(1)
    
    try:
        day = int(args.event.split("_")[0].replace("DAY", ""))
    except (IndexError, ValueError):
        print(f"错误：无法从Event ID中提取DAY信息: {args.event}")
        sys.exit(1)
    
    # 检查Event文件是否存在
    event_file = os.path.join(config.events_dir, f"DAY{day}", f"{args.event}.json")
    if not os.path.exists(event_file):
        print(f"错误：Event文件不存在: {event_file}")
        sys.exit(1)
    
    print(f"{'='*60}")
    print(f"测试Event关系处理")
    print(f"{'='*60}")
    print(f"配置文件: {args.config}")
    print(f"Event ID: {args.event}")
    print(f"Event文件: {event_file}")
    print(f"Day: {day}")
    print(f"{'='*60}\n")
    
    # 创建处理器
    processor = EventRelationProcessor(config)
    
    # 处理Event
    result = processor.process_event(args.event, day)
    
    # 打印结果摘要
    print(f"\n{'='*60}")
    print("处理结果摘要")
    print(f"{'='*60}")
    print(f"Event ID: {result['event_id']}")
    print(f"总判断数: {len(result['all_judgments'])}")
    print(f"因果/活动关系数: {len(result['causal_activity_relations'])}")
    print(f"纯时序关系数: {len(result['temporal_relations'])}")
    print(f"总有效关系数: {len(result['all_valid_relations'])}")
    
    # 打印详细关系
    if result['all_judgments']:
        print(f"\n{'='*60}")
        print("所有LLM判断结果（包括no_relationship）")
        print(f"{'='*60}")
        for i, judgment in enumerate(result['all_judgments'], 1):
            print(f"\n判断 {i}:")
            print(f"  相关Event: {judgment['related_event_id']}")
            print(f"  关系类型: {judgment['type']}")
            print(f"  原因: {judgment['reason']}")
    
    if result['temporal_relations']:
        print(f"\n{'='*60}")
        print("时序关系（独立记录）")
        print(f"{'='*60}")
        for i, rel in enumerate(result['temporal_relations'], 1):
            print(f"\n时序关系 {i}:")
            print(f"  相关Event: {rel['related_event_id']}")
            print(f"  类型: {rel['type']}")
            print(f"  时间差: {rel['time_diff_seconds']:.1f}秒")
            print(f"  时间顺序: {rel['reason']}")
    
    # 验证Event文件已更新
    with open(event_file, 'r', encoding='utf-8') as f:
        updated_event = json.load(f)
    
    print(f"\n{'='*60}")
    print("Event文件更新验证")
    print(f"{'='*60}")
    print(f"relations字段长度: {len(updated_event.get('relations', []))}")
    
    if updated_event.get('relations'):
        print(f"\nrelations内容:")
        for i, rel in enumerate(updated_event['relations'], 1):
            print(f"\n  关系 {i}:")
            print(f"    类型: {rel['type']}")
            print(f"    相关Event: {rel['related_event_id']}")
            if 'time_diff_seconds' in rel:
                print(f"    时间差: {rel['time_diff_seconds']:.1f}秒")
    
    # 显示关系判断文件位置
    judgment_file = os.path.join(processor.relationships_dir, f"{args.event}.json")
    print(f"\n{'='*60}")
    print(f"完整判断结果已保存到:")
    print(f"{judgment_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
