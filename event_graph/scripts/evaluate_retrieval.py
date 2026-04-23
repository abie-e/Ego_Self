#!/usr/bin/env python3
import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from event_graph.neo4j_client import Neo4jClient
from event_graph.retriever import EventGraphRetriever

# Default config path
DEFAULT_CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "config.yaml")


def parse_timestamp(timestamp: str) -> Optional[int]:
    """
    Parse timestamp string to integer seconds
    Format: DAYx_HHMMSSFF -> total seconds from start

    Args:
        timestamp: e.g., "DAY1_11210217"

    Returns:
        Total seconds as integer, or None if invalid
    """
    try:
        parts = timestamp.split('_')
        if len(parts) != 2:
            return None

        day_str = parts[0]  # e.g., "DAY1"
        time_str = parts[1]  # e.g., "11210217"

        # Extract day number
        day_num = int(day_str.replace('DAY', ''))

        # Extract time components
        if len(time_str) < 6:
            return None

        hours = int(time_str[0:2])
        minutes = int(time_str[2:4])
        seconds = int(time_str[4:6])

        # Calculate total seconds
        # Assume each day starts at 0 seconds
        total_seconds = (day_num - 1) * 86400 + hours * 3600 + minutes * 60 + seconds

        return total_seconds

    except (ValueError, IndexError):
        return None


def check_time_match(event_timestamp: str, target_timestamp: str,
                     time_window: int = 600) -> bool:
    """
    Check if event_timestamp is within target_timestamp ± time_window

    Args:
        event_timestamp: Retrieved event timestamp
        target_timestamp: Target ground truth timestamp
        time_window: Time window in seconds (default: 600 = ±10 minutes)

    Returns:
        True if within window, False otherwise
    """
    event_time = parse_timestamp(event_timestamp)
    target_time = parse_timestamp(target_timestamp)

    if event_time is None or target_time is None:
        return False

    time_diff = abs(event_time - target_time)
    return time_diff <= time_window


def calculate_time_difference(event_timestamp: str, target_timestamp: str) -> Optional[int]:
    """
    Calculate time difference in seconds between two timestamps

    Returns:
        Absolute time difference in seconds, or None if invalid
    """
    event_time = parse_timestamp(event_timestamp)
    target_time = parse_timestamp(target_timestamp)

    if event_time is None or target_time is None:
        return None

    return abs(event_time - target_time)


def evaluate_retrieval(retriever: EventGraphRetriever,
                       qa_data: List[Dict[str, Any]],
                       time_window: int = 600,
                       top_k: int = 5,
                       expansion_hops: int = 2,
                       expansion_limit: int = 10) -> Dict[str, Any]:
    """
    Evaluate retrieval accuracy on QA dataset

    Args:
        retriever: EventGraphRetriever instance
        qa_data: List of QA dictionaries
        time_window: Time window in seconds (±600 = ±10 minutes)
        top_k: Number of top events to retrieve
        expansion_hops: Graph expansion hops
        expansion_limit: Graph expansion limit

    Returns:
        Evaluation results dictionary
    """
    total_questions = len(qa_data)
    results = []

    # Statistics
    recall_count = 0
    core_recall_count = 0
    expanded_recall_count = 0
    time_diffs = []

    # By question type
    type_stats = defaultdict(lambda: {'total': 0, 'hit': 0})

    # By need_name
    name_stats = {'True': {'total': 0, 'hit': 0}, 'False': {'total': 0, 'hit': 0}}

    # By need_audio
    audio_stats = {'True': {'total': 0, 'hit': 0}, 'False': {'total': 0, 'hit': 0}}

    pbar = tqdm(qa_data, desc="Evaluating", unit="q")
    for i, qa in enumerate(pbar, 1):
        question = qa['question']
        date = qa['query_time']['date']
        time = qa['query_time']['time']
        query_time = f"{date}_{time}"

        target_date = qa['target_time']['date']

        # Handle different target_time formats
        target_timestamps = []
        if 'time' in qa['target_time']:
            # Single or multiple times in 'time' field
            time_str = qa['target_time']['time']
            # Check if it contains multiple timestamps (e.g., "17044917DAY4_20593813DAY6_19483807")
            if 'DAY' in time_str:
                # Split by 'DAY' and reconstruct timestamps
                parts = time_str.split('DAY')
                for j in range(len(parts)):
                    if parts[j]:
                        # First part: use target_date
                        if j == 0:
                            target_timestamps.append(f"{target_date}_{parts[j]}")
                        else:
                            # Subsequent parts: extract day number and time
                            # Format: "4_20593813" -> "DAY4_20593813"
                            if '_' in parts[j]:
                                day_num, timestamp = parts[j].split('_', 1)
                                target_timestamps.append(f"DAY{day_num}_{timestamp}")
            else:
                target_timestamps.append(f"{target_date}_{time_str}")
        elif 'time_list' in qa['target_time']:
            # Multiple times in 'time_list' field
            for time_str in qa['target_time']['time_list']:
                target_timestamps.append(f"{target_date}_{time_str}")
        else:
            # No valid time field, skip this question
            tqdm.write(f"  Warning: No valid target_time for question {qa.get('ID', i)}")
            continue

        question_type = qa.get('type', 'Unknown')
        need_name = str(qa.get('need_name', False))
        need_audio = str(qa.get('need_audio', False))

        try:
            # Retrieve events
            retrieval_result = retriever.retrieve_and_summarize(
                question=question,
                query_time=query_time,
                top_k=top_k,
                expansion_hops=expansion_hops,
                expansion_limit=expansion_limit
            )

            # Check core events against all target timestamps
            core_events = retrieval_result.get('core_events', [])
            core_hit = False
            core_closest_diff = None

            for event in core_events:
                event_ts = event.get('timestamp', '')
                # Check against all target timestamps
                for target_ts in target_timestamps:
                    if check_time_match(event_ts, target_ts, time_window):
                        core_hit = True
                        diff = calculate_time_difference(event_ts, target_ts)
                        if diff is not None:
                            if core_closest_diff is None or diff < core_closest_diff:
                                core_closest_diff = diff

            # Check expanded events against all target timestamps
            expanded_events = retrieval_result.get('expanded_events', [])
            expanded_hit = False
            expanded_closest_diff = None

            for event in expanded_events:
                event_ts = event.get('timestamp', '')
                # Check against all target timestamps
                for target_ts in target_timestamps:
                    if check_time_match(event_ts, target_ts, time_window):
                        expanded_hit = True
                        diff = calculate_time_difference(event_ts, target_ts)
                        if diff is not None:
                            if expanded_closest_diff is None or diff < expanded_closest_diff:
                                expanded_closest_diff = diff

            # Overall hit
            hit = core_hit or expanded_hit
            closest_diff = None
            if core_closest_diff is not None and expanded_closest_diff is not None:
                closest_diff = min(core_closest_diff, expanded_closest_diff)
            elif core_closest_diff is not None:
                closest_diff = core_closest_diff
            elif expanded_closest_diff is not None:
                closest_diff = expanded_closest_diff

            # Update statistics
            if hit:
                recall_count += 1
                if closest_diff is not None:
                    time_diffs.append(closest_diff)

            if core_hit:
                core_recall_count += 1

            if expanded_hit:
                expanded_recall_count += 1

            # Update type stats
            type_stats[question_type]['total'] += 1
            if hit:
                type_stats[question_type]['hit'] += 1

            # Update name stats
            name_stats[need_name]['total'] += 1
            if hit:
                name_stats[need_name]['hit'] += 1

            # Update audio stats
            audio_stats[need_audio]['total'] += 1
            if hit:
                audio_stats[need_audio]['hit'] += 1

            # Store result
            result = {
                'id': qa.get('ID', i),
                'question': question,
                'query_time': query_time,
                'target_times': target_timestamps,  # Store all target timestamps
                'type': question_type,
                'need_name': need_name,
                'need_audio': need_audio,
                'hit': hit,
                'core_hit': core_hit,
                'expanded_hit': expanded_hit,
                'closest_time_diff': closest_diff,
                'core_events_count': len(core_events),
                'expanded_events_count': len(expanded_events),
                'retrieved_core_timestamps': [e.get('timestamp', '') for e in core_events[:5]],
                'retrieved_expanded_timestamps': [e.get('timestamp', '') for e in expanded_events[:5]]
            }
            results.append(result)

            pbar.set_postfix(recall=f"{recall_count}/{i} ({recall_count/i*100:.1f}%)")

        except Exception as e:
            tqdm.write(f"  Error on Q{qa.get('ID', i)}: {e}")

            result = {
                'id': qa.get('ID', i),
                'question': question,
                'query_time': query_time,
                'target_times': target_timestamps,
                'type': question_type,
                'need_name': need_name,
                'need_audio': need_audio,
                'error': str(e),
                'hit': False
            }
            results.append(result)

            # Update statistics for error cases
            type_stats[question_type]['total'] += 1
            name_stats[need_name]['total'] += 1
            audio_stats[need_audio]['total'] += 1

    # Calculate metrics
    recall = (recall_count / total_questions * 100) if total_questions > 0 else 0
    core_recall = (core_recall_count / total_questions * 100) if total_questions > 0 else 0
    expanded_recall = (expanded_recall_count / total_questions * 100) if total_questions > 0 else 0
    avg_time_diff = (sum(time_diffs) / len(time_diffs)) if time_diffs else None

    return {
        'total_questions': total_questions,
        'recall_count': recall_count,
        'recall': recall,
        'core_recall_count': core_recall_count,
        'core_recall': core_recall,
        'expanded_recall_count': expanded_recall_count,
        'expanded_recall': expanded_recall,
        'avg_time_diff': avg_time_diff,
        'time_window': time_window,
        'type_stats': dict(type_stats),
        'name_stats': name_stats,
        'audio_stats': audio_stats,
        'results': results
    }


def print_summary(eval_results: Dict[str, Any]):
    """Print a one-screen summary to stdout."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Questions:    {eval_results['total_questions']}")
    print(f"Time Window:        ±{eval_results['time_window']} s")
    print(f"Recall:             {eval_results['recall']:.2f}%  ({eval_results['recall_count']}/{eval_results['total_questions']})")
    if eval_results['avg_time_diff'] is not None:
        print(f"Avg Time Diff:      {eval_results['avg_time_diff']/60:.2f} min")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate event graph retrieval recall"
    )

    parser.add_argument(
        "--qa-file",
        type=str,
        default="event_graph/data/EgoLifeQA/EgoLifeQA_A1_JAKE.json",
        help="Path to the QA JSON file (e.g. EgoLifeQA)"
    )
    parser.add_argument(
        "--time-window",
        type=int,
        default=300,
        help="Time window in seconds (default: 300 = ±5 minutes)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top events to retrieve (default: 5)"
    )
    parser.add_argument(
        "--expansion-hops",
        type=int,
        default=2,
        help="Graph expansion hops (default: 2)"
    )
    parser.add_argument(
        "--expansion-limit",
        type=int,
        default=10,
        help="Graph expansion limit (default: 10)"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j URI (default: bolt://localhost:7687)"
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default="password",
        help="Neo4j password (default: password)"
    )

    args = parser.parse_args()

    # Load QA file
    with open(args.qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    print(f"Loaded {len(qa_data)} questions from {args.qa_file}")

    # Initialize Neo4j client + retriever
    client = Neo4jClient(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password
    )
    retriever = EventGraphRetriever(client, config_path=DEFAULT_CONFIG_PATH)

    # Run evaluation
    eval_results = evaluate_retrieval(
        retriever=retriever,
        qa_data=qa_data,
        time_window=args.time_window,
        top_k=args.top_k,
        expansion_hops=args.expansion_hops,
        expansion_limit=args.expansion_limit
    )

    print_summary(eval_results)
    client.close()


if __name__ == "__main__":
    main()
