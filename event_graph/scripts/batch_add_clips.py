#!/usr/bin/env python3
"""
Batch Add Clips Script
Process all video clips in a directory and add them to the event graph
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List

# Add parent directory to path to import event_graph module
sys.path.insert(0, str(Path(__file__).parent.parent))

from event_graph.neo4j_client import Neo4jClient
from event_graph.online_builder import OnlineEventBuilder


def find_video_clips(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Find all video clips in a directory

    Args:
        directory: Directory to search
        extensions: List of video file extensions (default: mp4, avi, mov, mkv)

    Returns:
        Sorted list of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']

    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                video_files.append(os.path.join(root, file))

    return sorted(video_files)


def main():
    parser = argparse.ArgumentParser(
        description="Batch add video clips from a directory to the event graph"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing video clips"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)"
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
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing events before adding new ones"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=None,
        help="Video file extensions to process (default: mp4 avi mov mkv)"
    )

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: Directory does not exist: {args.directory}", file=sys.stderr)
        sys.exit(1)

    try:
        # Initialize Neo4j client
        print(f"Connecting to Neo4j at {args.neo4j_uri}...")
        client = Neo4jClient(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password
        )

        # Clear existing events if requested
        if args.clear:
            print("Clearing existing events...")
            client.clear_all_events()
            print("✓ Cleared all existing events")

        # Find video clips
        print(f"\nScanning directory: {args.directory}")
        video_files = find_video_clips(args.directory, args.extensions)
        print(f"Found {len(video_files)} video clip(s)")

        if len(video_files) == 0:
            print("No video files found. Exiting.")
            client.close()
            sys.exit(0)

        # Initialize online builder
        builder = OnlineEventBuilder(client)

        # Process each video clip
        print("\nProcessing clips...")
        print("="*60)
        for i, clip_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] Processing: {os.path.basename(clip_path)}")
            try:
                event_id = builder.add_event(clip_path=clip_path)
                print(f"  ✓ Added event: {event_id}")
            except Exception as e:
                print(f"  ✗ Error: {e}", file=sys.stderr)

        # Print final summary
        print("\n" + "="*60)
        print("Processing Complete - Graph Summary")
        print("="*60)
        summary = builder.get_graph_summary()
        print(f"Total chains: {summary['total_chains']}")
        print(f"Total events: {summary['total_events']}")
        print("\nChains:")
        for i, chain in enumerate(summary['chains'], 1):
            print(f"  Chain {i}: {chain['length']} event(s)")
            print(f"    Head: {os.path.basename(chain['head'])}")
            print(f"    Tail: {os.path.basename(chain['tail'])}")

        client.close()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
