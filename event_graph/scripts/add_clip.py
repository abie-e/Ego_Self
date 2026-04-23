#!/usr/bin/env python3
"""
Add Clip Script
Command-line interface to add video clips to the event graph
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path to import event_graph module
sys.path.insert(0, str(Path(__file__).parent.parent))

from event_graph.neo4j_client import Neo4jClient
from event_graph.online_builder import OnlineEventBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Add a video clip to the event graph"
    )
    parser.add_argument(
        "clip_path",
        type=str,
        help="Path to the video clip"
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="",
        help="Event caption (optional)"
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
        "--summary",
        action="store_true",
        help="Print graph summary after adding event"
    )

    args = parser.parse_args()

    # Validate clip path
    if not os.path.exists(args.clip_path):
        print(f"Error: Clip path does not exist: {args.clip_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Initialize Neo4j client
        print(f"Connecting to Neo4j at {args.neo4j_uri}...")
        client = Neo4jClient(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password
        )

        # Initialize online builder
        builder = OnlineEventBuilder(client)

        # Add the event
        event_id = builder.add_event(
            clip_path=args.clip_path,
            caption=args.caption
        )

        print(f"\n✓ Successfully added event: {event_id}")

        # Print summary if requested
        if args.summary:
            print("\n" + "="*50)
            print("Graph Summary")
            print("="*50)
            summary = builder.get_graph_summary()
            print(f"Total chains: {summary['total_chains']}")
            print(f"Total events: {summary['total_events']}")
            print("\nChains:")
            for i, chain in enumerate(summary['chains'], 1):
                print(f"  Chain {i}: {chain['length']} event(s)")
                print(f"    Head: {chain['head']}")
                print(f"    Tail: {chain['tail']}")

        client.close()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
