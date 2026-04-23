#!/usr/bin/env python3
"""
Build Event Clusters - Offline Preprocessing
Pure semantic clustering using HDBSCAN
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from openai import OpenAI

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from event_graph.neo4j_client import Neo4jClient
from event_graph.clusterer import EventClusterer
from event_graph.embedding_extractor import get_extractor

DEFAULT_CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "config.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="Build semantic clusters from events in Neo4j"
    )

    # Neo4j connection
    parser.add_argument("--neo4j-uri", type=str,
                       default="bolt://localhost:7687",
                       help="Neo4j URI")
    parser.add_argument("--neo4j-user", type=str, default="neo4j",
                       help="Neo4j username")
    parser.add_argument("--neo4j-password", type=str, default="password",
                       help="Neo4j password")

    # Clustering parameters
    parser.add_argument("--min-cluster-size", type=int, default=5,
                       help="HDBSCAN min_cluster_size (default: 5)")
    parser.add_argument("--min-samples", type=int, default=None,
                       help="HDBSCAN min_samples (default: min_cluster_size-1)")

    # Options
    parser.add_argument("--participant-id", type=str, default=None,
                       help="Filter by participant ID (e.g., JAKE)")
    parser.add_argument("--generate-labels", action="store_true",
                       help="Generate LLM labels for clusters")
    parser.add_argument("--embedding-method", type=str, default="summary",
                       choices=["summary", "mean", "medoid"],
                       help="Method to compute cluster embedding")
    parser.add_argument("--clear-existing", action="store_true",
                       help="Clear existing clusters before building")

    # Output
    parser.add_argument("--output-stats", type=str, default=None,
                       help="Save clustering statistics to JSON file")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                       help="Path to config.yaml")

    args = parser.parse_args()

    try:
        # Load config
        print(f"Loading config from {args.config}...")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize Neo4j client
        print(f"\nConnecting to Neo4j at {args.neo4j_uri}...")
        neo4j_client = Neo4jClient(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password
        )
        print("✓ Connected to Neo4j")

        # Clear existing clusters if requested
        if args.clear_existing:
            print("\nClearing existing clusters...")
            neo4j_client.clear_clusters()
            print("✓ Cleared existing clusters")

        # Initialize embedding extractor
        print("\nInitializing embedding extractor...")
        embedding_extractor = get_extractor(args.config)
        print("✓ Embedding extractor ready")

        # Initialize LLM client
        print("\nInitializing LLM client...")
        llm_config = config['api']['llm']
        llm_client = OpenAI(
            base_url=llm_config['base_url'],
            api_key=llm_config['api_key']
        )
        llm_model = llm_config['model']
        print(f"✓ LLM client ready ({llm_model})")

        # Initialize clusterer
        clusterer = EventClusterer(
            neo4j_client=neo4j_client,
            embedding_extractor=embedding_extractor,
            llm_client=llm_client,
            llm_model=llm_model
        )

        # Build and store clusters
        print("\n" + "=" * 60)
        print("Starting cluster building process...")
        print("=" * 60)

        stats = clusterer.build_and_store_clusters(
            participant_id=args.participant_id,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            generate_labels=args.generate_labels,
            embedding_method=args.embedding_method
        )

        # Print statistics
        print("\n" + "=" * 60)
        print("CLUSTERING STATISTICS")
        print("=" * 60)
        print(f"Total Events: {stats['total_events']}")
        print(f"Clusters Found: {stats['n_clusters']}")
        print(f"Noise Points: {stats['n_noise']} ({stats['n_noise']/max(stats['total_events'],1)*100:.1f}%)")
        print(f"\nCluster Distribution:")

        # Sort clusters by size (descending)
        sorted_clusters = sorted(
            stats['cluster_sizes'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for cluster_id, size in sorted_clusters:
            label = stats['cluster_labels'].get(cluster_id, f"Cluster {cluster_id}")
            print(f"  [{cluster_id:2d}] {label:40s}: {size:3d} events")

        print("=" * 60)

        # Save statistics if requested
        if args.output_stats:
            # Remove 'events' from cluster_metadata to reduce size
            stats_to_save = stats.copy()
            if 'cluster_metadata' in stats_to_save:
                for cluster in stats_to_save['cluster_metadata']:
                    if 'events' in cluster:
                        cluster['events'] = len(cluster['events'])  # Just count
                    if 'embedding' in cluster:
                        del cluster['embedding']  # Remove large embeddings

            with open(args.output_stats, 'w') as f:
                json.dump(stats_to_save, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Statistics saved to {args.output_stats}")

        neo4j_client.close()
        print("\n✓ Cluster building complete!")

    except FileNotFoundError as e:
        print(f"\nError: Config file not found: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
