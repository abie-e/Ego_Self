"""
Event Clusterer - Pure Semantic Clustering
Uses HDBSCAN for automatic cluster discovery based on event embeddings
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
import hdbscan
from sklearn.metrics.pairwise import cosine_distances

from .neo4j_client import Neo4jClient
from .embedding_extractor import EmbeddingExtractor


class EventClusterer:
    """Pure semantic clustering of events using HDBSCAN"""

    def __init__(self, neo4j_client: Neo4jClient,
                 embedding_extractor: EmbeddingExtractor,
                 llm_client,
                 llm_model: str):
        """
        Initialize clusterer

        Args:
            neo4j_client: Neo4jClient instance
            embedding_extractor: EmbeddingExtractor instance
            llm_client: OpenAI client for summarization
            llm_model: Model name for LLM
        """
        self.client = neo4j_client
        self.embedding_extractor = embedding_extractor
        self.llm_client = llm_client
        self.llm_model = llm_model

    def fetch_all_events(self, participant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch all events with embeddings from Neo4j

        Args:
            participant_id: Optional participant filter

        Returns:
            List of event dictionaries
        """
        events = self.client.get_all_events_with_embeddings(participant_id)
        print(f"Fetched {len(events)} events from Neo4j")
        return events

    def cluster_events_semantic(self, events: List[Dict[str, Any]],
                               min_cluster_size: int = 5,
                               min_samples: Optional[int] = None) -> Dict[int, List[Dict[str, Any]]]:
        """
        Cluster events using HDBSCAN based on semantic similarity

        Args:
            events: List of event dictionaries with embeddings
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples (defaults to min_cluster_size - 1)

        Returns:
            Dictionary mapping cluster_id to list of events
        """
        if len(events) < min_cluster_size:
            print(f"Warning: Only {len(events)} events, less than min_cluster_size={min_cluster_size}")
            return {}

        # Extract embeddings
        embeddings = np.array([e['embedding'] for e in events])
        print(f"Clustering {len(embeddings)} events with embeddings of shape {embeddings.shape}")

        # Set min_samples
        if min_samples is None:
            min_samples = max(2, min_cluster_size - 1)

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='cosine',
            cluster_selection_method='eom'
        )

        print(f"Running HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        labels = clusterer.fit_predict(embeddings)

        # Group events by cluster
        clusters = defaultdict(list)
        for event, label in zip(events, labels):
            if label != -1:  # Skip noise
                clusters[label].append(event)

        print(f"Found {len(clusters)} clusters")
        print(f"Noise points: {sum(1 for l in labels if l == -1)}")

        return dict(clusters)

    def summarize_cluster(self, cluster_events: List[Dict[str, Any]]) -> str:
        """
        Generate LLM summary for a cluster

        Args:
            cluster_events: List of events in the cluster

        Returns:
            Summary string
        """
        # Format events for prompt
        event_descriptions = []
        for event in cluster_events[:10]:  # Limit to first 10 for brevity
            timestamp = event.get('timestamp', 'unknown')
            caption = event.get('caption', 'No caption')
            event_descriptions.append(f"- [{timestamp}] {caption}")

        if len(cluster_events) > 10:
            event_descriptions.append(f"... and {len(cluster_events) - 10} more similar events")

        prompt = f"""总结以下语义相似的活动:

Events (跨多天):
{chr(10).join(event_descriptions)}

用1-2句话总结这组活动的共同主题。保持简洁。"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            print(f"Warning: LLM summarization failed: {e}")
            # Fallback: use first event caption
            return cluster_events[0].get('caption', 'Activity cluster')

    def generate_cluster_label(self, summary: str) -> str:
        """
        Generate short label from summary

        Args:
            summary: Cluster summary

        Returns:
            Short label (3-5 words)
        """
        # Simple heuristic: take first 50 characters or first sentence
        label = summary.split('。')[0]  # Take first sentence
        if len(label) > 50:
            label = label[:47] + '...'
        return label

    def compute_cluster_embedding(self, cluster_events: List[Dict[str, Any]],
                                 summary: str,
                                 method: str = 'summary') -> List[float]:
        """
        Compute cluster embedding

        Args:
            cluster_events: List of events in cluster
            summary: Cluster summary
            method: 'summary', 'mean', or 'medoid'

        Returns:
            Cluster embedding
        """
        if method == 'summary':
            # Generate embedding from summary (recommended)
            embedding = self.embedding_extractor.extract_text_embedding(summary)
            return embedding

        elif method == 'mean':
            # Mean of all event embeddings
            embeddings = np.array([e['embedding'] for e in cluster_events])
            return embeddings.mean(axis=0).tolist()

        elif method == 'medoid':
            # Use the most central event's embedding
            embeddings = np.array([e['embedding'] for e in cluster_events])
            centroid = embeddings.mean(axis=0)
            distances = cosine_distances(embeddings, centroid.reshape(1, -1)).flatten()
            medoid_idx = distances.argmin()
            return embeddings[medoid_idx].tolist()

        else:
            raise ValueError(f"Unknown method: {method}")

    def build_and_store_clusters(self, participant_id: Optional[str] = None,
                                 min_cluster_size: int = 5,
                                 min_samples: Optional[int] = None,
                                 generate_labels: bool = True,
                                 embedding_method: str = 'summary') -> Dict[str, Any]:
        """
        Complete pipeline: fetch events → cluster → summarize → store

        Args:
            participant_id: Optional participant filter
            min_cluster_size: HDBSCAN min_cluster_size
            min_samples: HDBSCAN min_samples
            generate_labels: Whether to generate LLM labels
            embedding_method: Method for cluster embedding

        Returns:
            Statistics dictionary
        """
        print("=" * 60)
        print("Building Event Clusters (Pure Semantic)")
        print("=" * 60)

        # Step 1: Fetch events
        print("\n[1/5] Fetching events from Neo4j...")
        events = self.fetch_all_events(participant_id)

        if len(events) == 0:
            print("No events found!")
            return {'total_events': 0, 'n_clusters': 0, 'n_noise': 0}

        # Step 2: Cluster events
        print("\n[2/5] Clustering events with HDBSCAN...")
        clusters = self.cluster_events_semantic(events, min_cluster_size, min_samples)

        if len(clusters) == 0:
            print("No clusters formed! Try lowering min_cluster_size.")
            return {'total_events': len(events), 'n_clusters': 0, 'n_noise': len(events)}

        # Step 3: Generate cluster metadata
        print("\n[3/5] Generating cluster summaries and embeddings...")
        cluster_metadata = []

        for cluster_id, cluster_events in clusters.items():
            print(f"  Cluster {cluster_id}: {len(cluster_events)} events... ", end='')

            # Generate summary
            summary = self.summarize_cluster(cluster_events)

            # Generate label
            if generate_labels:
                label = self.generate_cluster_label(summary)
            else:
                label = f"Cluster {cluster_id}"

            # Compute embedding
            embedding = self.compute_cluster_embedding(cluster_events, summary, method=embedding_method)

            # Get time span
            timestamps = [e['timestamp'] for e in cluster_events if e.get('timestamp')]
            time_span = f"{min(timestamps)} - {max(timestamps)}" if timestamps else "unknown"

            cluster_metadata.append({
                'cluster_id': cluster_id,
                'label': label,
                'summary': summary,
                'embedding': embedding,
                'event_count': len(cluster_events),
                'events': cluster_events,
                'time_span': time_span
            })

            print(f"✓ {label}")

        # Step 4: Store to Neo4j
        print("\n[4/5] Storing clusters to Neo4j...")
        for cluster_meta in cluster_metadata:
            # Create cluster node
            self.client.create_cluster(
                cluster_id=cluster_meta['cluster_id'],
                label=cluster_meta['label'],
                summary=cluster_meta['summary'],
                embedding=cluster_meta['embedding'],
                event_count=cluster_meta['event_count'],
                time_span=cluster_meta['time_span']
            )

            # Link events to cluster
            for event in cluster_meta['events']:
                self.client.link_event_to_cluster(event['id'], cluster_meta['cluster_id'])

        print(f"✓ Stored {len(cluster_metadata)} clusters to Neo4j")

        # Step 5: Compute statistics
        print("\n[5/5] Computing statistics...")
        labels_all = [-1] * len(events)  # Default to noise
        for cluster_meta in cluster_metadata:
            for event in cluster_meta['events']:
                idx = next((i for i, e in enumerate(events) if e['id'] == event['id']), None)
                if idx is not None:
                    labels_all[idx] = cluster_meta['cluster_id']

        n_noise = sum(1 for l in labels_all if l == -1)

        stats = {
            'total_events': len(events),
            'n_clusters': len(clusters),
            'n_noise': n_noise,
            'cluster_sizes': {meta['cluster_id']: meta['event_count'] for meta in cluster_metadata},
            'cluster_labels': {meta['cluster_id']: meta['label'] for meta in cluster_metadata},
            'cluster_metadata': cluster_metadata
        }

        print("\n" + "=" * 60)
        print("Clustering Complete!")
        print("=" * 60)

        return stats
