"""
Online Event Builder
Handles adding new events to the graph in real-time
"""

from typing import Optional, Dict, Any, Callable
from .neo4j_client import Neo4jClient


class OnlineEventBuilder:
    """Builds event graph incrementally as new clips arrive"""

    def __init__(self, neo4j_client: Neo4jClient,
                 similarity_checker: Optional[Callable] = None):
        """
        Initialize the online builder

        Args:
            neo4j_client: Neo4j client instance
            similarity_checker: Function to check if two events are similar
                                Signature: f(new_event_data, tail_event_data) -> bool
                                If None, uses a placeholder that always returns False
        """
        self.client = neo4j_client
        self.similarity_checker = similarity_checker or self._default_similarity_checker

    def _default_similarity_checker(self, new_event: Dict[str, Any],
                                    tail_event: Dict[str, Any]) -> bool:
        """
        Default similarity checker (modified)
        Always returns True, so each event connects to the previous one

        Args:
            new_event: New event data
            tail_event: Existing tail event data

        Returns:
            True or False
        """
        print(f"[Similarity Check] Comparing new event with tail {tail_event['clip_path']}")
        print(f"  Result: Similar (auto-connect mode)")

        # query = """
        # CALL db.index.vector.queryNodes('doc_embedding_index', 3, $embedding)
        # YIELD node, score
        # RETURN node.title AS title, score
        # ORDER BY score DESC
        # """

        # embedding = new_event["embedding"]
        # score = 0.0

        # with self.driver.session() as session:
        #     result = session.run(query, {"embedding": embedding})
        #     record = result.single()
        #     if record:
        #         score = record["score"]

        # print(f"  → Similarity Score: {score:.3f}")
        # is_similar = score > 0.5
        
        return True

    def add_event(self, clip_path: str, caption: str = "",
                  attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new event to the graph

        Process:
        1. Create new Event node
        2. Get all chain tail events
        3. Check similarity with each tail
        4. If similar to any tail -> link to that tail
        5. If not similar to any -> start new chain (no link)

        Args:
            clip_path: Path to video clip
            caption: Event caption (optional)
            attributes: Additional attributes (optional)

        Returns:
            Event ID of the newly created event
        """
        # Step 1: Create new event node
        print(f"\n[Add Event] Creating event for: {clip_path}")
        new_event_id = self.client.create_event(
            clip_path=clip_path,
            caption=caption,
            attributes=attributes
        )
        print(f"  Created event with ID: {new_event_id}")

        # Step 2: Get all chain tail events
        tail_events = self.client.get_chain_tail_events()
        print(f"  Found {len(tail_events)} chain tail(s)")

        # If this is the first event, no linking needed
        if len(tail_events) <= 1:  # Only the newly created event itself
            print(f"  First event in graph - starting new chain")
            return new_event_id

        # Step 3 & 4: Check similarity and link if matched
        new_event_data = {
            "clip_path": clip_path,
            "caption": caption,
            "attributes": attributes
        }

        for tail in tail_events:
            # Skip the newly created event itself
            if tail["event_id"] == new_event_id:
                continue

            # Check similarity
            if self.similarity_checker(new_event_data, tail):
                print(f"  Matched with tail: {tail['clip_path']}")
                print(f"  Linking events...")
                self.client.link_events(
                    from_event_id=tail["event_id"],
                    to_event_id=new_event_id
                )
                print(f"  Successfully linked to existing chain")
                return new_event_id

        # Step 5: No match found - this starts a new chain
        print(f"  No match found - starting new chain")
        return new_event_id

    def get_graph_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current graph state

        Returns:
            Dictionary with graph statistics
        """
        chains = self.client.get_all_chains()
        return {
            "total_chains": len(chains),
            "total_events": sum(len(chain) for chain in chains),
            "chains": [
                {
                    "length": len(chain),
                    "head": chain[0]["clip_path"] if chain else None,
                    "tail": chain[-1]["clip_path"] if chain else None
                }
                for chain in chains
            ]
        }
