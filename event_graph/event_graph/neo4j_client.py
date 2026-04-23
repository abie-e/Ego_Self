"""
Neo4j Client for Event Graph Management
Simple connection and CRUD operations for Event nodes
"""

from neo4j import GraphDatabase
from typing import Optional, List, Dict, Any
from datetime import datetime


class Neo4jClient:
    """Simple Neo4j client for event graph operations"""

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password"):
        """
        Initialize Neo4j connection

        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the connection"""
        self.driver.close()

    def create_event(self, clip_path: str, caption: str = "",
                     attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new Event node

        Args:
            clip_path: Path to the video clip
            caption: Event caption (optional)
            attributes: Additional attributes (optional)

        Returns:
            Event ID (UUID)
        """
        with self.driver.session() as session:
            # Convert attributes to string to avoid Neo4j Map type issue
            attrs_str = str(attributes) if attributes else ""
            result = session.run(
                """
                CREATE (e:Event {
                    clip_path: $clip_path,
                    caption: $caption,
                    attributes: $attributes,
                    timestamp: datetime()
                })
                RETURN elementId(e) as event_id
                """,
                clip_path=clip_path,
                caption=caption,
                attributes=attrs_str
            )
            record = result.single()
            return record["event_id"]

    def get_chain_tail_events(self) -> List[Dict[str, Any]]:
        """
        Get all events that are at the end of chains
        (events with no outgoing NEXT_EVENT relationships)

        Returns:
            List of tail event dictionaries
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Event)
                WHERE NOT (e)-[:NEXT_EVENT]->()
                RETURN elementId(e) as event_id, e.clip_path as clip_path,
                       e.caption as caption, e.timestamp as timestamp
                ORDER BY e.timestamp DESC
                """
            )
            return [dict(record) for record in result]

    def link_events(self, from_event_id: str, to_event_id: str,
                    similarity_score: Optional[float] = None):
        """
        Create NEXT_EVENT relationship between two events

        Args:
            from_event_id: Source event ID
            to_event_id: Target event ID
            similarity_score: Optional similarity score
        """
        with self.driver.session() as session:
            session.run(
                """
                MATCH (from:Event), (to:Event)
                WHERE elementId(from) = $from_id AND elementId(to) = $to_id
                CREATE (from)-[r:NEXT_EVENT]->(to)
                SET r.similarity_score = $score
                """,
                from_id=from_event_id,
                to_id=to_event_id,
                score=similarity_score
            )

    def get_all_chains(self) -> List[List[Dict[str, Any]]]:
        """
        Get all event chains (for visualization/debugging)

        Returns:
            List of chains, where each chain is a list of events
        """
        with self.driver.session() as session:
            # Find all chain heads (events with no incoming relationships)
            result = session.run(
                """
                MATCH (head:Event)
                WHERE NOT ()-[:NEXT_EVENT]->(head)
                MATCH path = (head)-[:NEXT_EVENT*0..]->(tail)
                WHERE NOT (tail)-[:NEXT_EVENT]->()
                RETURN [node in nodes(path) | {
                    id: elementId(node),
                    clip_path: node.clip_path,
                    caption: node.caption,
                    timestamp: node.timestamp
                }] as chain
                ORDER BY head.timestamp
                """
            )
            return [record["chain"] for record in result]

    def clear_all_events(self):
        """Clear all events and relationships (for testing)"""
        with self.driver.session() as session:
            session.run("MATCH (e:Event) DETACH DELETE e")

    def create_event_with_id(self, event_id: str, video_path: str,
                            caption: str = "", timestamp: str = "",
                            meta_id: int = 0, embedding: Optional[List[float]] = None) -> str:
        """
        Create a new Event node with specific event_id from JSON

        Args:
            event_id: Event ID from JSON (used for internal mapping only)
            video_path: Path to the video clip
            caption: Event caption
            timestamp: Event timestamp
            meta_id: Sequential ID (1, 2, 3...)
            embedding: OpenCLIP embedding vector

        Returns:
            Neo4j internal element ID
        """
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (e:Event {
                    meta_id: $meta_id,
                    timestamp: $timestamp,
                    video_path: $video_path,
                    caption: $caption,
                    embedding: $embedding
                })
                RETURN elementId(e) as element_id
                """,
                meta_id=meta_id,
                timestamp=timestamp,
                video_path=video_path,
                caption=caption,
                embedding=embedding
            )
            record = result.single()
            return record["element_id"]

    def create_object(self, name: str, description: str = "",
                     properties: Optional[Dict[str, Any]] = None,
                     embedding: Optional[List[float]] = None) -> str:
        """
        Create a new Object node (always creates a new unique node)

        Args:
            name: Object name/identifier
            description: Object description
            properties: Additional properties
            embedding: OpenCLIP embedding vector

        Returns:
            Neo4j internal element ID
        """
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (o:Object {
                    name: $name,
                    description: $description,
                    properties: $properties,
                    embedding: $embedding
                })
                RETURN elementId(o) as element_id
                """,
                name=name,
                description=description,
                properties=str(properties) if properties else "",
                embedding=embedding
            )
            record = result.single()
            return record["element_id"]

    def create_person(self, name: str, description: str = "",
                     properties: Optional[Dict[str, Any]] = None,
                     embedding: Optional[List[float]] = None) -> str:
        """
        Create a new Person node (always creates a new unique node)

        Args:
            name: Person name/identifier
            description: Person description
            properties: Additional properties
            embedding: OpenCLIP embedding vector

        Returns:
            Neo4j internal element ID
        """
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (p:Person {
                    name: $name,
                    description: $description,
                    properties: $properties,
                    embedding: $embedding
                })
                RETURN elementId(p) as element_id
                """,
                name=name,
                description=description,
                properties=str(properties) if properties else "",
                embedding=embedding
            )
            record = result.single()
            return record["element_id"]

    def create_voice(self, voice_id: str, name: str = "",
                    register_event: str = "", register_time: str = "",
                    feature_path: str = "",
                    properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new Voice node

        Args:
            voice_id: Voice ID (e.g., "voice_1")
            name: Voice name (e.g., "person_1")
            register_event: First event where this voice was registered
            register_time: Registration timestamp
            feature_path: Path to voice features
            properties: Additional properties

        Returns:
            Neo4j internal element ID
        """
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (v:Voice {
                    voice_id: $voice_id,
                    name: $name,
                    register_event: $register_event,
                    register_time: $register_time,
                    feature_path: $feature_path,
                    properties: $properties
                })
                RETURN elementId(v) as element_id
                """,
                voice_id=voice_id,
                name=name,
                register_event=register_event,
                register_time=register_time,
                feature_path=feature_path,
                properties=str(properties) if properties else ""
            )
            record = result.single()
            return record["element_id"]

    def link_event_to_entity(self, event_id: str, entity_id: str,
                            relationship_type: str = "INVOLVES",
                            properties: Optional[Dict[str, Any]] = None):
        """
        Create relationship between Event and Entity (Object/Person)

        Args:
            event_id: Event element ID
            entity_id: Entity element ID
            relationship_type: Type of relationship (INVOLVES, INTERACTS_WITH, etc.)
            properties: Relationship properties (action, first_appearance_time, etc.)
        """
        with self.driver.session() as session:
            # Create relationship and set properties individually
            session.run(
                f"""
                MATCH (e:Event), (entity)
                WHERE elementId(e) = $event_id AND elementId(entity) = $entity_id
                CREATE (e)-[r:{relationship_type}]->(entity)
                SET r += $properties
                """,
                event_id=event_id,
                entity_id=entity_id,
                properties=properties if properties else {}
            )

    def link_events_by_event_id(self, from_event_id: str, to_event_id: str,
                               relationship_type: str = "NEXT_EVENT",
                               properties: Optional[Dict[str, Any]] = None):
        """
        Create relationship between two events using their event_id
        Converts event_id to timestamp for matching (removes _evt suffix)

        Args:
            from_event_id: Source event's event_id (e.g., "DAY1_11094208_evt")
            to_event_id: Target event's event_id (e.g., "DAY1_11100000_evt")
            relationship_type: Type of relationship (causal, temporal_after, etc.)
            properties: Relationship properties (reason, time_diff_seconds, etc.)
        """
        # Convert event_id to timestamp (remove _evt suffix)
        from_timestamp = from_event_id.replace('_evt', '')
        to_timestamp = to_event_id.replace('_evt', '')

        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (from:Event {{timestamp: $from_ts}}), (to:Event {{timestamp: $to_ts}})
                CREATE (from)-[r:{relationship_type}]->(to)
                SET r += $properties
                """,
                from_ts=from_timestamp,
                to_ts=to_timestamp,
                properties=properties if properties else {}
            )

    def clear_all(self):
        """Clear all nodes and relationships (for testing)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def search_similar_nodes(self, query_embedding: List[float],
                            node_type: Optional[str] = None,
                            top_k: int = 10,
                            min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar nodes using cosine similarity

        Args:
            query_embedding: Query embedding vector
            node_type: Filter by node type (Event, Person, Object). None for all types.
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of similar nodes with similarity scores
        """
        with self.driver.session() as session:
            # Build type filter
            type_filter = f":{node_type}" if node_type else ""

            result = session.run(
                f"""
                MATCH (n{type_filter})
                WHERE n.embedding IS NOT NULL
                WITH n,
                     gds.similarity.cosine(n.embedding, $query_emb) AS similarity
                WHERE similarity >= $min_sim
                RETURN elementId(n) as id,
                       labels(n) as labels,
                       properties(n) as properties,
                       similarity
                ORDER BY similarity DESC
                LIMIT $top_k
                """,
                query_emb=query_embedding,
                min_sim=min_similarity,
                top_k=top_k
            )
            return [dict(record) for record in result]

    def search_similar_events(self, query_text: str,
                             query_embedding: List[float],
                             top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar events using embedding similarity

        Args:
            query_text: Query text (for reference)
            query_embedding: Query embedding vector
            top_k: Number of top results to return

        Returns:
            List of similar events with details
        """
        results = self.search_similar_nodes(
            query_embedding=query_embedding,
            node_type="Event",
            top_k=top_k
        )

        # Format results
        formatted_results = []
        for result in results:
            props = result['properties']
            formatted_results.append({
                'id': result['id'],
                'timestamp': props.get('timestamp', ''),
                'caption': props.get('caption', ''),
                'video_path': props.get('video_path', ''),
                'similarity': result['similarity']
            })

        return formatted_results

    def find_events_involving_entities(self, entity_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Find all events that involve the given entities (reverse INVOLVES lookup)

        Args:
            entity_ids: List of entity element IDs (Object or Person)

        Returns:
            List of event dictionaries
        """
        if not entity_ids:
            return []

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Event)-[:INVOLVES]->(entity)
                WHERE elementId(entity) IN $entity_ids
                RETURN DISTINCT
                    elementId(e) as id,
                    e.meta_id as meta_id,
                    e.timestamp as timestamp,
                    e.caption as caption,
                    e.video_path as video_path,
                    e.embedding as embedding
                ORDER BY e.timestamp DESC
                """,
                entity_ids=entity_ids
            )
            return [dict(record) for record in result]

    def expand_same_activity(self, event_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Expand events through SAME_ACTIVITY relationships

        Args:
            event_ids: List of event element IDs

        Returns:
            List of related event dictionaries
        """
        if not event_ids:
            return []

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Event)-[:SAME_ACTIVITY]-(related:Event)
                WHERE elementId(e) IN $event_ids
                RETURN DISTINCT
                    elementId(related) as id,
                    related.meta_id as meta_id,
                    related.timestamp as timestamp,
                    related.caption as caption,
                    related.video_path as video_path,
                    related.embedding as embedding
                ORDER BY related.timestamp DESC
                """,
                event_ids=event_ids
            )
            return [dict(record) for record in result]

    def get_event_entities(self, event_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all entities (Objects and Persons) involved in the given events

        Args:
            event_ids: List of event element IDs

        Returns:
            Dictionary with 'objects' and 'persons' lists
        """
        if not event_ids:
            return {'objects': [], 'persons': []}

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Event)-[:INVOLVES]->(entity)
                WHERE elementId(e) IN $event_ids
                RETURN DISTINCT
                    elementId(entity) as id,
                    labels(entity)[0] as type,
                    entity.name as name,
                    entity.description as description,
                    entity.embedding as embedding
                """,
                event_ids=event_ids
            )

            entities = {'objects': [], 'persons': []}
            for record in result:
                entity_dict = dict(record)
                if entity_dict['type'] == 'Object':
                    entities['objects'].append(entity_dict)
                elif entity_dict['type'] == 'Person':
                    entities['persons'].append(entity_dict)

            return entities

    # ==================== Cluster Management Methods ====================

    def create_cluster(self, cluster_id: int, label: str, summary: str,
                      embedding: List[float], event_count: int,
                      time_span: str) -> str:
        """
        Create a Cluster node

        Args:
            cluster_id: HDBSCAN cluster ID
            label: LLM-generated label
            summary: LLM-generated summary
            embedding: Cluster embedding (3072-dim)
            event_count: Number of events in cluster
            time_span: Time range of events

        Returns:
            Cluster element ID
        """
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (c:Cluster {
                    cluster_id: $cluster_id,
                    label: $label,
                    summary: $summary,
                    embedding: $embedding,
                    event_count: $event_count,
                    time_span: $time_span
                })
                RETURN elementId(c) as element_id
                """,
                cluster_id=cluster_id,
                label=label,
                summary=summary,
                embedding=embedding,
                event_count=event_count,
                time_span=time_span
            )
            return result.single()["element_id"]

    def link_event_to_cluster(self, event_id: str, cluster_id: int):
        """
        Create BELONGS_TO relationship from Event to Cluster

        Args:
            event_id: Event element ID
            cluster_id: Cluster ID
        """
        with self.driver.session() as session:
            session.run(
                """
                MATCH (e:Event), (c:Cluster)
                WHERE elementId(e) = $event_id AND c.cluster_id = $cluster_id
                MERGE (e)-[:BELONGS_TO {cluster_id: $cluster_id}]->(c)
                MERGE (c)-[:CONTAINS]->(e)
                """,
                event_id=event_id,
                cluster_id=cluster_id
            )

    def get_all_events_with_embeddings(self, participant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all events with their embeddings

        Args:
            participant_id: Optional participant filter

        Returns:
            List of event dictionaries
        """
        with self.driver.session() as session:
            query = """
                MATCH (e:Event)
                WHERE e.embedding IS NOT NULL
            """
            if participant_id:
                query += " AND e.participant_id = $participant_id"

            query += """
                RETURN
                    elementId(e) as id,
                    e.meta_id as meta_id,
                    e.timestamp as timestamp,
                    e.caption as caption,
                    e.video_path as video_path,
                    e.embedding as embedding
                ORDER BY e.timestamp
            """

            result = session.run(query, participant_id=participant_id)
            return [dict(record) for record in result]

    def get_cluster_by_id(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a cluster by its cluster_id

        Args:
            cluster_id: Cluster ID

        Returns:
            Cluster dictionary or None
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Cluster {cluster_id: $cluster_id})
                RETURN
                    elementId(c) as id,
                    c.cluster_id as cluster_id,
                    c.label as label,
                    c.summary as summary,
                    c.embedding as embedding,
                    c.event_count as event_count,
                    c.time_span as time_span
                """,
                cluster_id=cluster_id
            )
            record = result.single()
            return dict(record) if record else None

    def get_all_clusters(self) -> List[Dict[str, Any]]:
        """
        Get all clusters

        Returns:
            List of cluster dictionaries
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Cluster)
                RETURN
                    elementId(c) as id,
                    c.cluster_id as cluster_id,
                    c.label as label,
                    c.summary as summary,
                    c.embedding as embedding,
                    c.event_count as event_count,
                    c.time_span as time_span
                ORDER BY c.cluster_id
                """
            )
            return [dict(record) for record in result]

    def get_events_from_cluster(self, cluster_id: int, query_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all events from a cluster

        Args:
            cluster_id: Cluster ID
            query_time: Optional time filter (only events before this time)

        Returns:
            List of event dictionaries
        """
        with self.driver.session() as session:
            query = """
                MATCH (c:Cluster {cluster_id: $cluster_id})-[:CONTAINS]->(e:Event)
            """
            if query_time:
                query += " WHERE e.timestamp <= $query_time"

            query += """
                RETURN
                    elementId(e) as id,
                    e.meta_id as meta_id,
                    e.timestamp as timestamp,
                    e.caption as caption,
                    e.video_path as video_path,
                    e.embedding as embedding
                ORDER BY e.timestamp DESC
            """

            result = session.run(query, cluster_id=cluster_id, query_time=query_time)
            return [dict(record) for record in result]

    def clear_clusters(self):
        """
        Delete all Cluster nodes and their relationships
        """
        with self.driver.session() as session:
            session.run("MATCH (c:Cluster) DETACH DELETE c")
            # Also remove BELONGS_TO relationships from Events
            session.run("""
                MATCH (e:Event)-[r:BELONGS_TO]->()
                DELETE r
            """)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
