"""
Event Graph Retriever
Retrieve and summarize events from Neo4j graph based on questions
"""

import os
import json
import yaml
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from .neo4j_client import Neo4jClient
from .embedding_extractor import get_extractor
from openai import OpenAI


class EventGraphRetriever:
    """Retrieve events and generate context summaries for QA"""

    def __init__(self, neo4j_client: Neo4jClient, config_path: Optional[str] = None):
        """
        Initialize retriever

        Args:
            neo4j_client: Neo4j client instance
            config_path: Path to config.yaml file
        """
        self.client = neo4j_client

        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize LLM client for intent analysis
        llm_config = config['api']['llm']
        self.llm_client = OpenAI(
            base_url=llm_config['base_url'],
            api_key=llm_config['api_key']
        )
        self.llm_model = llm_config['model']
        self.llm_temperature = llm_config.get('temperature', None)  # None means use model default

        # Initialize embedding extractor
        self.embedding_extractor = get_extractor(config_path)

        # Retrieval parameters
        retrieval_config = config.get('retrieval', {})
        self.top_k = retrieval_config.get('top_k', 10)
        self.min_similarity = retrieval_config.get('min_similarity', 0.0)
        self.max_events_from_entities = retrieval_config.get('max_events_from_entities', 10)

    def analyze_query_intent(self, question: str) -> Dict[str, bool]:
        """
        Analyze question to determine which node types to retrieve

        Args:
            question: Question text

        Returns:
            Dictionary with 'event', 'object', 'person' boolean flags
        """
        prompt = f"""Analyze the following question and determine which types of information need to be retrieved from a knowledge graph.

Question: {question}

The knowledge graph contains three types of nodes:
- Event: Activities or actions that happened (e.g., "cooking breakfast", "using laptop")
- Object: Physical objects or tools (e.g., "smartphone", "coffee mug", "laptop")
- Person: People involved (e.g., "Jake", "person wearing red shirt")

Return a JSON object indicating which types are needed to answer the question:
{{"event": true/false, "object": true/false, "person": true/false}}

Rules:
- If the question asks about what happened, actions, or activities: set "event" to true
- If the question mentions specific objects or tools: set "object" to true
- If the question asks about people or who did something: set "person" to true
- Multiple types can be true simultaneously

Return ONLY the JSON object, no explanation."""

        try:
            # Build API call parameters
            api_params = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": "You are a query intent analyzer. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            }

            # Only add temperature if specified in config
            if self.llm_temperature is not None:
                api_params["temperature"] = self.llm_temperature

            response = self.llm_client.chat.completions.create(**api_params)

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            intent = json.loads(result_text)
            return {
                'event': intent.get('event', True),
                'object': intent.get('object', False),
                'person': intent.get('person', False)
            }

        except Exception as e:
            print(f"Error analyzing query intent: {e}")
            # Default: search all types
            return {'event': True, 'object': True, 'person': True}

    def retrieve_core_events(self, question: str, query_time: str,
                            limit: int = 20) -> Tuple[List[Dict[str, Any]], Dict[str, bool]]:
        """
        Retrieve core events relevant to the question using intelligent search

        Args:
            question: Question text
            query_time: Query time (e.g., "DAY1_11210217")
            limit: Maximum number of events to retrieve

        Returns:
            Tuple of (event list, query intent dict)
        """
        # Step 1: Analyze query intent
        intent = self.analyze_query_intent(question)
        print(f"Query intent: {intent}")

        # Step 2: Generate query embedding
        query_embedding = self.embedding_extractor.extract_text_embedding(question)

        # Step 3: Search for relevant nodes based on intent
        all_event_ids = set()
        all_events = []

        # Search all node types according to intent
        similar_events = []
        similar_entities = []
        entity_ids = []

        # Search Events
        if intent['event']:
            event_results = self.client.search_similar_nodes(
                query_embedding=query_embedding,
                node_type="Event",
                top_k=self.top_k,
                min_similarity=self.min_similarity
            )

            for result in event_results:
                props = result['properties']
                if props.get('timestamp', '') <= query_time:
                    similar_events.append({
                        'id': result['id'],
                        'meta_id': props.get('meta_id', 0),
                        'timestamp': props.get('timestamp', ''),
                        'caption': props.get('caption', ''),
                        'video_path': props.get('video_path', ''),
                        'embedding': props.get('embedding', []),
                        'similarity': result.get('similarity', 0.0)
                    })

        # Search Objects
        if intent['object']:
            object_results = self.client.search_similar_nodes(
                query_embedding=query_embedding,
                node_type="Object",
                top_k=self.top_k,
                min_similarity=self.min_similarity
            )
            entity_ids.extend([r['id'] for r in object_results])
            similar_entities.extend([{'id': r['id'], 'type': 'Object'} for r in object_results])

        # Search Persons
        if intent['person']:
            person_results = self.client.search_similar_nodes(
                query_embedding=query_embedding,
                node_type="Person",
                top_k=self.top_k,
                min_similarity=self.min_similarity
            )
            entity_ids.extend([r['id'] for r in person_results])
            similar_entities.extend([{'id': r['id'], 'type': 'Person'} for r in person_results])

        # Step 4: Check connectivity between searched nodes
        # Only keep nodes that have connections to each other
        validated_events = []
        validated_entity_ids = []

        if similar_events and similar_entities:
            # Check which Events connect to which Entities
            event_ids_list = [e['id'] for e in similar_events]
            connected_entities_data = self.client.get_event_entities(event_ids_list)

            # Get all entity IDs connected to these events
            connected_entity_set = set()
            for obj in connected_entities_data['objects']:
                connected_entity_set.add(obj.get('id'))
            for person in connected_entities_data['persons']:
                connected_entity_set.add(person.get('id'))

            # Filter: only keep Events that connect to searched Entities
            for event in similar_events:
                event_single_entities = self.client.get_event_entities([event['id']])
                event_entity_ids = set()
                for obj in event_single_entities['objects']:
                    event_entity_ids.add(obj.get('id'))
                for person in event_single_entities['persons']:
                    event_entity_ids.add(person.get('id'))

                # Check if this event connects to any searched entity
                if any(eid in entity_ids for eid in event_entity_ids):
                    validated_events.append(event)

            # Filter: only keep Entities that connect to searched Events
            validated_entity_ids = [eid for eid in entity_ids if eid in connected_entity_set]

            print(f"✓ Similar Events: {len(similar_events)} → {len(validated_events)} connected")
            print(f"✓ Similar Entities: {len(similar_entities)} → {len(validated_entity_ids)} connected")

        elif similar_events:
            # Only events, no entities to check
            validated_events = similar_events
            print(f"✓ Similar Events: found {len(validated_events)}")

        elif similar_entities:
            # Only entities, no events yet
            validated_entity_ids = entity_ids
            print(f"✓ Similar Entities: found {len(validated_entity_ids)}")

        # Step 5: Layered retrieval strategy
        # Priority: If we have Events (connected or not), use them
        if similar_events:
            # Use Events (prefer connected ones if available, otherwise use all)
            events_to_use = validated_events if validated_events else similar_events
            for event in events_to_use:
                event_id = event['id']
                all_event_ids.add(event_id)
                all_events.append(event)

        elif validated_entity_ids or entity_ids:
            # Priority 2: No Events found, find events through entities
            entities_to_use = validated_entity_ids if validated_entity_ids else entity_ids
            entity_events = self.client.find_events_involving_entities(entities_to_use)
            added_count = 0

            for event in entity_events:
                # Stop if we've reached the limit
                if added_count >= self.max_events_from_entities:
                    break

                if event.get('timestamp', '') <= query_time:
                    event_id = event['id']
                    if event_id not in all_event_ids:
                        all_event_ids.add(event_id)
                        all_events.append(event)
                        added_count += 1

            total_found = len(entity_events)
            if total_found > added_count:
                print(f"  → Events through entities: found {total_found}, added {added_count} (limited to {self.max_events_from_entities})")
            else:
                print(f"  → Events through entities: found {total_found}, added {added_count}")

        # Step 5: Get entities for all retrieved events
        connected_entity_ids = set()
        if all_event_ids:
            entities_data = self.client.get_event_entities(list(all_event_ids))

            # Attach entities to events and track connected entities
            for event in all_events:
                event['entities'] = []

                # Add objects
                for obj in entities_data['objects']:
                    event['entities'].append({
                        'type': 'Object',
                        'name': obj.get('name', ''),
                        'description': obj.get('description', '')
                    })
                    connected_entity_ids.add(obj.get('id'))

                # Add persons
                for person in entities_data['persons']:
                    event['entities'].append({
                        'type': 'Person',
                        'name': person.get('name', ''),
                        'description': person.get('description', '')
                    })
                    connected_entity_ids.add(person.get('id'))

        # Step 6: Sort by timestamp and limit
        all_events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Print final summary
        print(f"\n📊 Final Result: {len(all_event_ids)} unique events")
        if len(all_events) > limit:
            print(f"   Returning top {limit} events sorted by timestamp")

        return all_events[:limit], intent

    def expand_from_events(self, timestamps: List[str], hops: int = 2,
                          limit: int = 10) -> Dict[str, Any]:
        """
        Expand graph from core events

        Args:
            timestamps: List of core event timestamps
            hops: Number of hops to expand
            limit: Maximum number of neighbors per event

        Returns:
            Dictionary with expanded events and relationships
        """
        if not timestamps:
            return {'events': [], 'relationships': []}

        with self.client.driver.session() as session:
            # Build query with literal hops value (Cypher doesn't support parameters in relationship patterns)
            query = f"""
            MATCH (core:Event)
            WHERE core.timestamp IN $timestamps

            // Expand to neighbors
            OPTIONAL MATCH path = (core)-[r*1..{hops}]-(neighbor:Event)
            WHERE neighbor.timestamp <> core.timestamp

            WITH core, path, neighbor, r
            LIMIT $limit

            RETURN
                DISTINCT neighbor.meta_id as meta_id,
                neighbor.timestamp as timestamp,
                neighbor.caption as caption,
                type(relationships(path)[0]) as relation_type,
                core.timestamp as related_to_core
            """

            result = session.run(query,
                               timestamps=timestamps,
                               limit=limit)

            expanded_events = []
            relationships = []

            for record in result:
                if record['timestamp']:
                    expanded_events.append({
                        'meta_id': record['meta_id'],
                        'timestamp': record['timestamp'],
                        'caption': record['caption']
                    })

                    if record['relation_type']:
                        relationships.append({
                            'from': record['related_to_core'],
                            'to': record['timestamp'],
                            'type': record['relation_type']
                        })

            return {
                'events': expanded_events,
                'relationships': relationships
            }

    def generate_summary(self, question: str, query_time: str,
                        core_events: List[Dict],
                        expanded_events: List[Dict],
                        relationships: List[Dict],
                        intent: Dict[str, bool],
                        top_k: int = 5) -> str:
        """
        Generate summary context text in "Relevant Memory Snippets" format

        Args:
            question: Original question
            query_time: Query time
            core_events: Core retrieved events
            expanded_events: Expanded events from graph traversal
            relationships: Relationships between events
            intent: Query intent (which types were searched)
            top_k: Number of top events to include in detail

        Returns:
            Formatted summary text
        """
        # Merge core and expanded events
        all_events = core_events.copy()

        # Add expanded events that are not already in core
        core_timestamps = {e.get('timestamp') for e in core_events}
        for exp_event in expanded_events:
            if exp_event.get('timestamp') not in core_timestamps:
                all_events.append(exp_event)

        # Sort by timestamp (most recent first)
        all_events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Build relationship map: timestamp -> list of (relation_type, target_timestamp)
        relation_map = {}
        for rel in relationships:
            from_ts = rel.get('from')
            to_ts = rel.get('to')
            rel_type = rel.get('type', 'RELATED')

            if from_ts not in relation_map:
                relation_map[from_ts] = []
            relation_map[from_ts].append((rel_type, to_ts))

            # Also add reverse direction
            if to_ts not in relation_map:
                relation_map[to_ts] = []
            relation_map[to_ts].append((rel_type, from_ts))

        # Build summary in "Relevant Memory Snippets" format
        summary = []
        summary.append("=" * 60)
        summary.append("Relevant Memory Snippets:")
        summary.append("=" * 60)
        summary.append("")

        # Show each event with its entities and relationships
        for i, event in enumerate(all_events[:top_k], 1):
            summary.append(f"[{i}]")

            # Always show Event (caption) with similarity
            caption = event.get('caption', 'No caption')
            similarity = event.get('similarity', None)
            summary.append(f"Event: {caption}")
            if similarity is not None:
                summary.append(f"Similarity: {similarity:.4f}")
            else:
                summary.append(f"Similarity: N/A (expanded)")

            # Get entities for this event
            entities = event.get('entities', [])

            # Show Object only if intent includes object
            if intent.get('object', False):
                objects = [e['name'] for e in entities if e['type'] == 'Object']
                if objects:
                    summary.append(f"Object: {', '.join(objects)}")

            # Show Human only if intent includes person
            if intent.get('person', False):
                persons = [e['name'] for e in entities if e['type'] == 'Person']
                if persons:
                    summary.append(f"Human: {', '.join(persons)}")

            # Show Timestamp
            timestamp = event.get('timestamp', 'unknown')
            summary.append(f"Timestamp: {timestamp}")
            summary.append("")

        if len(all_events) > top_k:
            summary.append(f"... and {len(all_events) - top_k} more memory snippets")
            summary.append("")

        summary.append("=" * 60)

        return "\n".join(summary)

    def generate_cypher_query(self, core_events: List[Dict], expanded_events: List[Dict]) -> str:
        """
        Generate Cypher query to retrieve all nodes and relationships from results

        Args:
            core_events: Core retrieved events (with 'id' = elementId)
            expanded_events: Expanded events (with 'timestamp')

        Returns:
            Cypher query string that returns nodes and relationships (n, r, m)
        """
        # Collect element IDs from core events
        element_ids = []
        for event in core_events:
            event_id = event.get('id')
            if event_id:
                element_ids.append(event_id)

        # Collect timestamps from expanded events
        timestamps = []
        for event in expanded_events:
            timestamp = event.get('timestamp')
            if timestamp:
                timestamps.append(timestamp)

        if not element_ids and not timestamps:
            return "// No events found"

        # Build Cypher query
        cypher_lines = []
        cypher_lines.append("// Generated Cypher query for retrieved events and relationships")
        cypher_lines.append("// Copy and paste this into Neo4j Browser to visualize results")
        cypher_lines.append("")
        cypher_lines.append("MATCH (n:Event)")
        cypher_lines.append("WHERE")

        # Add element ID conditions
        conditions = []
        if element_ids:
            # Format element IDs with proper escaping
            ids_str = ", ".join([f"'{eid}'" for eid in element_ids])
            conditions.append(f"  elementId(n) IN [{ids_str}]")

        # Add timestamp conditions
        if timestamps:
            ts_str = ", ".join([f"'{ts}'" for ts in timestamps])
            if element_ids:
                conditions.append(f"  OR n.timestamp IN [{ts_str}]")
            else:
                conditions.append(f"  n.timestamp IN [{ts_str}]")

        cypher_lines.append("\n".join(conditions))
        cypher_lines.append("")
        cypher_lines.append("// Get all relationships and connected nodes")
        cypher_lines.append("OPTIONAL MATCH (n)-[r]-(m)")
        cypher_lines.append("")
        cypher_lines.append("// Return for visualization")
        cypher_lines.append("RETURN n, r, m")

        return "\n".join(cypher_lines)

    def retrieve_and_summarize(self, question: str, query_time: str,
                              top_k: int = 5,
                              expansion_hops: int = 2,
                              expansion_limit: int = 10,
                              core_limit: int = 20) -> Dict[str, Any]:
        """
        Main retrieval interface: retrieve events and generate summary with Cypher query

        Args:
            question: Question text
            query_time: Query time (e.g., "DAY1_11210217")
            top_k: Number of top events to show in detail
            expansion_hops: Number of hops for graph expansion
            expansion_limit: Limit for expanded neighbors
            core_limit: Maximum core events to retrieve

        Returns:
            Dictionary with:
                - 'summary': Summary context text
                - 'cypher': Cypher query to visualize results in Neo4j
                - 'core_events_count': Number of core events retrieved
                - 'expanded_events_count': Number of expanded events
        """
        # Step 1: Retrieve core events using intelligent search
        core_events, intent = self.retrieve_core_events(question, query_time, core_limit)

        if not core_events:
            return {
                'summary': "No relevant events found for the given query.",
                'cypher': "// No events found",
                'core_events_count': 0,
                'expanded_events_count': 0
            }

        # Step 2: Expand from core events
        core_timestamps = [e['timestamp'] for e in core_events]
        expansion_result = self.expand_from_events(
            timestamps=core_timestamps,
            hops=expansion_hops,
            limit=expansion_limit
        )
        expanded_events = expansion_result['events']
        relationships = expansion_result['relationships']

        # Step 3: Generate summary with intent and relationships
        summary = self.generate_summary(question, query_time,
                                       core_events, expanded_events,
                                       relationships, intent, top_k)

        # Step 4: Generate Cypher query
        cypher_query = self.generate_cypher_query(core_events, expanded_events)

        return {
            'summary': summary,
            'cypher': cypher_query,
            'core_events_count': len(core_events),
            'expanded_events_count': len(expanded_events),
            'core_events': core_events,  # Return actual events for evaluation
            'expanded_events': expanded_events  # Return actual events for evaluation
        }
