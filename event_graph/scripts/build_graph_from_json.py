#!/usr/bin/env python3
"""
Build Event Graph from JSON Files
Read event JSON files and build a graph in Neo4j with Events, Objects, and Persons
"""

import argparse
import sys
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from event_graph.neo4j_client import Neo4jClient
from event_graph.embedding_extractor import get_extractor


class GraphBuilder:
    """Build event graph from JSON files"""

    def __init__(self, neo4j_client: Neo4jClient, config_path: str = None,
                 embeddings_dir: str = None):
        self.client = neo4j_client
        self.event_element_ids = {}  # Map event_id -> Neo4j element_id
        self.global_entity_element_ids = {}  # Map global_name -> Neo4j element_id
        self.global_entities = {}  # Global entity data loaded from global_entities.json
        self.voice_database = []  # Voice data loaded from voice database.json
        self.event_sequence_id = 0  # Sequential ID counter for events (1, 2, 3...)
        # Initialize embedding extractor with config
        if config_path is None:
            # Default config path
            config_path = str(Path(__file__).parent.parent / "configs" / "config.yaml")
        self.embedding_extractor = get_extractor(config_path)

        # Set embeddings directory
        self.embeddings_dir = embeddings_dir  # Will be set when loading entities

    def load_json_file(self, json_path: str) -> Dict[str, Any]:
        """Load and parse JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_global_entities(self, global_entities_path: str):
        """
        Load global entities from global_entities.json

        Args:
            global_entities_path: Path to global_entities.json file
        """
        print(f"\nLoading global entities from: {global_entities_path}")
        self.global_entities = self.load_json_file(global_entities_path)
        print(f"  Loaded {len(self.global_entities)} global entities")

        # Set embeddings directory based on entities path if not already set
        if self.embeddings_dir is None:
            # entities_path: /path/to/data_day1/entities/global_entities.json
            # embeddings_dir: /path/to/data_day1/features/embeddings
            data_root = Path(global_entities_path).parent.parent
            self.embeddings_dir = str(data_root / "features" / "embeddings")
            print(f"  Embeddings will be saved to: {self.embeddings_dir}")

        # Count by type
        object_count = sum(1 for k in self.global_entities.keys() if k.startswith('object_'))
        person_count = sum(1 for k in self.global_entities.keys() if k.startswith('person_'))
        print(f"  - Objects: {object_count}")
        print(f"  - Persons: {person_count}")

    def create_global_entity_nodes(self):
        """
        Create nodes for all global entities from global_entities.json
        Uses the most representative description for each entity
        """
        if not self.global_entities:
            print("Warning: No global entities loaded. Call load_global_entities() first.")
            return

        print("\n=== Creating Global Entity Nodes ===")

        for global_name, entity_data in self.global_entities.items():
            # Get the most recent description (or aggregate descriptions)
            descriptions = entity_data.get('descriptions', [])
            if descriptions:
                # Use the most recent description
                latest_desc = descriptions[-1]
                description = latest_desc['text']
            else:
                description = ""

            # Get metadata
            metadata = entity_data.get('metadata', {})

            # Prepare properties
            properties = {
                'match_count': metadata.get('match_count', 0),
                'created_at': metadata.get('created_at', ''),
                'last_updated': metadata.get('last_updated', ''),
                'all_descriptions': [d['text'] for d in descriptions]
            }

            # Generate or load embedding from description
            embedding = None
            if description:
                try:
                    # Try to load existing embedding first
                    embedding = self.embedding_extractor.load_embedding(global_name, self.embeddings_dir)
                    if embedding is not None:
                        print(f"    ✓ Loaded cached embedding for {global_name}")
                    else:
                        # Generate new embedding if not cached
                        embedding = self.embedding_extractor.extract_text_embedding(description)
                        # Save embedding to file
                        self.embedding_extractor.save_embedding(
                            embedding, global_name, self.embeddings_dir
                        )
                        print(f"    ✓ Generated and saved embedding for {global_name}")
                except Exception as e:
                    print(f"    Warning: Failed to generate embedding for {global_name}: {e}")

            # Create node based on type
            if global_name.startswith('person_'):
                element_id = self.client.create_person(
                    name=global_name,
                    description=description,
                    properties=properties,
                    embedding=embedding
                )
                print(f"  Created Person: {global_name}")
            elif global_name.startswith('object_'):
                element_id = self.client.create_object(
                    name=global_name,
                    description=description,
                    properties=properties,
                    embedding=embedding
                )
                print(f"  Created Object: {global_name}")
            else:
                print(f"  Warning: Unknown entity type: {global_name}")
                continue

            # Store mapping
            self.global_entity_element_ids[global_name] = element_id

    def load_voice_database(self, voice_db_path: str):
        """
        Load voice database from database.json

        Args:
            voice_db_path: Path to voices/voiceprint/database.json file
        """
        print(f"\nLoading voice database from: {voice_db_path}")
        self.voice_database = self.load_json_file(voice_db_path)
        print(f"  Loaded {len(self.voice_database)} voice profiles")

        # Build voice_id -> related_person mapping for later use
        self.voice_to_person = {}
        for voice_data in self.voice_database:
            voice_id = voice_data.get('voice_id')
            related_persons = voice_data.get('related_person', {})
            self.voice_to_person[voice_id] = related_persons

    def create_event_node(self, event_data: Dict[str, Any]) -> str:
        """
        Create Event node from JSON data

        Args:
            event_data: Event data from JSON

        Returns:
            Neo4j element ID of created event
        """
        event_id = event_data.get('event_id')
        video_path = event_data.get('video_path', '')
        caption = event_data.get('caption', '')
        timestamp = event_data.get('timestamp', '')

        # Generate or load embedding from caption
        embedding = None
        if caption:
            try:
                # Try to load existing embedding first
                embedding = self.embedding_extractor.load_embedding(event_id, self.embeddings_dir)
                if embedding is not None:
                    print(f"    ✓ Loaded cached embedding for {event_id}")
                else:
                    # Generate new embedding if not cached
                    embedding = self.embedding_extractor.extract_text_embedding(caption)
                    # Save embedding to file
                    self.embedding_extractor.save_embedding(
                        embedding, event_id, self.embeddings_dir
                    )
                    print(f"    ✓ Generated and saved embedding for {event_id}")
            except Exception as e:
                print(f"    Warning: Failed to generate embedding for event {event_id}: {e}")

        # Increment sequence ID
        self.event_sequence_id += 1

        print(f"  Creating Event node: {event_id} (ID: {self.event_sequence_id})")
        element_id = self.client.create_event_with_id(
            event_id=event_id,
            meta_id=self.event_sequence_id,
            video_path=video_path,
            caption=caption,
            timestamp=timestamp,
            embedding=embedding
        )

        # Store mapping for later relationship creation
        self.event_element_ids[event_id] = element_id
        return element_id

    def extract_and_link_entities(self, event_data: Dict[str, Any], event_element_id: str):
        """
        Link event to global entities using global_name mapping from event JSON
        Each entity can link to multiple events, but no duplicate links within same event

        Args:
            event_data: Event data from JSON
            event_element_id: Neo4j element ID of the event
        """
        attributes = event_data.get('attributes', {})
        interaction_objects = attributes.get('interaction_object', [])

        if not interaction_objects:
            return

        print(f"  Linking {len(interaction_objects)} entities...")

        # Track entities linked in this event to avoid duplicates
        linked_in_this_event = set()

        for obj in interaction_objects:
            local_name = obj.get('name', '')
            global_name = obj.get('global_name', '')
            action = obj.get('action', '')
            match_info = obj.get('match_info', {})

            # Get interaction segments if available
            segments = obj.get('interaction_segments', [])

            if not global_name:
                print(f"    Warning: No global_name for {local_name}, skipping")
                continue

            # Skip if already linked in this event (avoid duplicates)
            if global_name in linked_in_this_event:
                print(f"    - Skipped {global_name} (already linked in this event)")
                continue

            # Find global entity element_id
            global_entity_id = self.global_entity_element_ids.get(global_name)

            if not global_entity_id:
                print(f"    Warning: Global entity {global_name} not found in graph, skipping")
                continue

            # Prepare relationship properties
            relationship_props = {
                'local_name': local_name,
                'action': action,
                'text_similarity': match_info.get('text_similarity', 0.0),
                'vision_similarity': match_info.get('vision_similarity', 0.0),
                'final_score': match_info.get('final_score', 0.0),
                'is_new': match_info.get('is_new', False)
            }

            # Add segment info if available
            if segments:
                relationship_props['segment_count'] = len(segments)

            # Link event to global entity
            self.client.link_event_to_entity(
                event_id=event_element_id,
                entity_id=global_entity_id,
                relationship_type="INVOLVES",
                properties=relationship_props
            )

            # Mark this entity as linked in this event
            linked_in_this_event.add(global_name)

            print(f"    - Linked {local_name} → {global_name} (score: {match_info.get('final_score', 0.0):.3f})")

    def link_voices_to_events(self, event_data: Dict[str, Any], event_element_id: str):
        """
        Link speakers (Person entities) to events via SPEAKS relationship
        Voice information is stored as relationship properties

        Args:
            event_data: Event data from JSON
            event_element_id: Neo4j element ID of the event
        """
        attributes = event_data.get('attributes', {})
        interaction_language = attributes.get('interaction_language', [])

        if not interaction_language:
            return

        print(f"  Linking {len(interaction_language)} speakers...")

        for lang in interaction_language:
            speaker = lang.get('speaker', '')
            global_voice = lang.get('global_voice', '')
            text = lang.get('text', '')
            start_time = lang.get('start_time', 0.0)
            end_time = lang.get('end_time', 0.0)
            description = lang.get('description', '')
            reason = lang.get('reason', '')

            if not global_voice:
                print(f"    Warning: No global_voice for speaker {speaker}, skipping")
                continue

            # Find corresponding person from voice database
            related_persons = self.voice_to_person.get(global_voice, {})

            # Find the most relevant person for this event
            # Priority: 1) "I" (camera wearer), 2) global person entities
            person_entity = None
            if speaker == 'I' or speaker == 'person1':
                # This is the camera wearer, try to find "I" in related_persons
                if 'I' in related_persons:
                    # "I" is not a global entity, skip for now
                    print(f"    - Skipped speaker '{speaker}' (camera wearer, no global entity)")
                    continue

            # Try to find a global person entity
            for person_name in related_persons.keys():
                if person_name.startswith('person_'):  # Global entity
                    person_entity = person_name
                    break

            if not person_entity:
                print(f"    Warning: No global person found for voice {global_voice}, skipping")
                continue

            # Find person element_id
            person_element_id = self.global_entity_element_ids.get(person_entity)

            if not person_element_id:
                print(f"    Warning: Person {person_entity} not found in graph, skipping")
                continue

            # Prepare relationship properties
            relationship_props = {
                'speaker_name': speaker,
                'voice_id': global_voice,
                'text': text,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'description': description,
                'reason': reason
            }

            # Link event to person via SPEAKS
            self.client.link_event_to_entity(
                event_id=event_element_id,
                entity_id=person_element_id,
                relationship_type="SPEAKS",
                properties=relationship_props
            )
            print(f"    - Linked speaker '{speaker}' ({global_voice}) → {person_entity}: '{text}'")

    def create_event_relationships(self, event_data: Dict[str, Any]):
        """
        Create relationships between events based on relations in JSON

        Args:
            event_data: Event data from JSON
        """
        event_id = event_data.get('event_id')
        relations = event_data.get('relations', [])

        if not relations:
            return

        print(f"  Creating {len(relations)} relationships from {event_id}...")

        for relation in relations:
            rel_type = relation.get('type', 'RELATED_TO')
            related_event_id = relation.get('related_event_id')

            if not related_event_id:
                continue

            # Map relation type to Neo4j relationship type
            relationship_type = self._map_relation_type(rel_type)

            # Extract properties
            properties = {}
            if 'reason' in relation:
                properties['reason'] = relation['reason']
            if 'time_diff_seconds' in relation:
                properties['time_diff_seconds'] = relation['time_diff_seconds']

            try:
                self.client.link_events_by_event_id(
                    from_event_id=event_id,
                    to_event_id=related_event_id,
                    relationship_type=relationship_type,
                    properties=properties
                )
                print(f"    - {relationship_type} -> {related_event_id}")
            except Exception as e:
                print(f"    ! Warning: Could not create relationship to {related_event_id}: {e}")

    def _map_relation_type(self, rel_type: str) -> str:
        """Map JSON relation type to Neo4j relationship type"""
        mapping = {
            'causal': 'CAUSAL',
            'temporal_after': 'TEMPORAL_AFTER',
            'same_activity_non_causal': 'SAME_ACTIVITY',
            'related_to': 'RELATED_TO'
        }
        return mapping.get(rel_type, rel_type.upper())

    def build_from_file(self, json_path: str):
        """
        Build graph from a single JSON file

        Args:
            json_path: Path to JSON file
        """
        print(f"\nProcessing: {os.path.basename(json_path)}")

        # Load JSON data
        event_data = self.load_json_file(json_path)

        # Create Event node
        event_element_id = self.create_event_node(event_data)

        # Link to global entities
        self.extract_and_link_entities(event_data, event_element_id)

        # Link to voices
        self.link_voices_to_events(event_data, event_element_id)

    def build_from_directory(self, directory: str):
        """
        Build graph from all JSON files in a directory

        Args:
            directory: Directory containing JSON files
        """
        # Find all JSON files
        json_files = sorted(Path(directory).glob('*.json'))

        if not json_files:
            print(f"No JSON files found in {directory}")
            return

        print(f"Found {len(json_files)} JSON file(s)")
        print("="*60)

        # First pass: Create all events and entities
        print("\n=== PASS 1: Creating Events and Entities ===")
        for json_path in json_files:
            self.build_from_file(str(json_path))

        # Second pass: Create relationships between events
        print("\n=== PASS 2: Creating Event Relationships ===")
        for json_path in json_files:
            print(f"\nProcessing relationships: {json_path.name}")
            event_data = self.load_json_file(str(json_path))
            self.create_event_relationships(event_data)

    def print_summary(self):
        """Print graph statistics"""
        with self.client.driver.session() as session:
            # Count events
            result = session.run("MATCH (e:Event) RETURN count(e) as count")
            event_count = result.single()['count']

            # Count objects
            result = session.run("MATCH (o:Object) RETURN count(o) as count")
            object_count = result.single()['count']

            # Count persons
            result = session.run("MATCH (p:Person) RETURN count(p) as count")
            person_count = result.single()['count']

            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            relationship_count = result.single()['count']

            # Count relationship types
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
            """)
            rel_types = {record['rel_type']: record['count'] for record in result}

            # Count entities linked vs total
            result = session.run("""
                MATCH (e:Event)-[:INVOLVES]->(entity)
                RETURN count(DISTINCT entity) as linked_count
            """)
            record = result.single()
            linked_count = record['linked_count'] if record else 0

            print("\n" + "="*60)
            print("Graph Summary")
            print("="*60)
            print(f"Events:         {event_count}")
            print(f"Objects:        {object_count}")
            print(f"Persons:        {person_count}")
            print(f"Total Entities: {object_count + person_count}")
            print(f"Linked Entities: {linked_count}")
            print(f"Relationships:  {relationship_count}")
            print("\nRelationship Types:")
            for rel_type, count in rel_types.items():
                print(f"  {rel_type}: {count}")
            print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Build event graph from JSON files with global entities and voices"
    )
    parser.add_argument(
        "path",
        type=str,
        nargs='?',
        default="memory/data/events/DAY1",
        help="Path to an event JSON file or directory (default: memory/data/events/DAY1)"
    )
    parser.add_argument(
        "--entities",
        type=str,
        default=None,
        help="Path to the global entities JSON file (auto-derived from `path` if omitted)"
    )
    parser.add_argument(
        "--voices",
        type=str,
        default=None,
        help="Path to the voice database JSON file (auto-derived from `path` if omitted)"
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
        help="Clear all existing data before building"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.path):
        print(f"Error: Event path does not exist: {args.path}", file=sys.stderr)
        sys.exit(1)

    # Auto-derive --entities and --voices by walking up `path` until we find a
    # day-root directory containing both `entities/global_entities.json` and
    # `voices/voiceprint/database.json`.
    if args.entities is None or args.voices is None:
        start = Path(args.path).resolve()
        if start.is_file():
            start = start.parent
        day_root = None
        for parent in [start, *start.parents]:
            if (parent / "entities" / "global_entities.json").exists() and \
               (parent / "voices" / "voiceprint" / "database.json").exists():
                day_root = parent
                break
        if day_root is None:
            print(
                f"Error: could not auto-derive --entities/--voices from {args.path}.\n"
                f"Pass them explicitly.", file=sys.stderr
            )
            sys.exit(1)
        if args.entities is None:
            args.entities = str(day_root / "entities" / "global_entities.json")
            print(f"Auto-derived --entities: {args.entities}")
        if args.voices is None:
            args.voices = str(day_root / "voices" / "voiceprint" / "database.json")
            print(f"Auto-derived --voices:   {args.voices}")

    if not os.path.exists(args.entities):
        print(f"Error: Entities file does not exist: {args.entities}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.voices):
        print(f"Error: Voice database file does not exist: {args.voices}", file=sys.stderr)
        sys.exit(1)

    try:
        # Initialize Neo4j client
        print(f"Connecting to Neo4j at {args.neo4j_uri}...")
        client = Neo4jClient(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password
        )
        print("✓ Connected to Neo4j")

        # Clear existing data if requested
        if args.clear:
            print("\nClearing all existing data...")
            client.clear_all()
            print("✓ Database cleared")

            # Also clear embeddings cache
            # Infer embeddings directory from entities path
            data_root = Path(args.entities).parent.parent
            embeddings_dir = data_root / "features" / "embeddings"
            if embeddings_dir.exists():
                print(f"Clearing embeddings cache: {embeddings_dir}")
                shutil.rmtree(embeddings_dir)
                embeddings_dir.mkdir(parents=True, exist_ok=True)
                print("✓ Embeddings cache cleared")

        # Build graph - 3 phases
        builder = GraphBuilder(client)

        # Phase 1: Load and create global entities
        print("\n" + "="*60)
        print("PHASE 1: Loading Global Entities")
        print("="*60)
        builder.load_global_entities(args.entities)
        builder.create_global_entity_nodes()

        # Phase 2: Load voice database (for speaker mapping)
        print("\n" + "="*60)
        print("PHASE 2: Loading Voice Database")
        print("="*60)
        builder.load_voice_database(args.voices)

        # Phase 3: Create events and link to entities/speakers
        print("\n" + "="*60)
        print("PHASE 3: Creating Events and Relationships")
        print("="*60)
        if os.path.isfile(args.path):
            # Single file
            builder.build_from_file(args.path)
        else:
            # Directory
            builder.build_from_directory(args.path)

        # Print summary
        builder.print_summary()

        print("\n✓ Graph construction complete!")
        print(f"View in Neo4j Browser: http://localhost:7474")

        client.close()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
