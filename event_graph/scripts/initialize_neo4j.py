#!/usr/bin/env python3
"""
Initialize Neo4j Database
Script to clear and initialize Neo4j database for event graph
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from event_graph.neo4j_client import Neo4jClient


def main():
    print("Initializing Neo4j Database...")
    print("="*50)

    try:
        # Connect to Neo4j
        print("\n1. Connecting to Neo4j...")
        client = Neo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        print("   ✓ Connected successfully")

        # Clear all existing data
        print("\n2. Clearing existing data...")
        client.clear_all_events()
        print("   ✓ All events and relationships deleted")

        # Create indexes for better performance
        print("\n3. Creating indexes...")
        with client.driver.session() as session:
            # Create index on clip_path for faster lookups
            try:
                session.run(
                    "CREATE INDEX event_clip_path IF NOT EXISTS FOR (e:Event) ON (e.clip_path)"
                )
                print("   ✓ Index on clip_path created")
            except Exception as e:
                print(f"   ⚠ Index already exists")

            # Create index on timestamp
            try:
                session.run(
                    "CREATE INDEX event_timestamp IF NOT EXISTS FOR (e:Event) ON (e.timestamp)"
                )
                print("   ✓ Index on timestamp created")
            except Exception as e:
                print(f"   ⚠ Index already exists")

        # Verify database is empty
        print("\n4. Verifying database...")
        with client.driver.session() as session:
            result = session.run("MATCH (e:Event) RETURN count(e) as count")
            count = result.single()["count"]
            print(f"   ✓ Current events in database: {count}")

        print("\n" + "="*50)
        print("Neo4j database initialized successfully!")
        print("Database is empty and ready for new data")
        print("Connection: bolt://localhost:7687")
        print("="*50)

        client.close()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
