# Event Graph

Convert event JSONs into a Neo4j graph.

## Pipeline

| Step | Script | Purpose |
|---|---|---|
| 1 | `scripts/initialize_neo4j.py` | One-time: create indexes/constraints on a fresh Neo4j instance |
| 2 | `scripts/build_graph_from_json.py` | Ingest memory's event JSONs → event nodes + embedding vectors |
| 3 | `scripts/build_clusters.py` | (optional) HDBSCAN cluster events, write cluster nodes |
| 4 | `scripts/evaluate_retrieval.py` | Retrieval-accuracy evaluation |

Other scripts in `scripts/` (add/query/visualize/alternative evaluators) are kept for reference.

## Running

```bash
# 1. Start Neo4j (Docker)
docker run -d --name neo4j-egoself \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# 2. Initialize schema (once)
python scripts/initialize_neo4j.py

# 3. Ingest events + embeddings
python scripts/build_graph_from_json.py /path/to/memory/data/events/DAY1

# 4. (optional) cluster
python scripts/build_clusters.py

# 5. Evaluate retrieval
python scripts/evaluate_retrieval.py
```

## Configs

- `configs/config.yaml` — embedding API, LLM API, Neo4j connection, retrieval parameters

## Graph schema

| Node | Properties |
|---|---|
| `Event` | `event_id`, `caption`, `timestamp`, `embedding` (vector), ... |
| `Entity` | `entity_id`, `description`, ... |
| `Cluster` | `cluster_id`, `summary` (optional) |

| Relationship | Meaning |
|---|---|
| `(Event)-[:HAS_ENTITY]->(Entity)` | Event references an interaction object / person |
| `(Event)-[:NEXT_EVENT]->(Event)` | Temporal / causal link |
| `(Event)-[:IN_CLUSTER]->(Cluster)` | Event belongs to a cluster |

## Outputs

- Neo4j graph (browse at `http://localhost:7474`, default `neo4j` / `password`)
