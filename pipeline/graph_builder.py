# graph_builder.py - reads deduped_graph.json and pushes everything into neo4j
# creates Entity nodes (labeled by type) and relationship edges
# for all the claims we extracted.
# every relationship has confidence, mention_count, and the actual evidence excerpts

import json
import os
import sys
import time

# Add project root to path for shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
from tqdm import tqdm

import config

# -- neo4j connection
def get_driver():
    return GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )


# -- schema setup: indexes and constraints for fast lookups
def setup_schema(driver):
    """creates uniqueness constraint on entity id + indexes on name and type"""
    with driver.session() as session:
        # Uniqueness constraint on entity ID
        session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        # Index on canonical_name for search
        session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.canonical_name)")
        # Index on entity_type
        session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)")
    print("  Schema indexes created")


# -- clear existing data (full wipe before re-ingestion)
def clear_graph(driver):
    """deletes everything in the graph - we do a full rebuild each time"""
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as cnt")
        count = result.single()['cnt']
        if count > 0:
            print(f"  Clearing {count} existing nodes...")
            session.run("MATCH (n) DETACH DELETE n")
    print("  Graph cleared")


# -- ingest entities as nodes
def ingest_entities(driver, entities: list):
    """creates Entity nodes and adds type-specific labels (Person, Org, etc)"""
    
    # Map entity_type to Neo4j label
    type_labels = {
        "person": "Person",
        "organization": "Organization",
        "project": "Project",
        "topic": "Topic",
        "meeting": "Meeting",
    }
    
    batch_size = 200
    created = 0
    
    with driver.session() as session:
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            
            # Use UNWIND for batch insert
            session.run("""
                UNWIND $batch AS e
                CREATE (n:Entity {
                    id: e.id,
                    canonical_name: e.canonical_name,
                    entity_type: e.entity_type,
                    aliases: e.aliases,
                    mention_count: e.mention_count
                })
            """, batch=batch)
            created += len(batch)
        
        # Add type-specific labels
        for etype, label in type_labels.items():
            session.run(f"""
                MATCH (n:Entity {{entity_type: $etype}})
                SET n:{label}
            """, etype=etype)
    
    return created


# -- ingest claims as relationships between entities
def ingest_claims(driver, claims: list):
    """creates typed relationships from claims. batches per claim_type
    since neo4j doesn't let you parameterize relationship types."""
    
    # Map claim_type to Neo4j relationship type
    rel_types = {
        "works_at": "WORKS_AT",
        "reports_to": "REPORTS_TO",
        "works_on": "WORKS_ON",
        "part_of": "PART_OF",
        "discussed_in": "DISCUSSED_IN",
        "decided": "DECIDED",
        "communicated_with": "COMMUNICATED_WITH",
        "attended": "ATTENDED",
        "assigned_to": "ASSIGNED_TO",
        "mentioned_in": "MENTIONED_IN",
        "contradicts": "CONTRADICTS",
        "supersedes": "SUPERSEDES",
    }
    
    created = 0
    skipped = 0
    
    with driver.session() as session:
        # Group claims by type for efficient batch creation
        for claim_type, rel_type in rel_types.items():
            type_claims = [c for c in claims if c['claim_type'] == claim_type]
            if not type_claims:
                continue
            
            batch_size = 100
            for i in range(0, len(type_claims), batch_size):
                batch = type_claims[i:i + batch_size]
                
                # Prepare batch data
                batch_data = []
                for c in batch:
                    # Extract first few evidence excerpts
                    excerpts = [e.get('excerpt', '')[:200] for e in c.get('evidence', [])[:5]]
                    batch_data.append({
                        "subject_id": c['subject_id'],
                        "object_id": c['object_id'],
                        "confidence": c['confidence'],
                        "mention_count": c['mention_count'],
                        "detail": c.get('detail', '') or '',
                        "excerpts": excerpts,
                        "claim_id": c['id'],
                    })
                
                # Use APOC or dynamic relationship creation
                # Since Neo4j doesn't support parameterized relationship types in MERGE,
                # we create per-type queries
                result = session.run(f"""
                    UNWIND $batch AS c
                    MATCH (s:Entity {{id: c.subject_id}})
                    MATCH (o:Entity {{id: c.object_id}})
                    CREATE (s)-[r:{rel_type} {{
                        claim_id: c.claim_id,
                        confidence: c.confidence,
                        mention_count: c.mention_count,
                        detail: c.detail,
                        excerpts: c.excerpts
                    }}]->(o)
                    RETURN count(r) AS cnt
                """, batch=batch_data)
                
                cnt = result.single()['cnt']
                created += cnt
                skipped += len(batch) - cnt
    
    return created, skipped


# -- graph stats
def get_graph_stats(driver) -> dict:
    """pull summary stats from neo4j - node/edge counts, top connected entities, etc"""
    with driver.session() as session:
        stats = {}
        
        # Node counts
        result = session.run("MATCH (n:Entity) RETURN count(n) as cnt")
        stats['total_nodes'] = result.single()['cnt']
        
        # Relationship counts
        result = session.run("MATCH ()-[r]->() RETURN count(r) as cnt")
        stats['total_relationships'] = result.single()['cnt']
        
        # Per-type counts
        result = session.run("""
            MATCH (n:Entity)
            RETURN n.entity_type AS type, count(n) AS cnt
            ORDER BY cnt DESC
        """)
        stats['node_types'] = {r['type']: r['cnt'] for r in result}
        
        # Relationship type counts
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS cnt
            ORDER BY cnt DESC
        """)
        stats['rel_types'] = {r['type']: r['cnt'] for r in result}
        
        # Most connected nodes
        result = session.run("""
            MATCH (n:Entity)-[r]-()
            RETURN n.canonical_name AS name, n.entity_type AS type, count(r) AS degree
            ORDER BY degree DESC
            LIMIT 10
        """)
        stats['top_connected'] = [
            {"name": r['name'], "type": r['type'], "degree": r['degree']}
            for r in result
        ]
        
        # Connected components (approximate)
        result = session.run("""
            MATCH (n:Entity)
            WHERE NOT (n)--()
            RETURN count(n) AS isolated
        """)
        stats['isolated_nodes'] = result.single()['isolated']
        
        return stats


# -- main
def run_graph_builder():
    """loads the deduped graph json and pushes it all into neo4j"""
    
    print("=" * 70)
    print("NEO4J GRAPH BUILDER")
    print("=" * 70)
    
    # Load deduped graph
    graph_path = os.path.join(config.OUTPUT_DIR, "deduped_graph.json")
    if not os.path.exists(graph_path):
        print(f"[ERROR] Deduped graph not found at {graph_path}")
        print("Run dedup.py first!")
        return
    
    print(f"\nLoading deduped graph from {graph_path}...")
    with open(graph_path, 'r') as f:
        graph = json.load(f)
    
    entities = graph['entities']
    claims = graph['claims']
    print(f"  Entities: {len(entities)}")
    print(f"  Claims:   {len(claims)}")
    
    # Connect to Neo4j
    print(f"\nConnecting to Neo4j at {config.NEO4J_URI}...")
    driver = get_driver()
    
    try:
        # Verify connection
        driver.verify_connectivity()
        print("  Connected!")
        
        # Setup schema
        print("\nSetting up schema...")
        setup_schema(driver)
        
        # Clear existing data
        print("\nClearing existing graph...")
        clear_graph(driver)
        
        # Ingest entities
        print(f"\nIngesting {len(entities)} entities...")
        entity_count = ingest_entities(driver, entities)
        print(f"  Created {entity_count} entity nodes")
        
        # Ingest claims
        print(f"\nIngesting {len(claims)} claims as relationships...")
        rel_count, rel_skipped = ingest_claims(driver, claims)
        print(f"  Created {rel_count} relationships (skipped {rel_skipped})")
        
        # Get stats
        print("\nGraph statistics:")
        stats = get_graph_stats(driver)
        print(f"  Total nodes:         {stats['total_nodes']}")
        print(f"  Total relationships: {stats['total_relationships']}")
        print(f"  Isolated nodes:      {stats['isolated_nodes']}")
        
        print(f"\n  Node types:")
        for ntype, cnt in stats['node_types'].items():
            print(f"    {cnt:5d} {ntype}")
        
        print(f"\n  Relationship types:")
        for rtype, cnt in stats['rel_types'].items():
            print(f"    {cnt:5d} {rtype}")
        
        print(f"\n  Most connected entities:")
        for item in stats['top_connected']:
            print(f"    {item['degree']:3d} connections — {item['name']} ({item['type']})")
        
        print("\n" + "=" * 70)
        print("GRAPH BUILD COMPLETE")
        print("=" * 70)
        print(f"  Neo4j Browser: http://localhost:7474")
        print(f"  Bolt URI:      {config.NEO4J_URI}")
        
        # Save stats for later use
        stats_path = os.path.join(config.OUTPUT_DIR, "graph_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    finally:
        driver.close()


if __name__ == "__main__":
    run_graph_builder()
