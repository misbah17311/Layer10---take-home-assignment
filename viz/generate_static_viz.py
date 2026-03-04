# generate_static_viz.py - creates standalone html graph visualizations using pyvis
# these are the ones we embed in the writeup / screenshots

import json
import os
import sys

# Add project root to path for shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyvis.network import Network
from neo4j import GraphDatabase
import config

TYPE_COLORS = {
    "person": "#4FC3F7",
    "organization": "#FF8A65",
    "project": "#81C784",
    "topic": "#CE93D8",
    "meeting": "#FFD54F",
}


def generate_full_graph_viz():
    """overview viz with top 60 most-connected entities"""
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    
    net = Network(height="700px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)
    
    with driver.session() as session:
        # Get top 60 most-connected entities
        result = session.run("""
            MATCH (n:Entity)-[r]-()
            WITH n, count(r) AS degree
            ORDER BY degree DESC LIMIT 60
            RETURN n.id AS id, n.canonical_name AS name, n.entity_type AS type,
                   n.mention_count AS mentions, degree
        """)
        nodes = {r['id']: dict(r) for r in result}
        
        # Get edges between them
        result = session.run("""
            MATCH (s:Entity)-[r]->(o:Entity)
            WHERE s.id IN $ids AND o.id IN $ids
            RETURN s.id AS src, o.id AS tgt, type(r) AS rel_type,
                   r.confidence AS confidence
        """, ids=list(nodes.keys()))
        edges = [dict(r) for r in result]
    
    for nid, n in nodes.items():
        size = max(8, min(50, n.get('degree', 1) * 2))
        net.add_node(
            nid,
            label=n['name'][:20],
            title=f"{n['name']}\nType: {n['type']}\nMentions: {n['mentions']}\nConnections: {n['degree']}",
            color=TYPE_COLORS.get(n['type'], '#BDBDBD'),
            size=size,
        )
    
    for e in edges:
        net.add_edge(
            e['src'], e['tgt'],
            title=e['rel_type'],
            color='#555555',
            width=max(0.5, (e.get('confidence') or 0.5) * 2),
        )
    
    viz_dir = os.path.join(config.OUTPUT_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    output_path = os.path.join(viz_dir, "graph_overview.html")
    net.save_graph(output_path)
    print(f"  Overview graph: {output_path} ({len(nodes)} nodes, {len(edges)} edges)")
    
    driver.close()
    return output_path


def generate_entity_spotlight(entity_name: str, depth: int = 1):
    """zoomed-in viz centered on one entity + its neighbours"""
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    
    net = Network(height="600px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")
    net.barnes_hut(gravity=-2000, central_gravity=0.5, spring_length=120)
    
    with driver.session() as session:
        # Find entity
        result = session.run("""
            MATCH (e:Entity)
            WHERE toLower(e.canonical_name) CONTAINS toLower($name)
            RETURN e.id AS id
            ORDER BY e.mention_count DESC LIMIT 1
        """, name=entity_name)
        record = result.single()
        if not record:
            print(f"  Entity '{entity_name}' not found")
            driver.close()
            return None
        
        center_id = record['id']
        
        # Get neighbourhood
        result = session.run("""
            MATCH (center:Entity {id: $eid})
            OPTIONAL MATCH path = (center)-[*1..%d]-(n:Entity)
            WITH center, n, min(length(path)) AS dist
            ORDER BY dist ASC, n.mention_count DESC LIMIT 40
            WITH center, collect(n) AS neighbours
            WITH neighbours + [center] AS allNodes
            UNWIND allNodes AS node
            WITH DISTINCT node
            RETURN node.id AS id, node.canonical_name AS name,
                   node.entity_type AS type, node.mention_count AS mentions
        """ % depth, eid=center_id)
        nodes = {r['id']: dict(r) for r in result}
        
        # Get edges
        result = session.run("""
            MATCH (s:Entity)-[r]->(o:Entity)
            WHERE s.id IN $ids AND o.id IN $ids
            RETURN s.id AS src, o.id AS tgt, type(r) AS rel_type,
                   r.confidence AS confidence
        """, ids=list(nodes.keys()))
        edges = [dict(r) for r in result]
    
    for nid, n in nodes.items():
        size = 30 if nid == center_id else max(8, min(25, (n.get('mentions') or 1)))
        border = 3 if nid == center_id else 1
        net.add_node(
            nid,
            label=n['name'][:20],
            title=f"{n['name']}\nType: {n['type']}\nMentions: {n['mentions']}",
            color=TYPE_COLORS.get(n['type'], '#BDBDBD'),
            size=size,
            borderWidth=border,
        )
    
    for e in edges:
        net.add_edge(
            e['src'], e['tgt'],
            title=e['rel_type'],
            label=e['rel_type'],
            color='#555555',
            width=max(0.5, (e.get('confidence') or 0.5) * 2),
            font={'size': 8, 'color': '#888'},
        )
    
    safe_name = entity_name.lower().replace(' ', '_')
    viz_dir = os.path.join(config.OUTPUT_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    output_path = os.path.join(viz_dir, f"graph_{safe_name}.html")
    net.save_graph(output_path)
    print(f"  Spotlight [{entity_name}]: {output_path} ({len(nodes)} nodes, {len(edges)} edges)")
    
    driver.close()
    return output_path


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING STATIC VISUALIZATIONS")
    print("=" * 70)
    
    print("\n1. Full graph overview (top 60 entities)...")
    generate_full_graph_viz()
    
    spotlights = ["Tim Belden", "Enron", "Kate Symes", "Mark Guzman"]
    for i, name in enumerate(spotlights, 2):
        print(f"\n{i}. Spotlight: {name}...")
        generate_entity_spotlight(name, depth=1)
    
    print("\n" + "=" * 70)
    print("DONE — check output/ for HTML files")
    print("=" * 70)
