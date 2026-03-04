# retrieval_api.py - fastapi REST layer over our neo4j knowledge graph
# endpoints: /health, /stats, /search, /entity/{name}, /neighbourhood/{name}, /query, /path
# also has internal query functions that the streamlit viz imports directly

import json
import os
import sys
from typing import Optional

# Add project root to path for shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from neo4j import GraphDatabase

import config

# -- app setup + neo4j driver (lazy init)
app = FastAPI(
    title="Layer10 Memory Graph API",
    version="1.0",
    description="Retrieval API for the Enron grounded long-term memory graph.",
)

_driver = None

def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
    return _driver


@app.on_event("shutdown")
def shutdown():
    global _driver
    if _driver:
        _driver.close()


# -- request/response models
class EntityResult(BaseModel):
    id: str
    canonical_name: str
    entity_type: str
    aliases: list[str] = []
    mention_count: int = 0
    degree: int = 0

class RelationshipResult(BaseModel):
    rel_type: str
    direction: str  # "outgoing" or "incoming"
    other_id: str
    other_name: str
    other_type: str
    confidence: float = 0.0
    mention_count: int = 0
    detail: str = ""

class EntityContext(BaseModel):
    entity: EntityResult
    relationships: list[RelationshipResult] = []

class QueryRequest(BaseModel):
    """structured query - filter by entity, claim type, confidence, etc"""
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    claim_type: Optional[str] = None
    min_confidence: float = 0.0
    limit: int = 20

class PathRequest(BaseModel):
    """find shortest path between two entities"""
    source: str
    target: str
    max_hops: int = 4


# -- internal query functions (also used by streamlit viz)
def search_entities(query: str, limit: int = 10) -> list[dict]:
    """fuzzy search by name or alias, returns top matches by mention count"""
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            WHERE toLower(e.canonical_name) CONTAINS toLower($q)
               OR any(a IN e.aliases WHERE toLower(a) CONTAINS toLower($q))
            RETURN e.id AS id, e.canonical_name AS name, e.entity_type AS type,
                   e.aliases AS aliases, e.mention_count AS mentions
            ORDER BY e.mention_count DESC
            LIMIT $limit
        """, q=query, limit=limit)
        return [dict(r) for r in result]


def get_entity_context(entity_id: str) -> dict | None:
    """fetches entity + all its outgoing/incoming relationships + evidence"""
    driver = get_driver()
    with driver.session() as session:
        # Get entity
        result = session.run("""
            MATCH (e:Entity {id: $eid})
            OPTIONAL MATCH (e)-[r]->(o:Entity)
            WITH e, collect({
                rel_type: type(r),
                direction: 'outgoing',
                other_id: o.id,
                other_name: o.canonical_name,
                other_type: o.entity_type,
                confidence: r.confidence,
                mention_count: r.mention_count,
                detail: r.detail,
                excerpts: r.excerpts
            }) AS outgoing
            OPTIONAL MATCH (e)<-[r2]-(i:Entity)
            WITH e, outgoing, collect({
                rel_type: type(r2),
                direction: 'incoming',
                other_id: i.id,
                other_name: i.canonical_name,
                other_type: i.entity_type,
                confidence: r2.confidence,
                mention_count: r2.mention_count,
                detail: r2.detail,
                excerpts: r2.excerpts
            }) AS incoming
            RETURN e.id AS id, e.canonical_name AS name, e.entity_type AS type,
                   e.aliases AS aliases, e.mention_count AS mentions,
                   outgoing, incoming
        """, eid=entity_id)
        
        record = result.single()
        if record is None:
            return None
        
        # Compute degree
        outgoing = [r for r in record['outgoing'] if r['other_id'] is not None]
        incoming = [r for r in record['incoming'] if r['other_id'] is not None]
        
        return {
            "entity": {
                "id": record['id'],
                "canonical_name": record['name'],
                "entity_type": record['type'],
                "aliases": record['aliases'] or [],
                "mention_count": record['mentions'],
                "degree": len(outgoing) + len(incoming),
            },
            "relationships": outgoing + incoming,
        }


def get_entity_by_name(name: str) -> dict | None:
    """find entity by exact/close name match then return full context.
    tries: exact name -> alias match -> contains match (shortest name wins)"""
    driver = get_driver()
    with driver.session() as session:
        # Try exact
        result = session.run("""
            MATCH (e:Entity)
            WHERE toLower(e.canonical_name) = toLower($name)
            RETURN e.id AS id
            LIMIT 1
        """, name=name)
        record = result.single()
        
        if record is None:
            # Try alias
            result = session.run("""
                MATCH (e:Entity)
                WHERE any(a IN e.aliases WHERE toLower(a) = toLower($name))
                RETURN e.id AS id
                LIMIT 1
            """, name=name)
            record = result.single()
        
        if record is None:
            # Try contains
            result = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.canonical_name) CONTAINS toLower($name)
                RETURN e.id AS id
                ORDER BY size(e.canonical_name) ASC
                LIMIT 1
            """, name=name)
            record = result.single()
        
        if record is None:
            return None
        
        return get_entity_context(record['id'])


def get_shortest_path(source_name: str, target_name: str, max_hops: int = 4) -> dict | None:
    """shortest path between two entities, up to max_hops"""
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (s:Entity), (t:Entity)
            WHERE toLower(s.canonical_name) CONTAINS toLower($src)
              AND toLower(t.canonical_name) CONTAINS toLower($tgt)
            WITH s, t
            ORDER BY size(s.canonical_name) + size(t.canonical_name) ASC
            LIMIT 1
            MATCH p = shortestPath((s)-[*1..%d]-(t))
            RETURN [n IN nodes(p) | {id: n.id, name: n.canonical_name, type: n.entity_type}] AS nodes,
                   [r IN relationships(p) | {type: type(r), confidence: r.confidence}] AS rels
        """ % max_hops, src=source_name, tgt=target_name)
        
        record = result.single()
        if record is None:
            return None
        
        return {
            "nodes": record['nodes'],
            "relationships": record['rels'],
            "hops": len(record['rels']),
        }


def get_claims_by_type(claim_type: str, min_confidence: float = 0.0, limit: int = 20) -> list[dict]:
    """get all claims of a given type, sorted by confidence"""
    rel_type = claim_type.upper()
    driver = get_driver()
    with driver.session() as session:
        result = session.run(f"""
            MATCH (s:Entity)-[r:{rel_type}]->(o:Entity)
            WHERE r.confidence >= $min_conf
            RETURN s.canonical_name AS subject, type(r) AS rel_type,
                   o.canonical_name AS object, r.confidence AS confidence,
                   r.mention_count AS mentions, r.detail AS detail,
                   r.excerpts AS evidence
            ORDER BY r.confidence DESC, r.mention_count DESC
            LIMIT $limit
        """, min_conf=min_confidence, limit=limit)
        return [dict(r) for r in result]


def get_neighbourhood(entity_name: str, depth: int = 2) -> dict:
    """k-hop neighbourhood subgraph for visualization. capped at 50 nodes."""
    driver = get_driver()
    with driver.session() as session:
        # Find base entity
        result = session.run("""
            MATCH (e:Entity)
            WHERE toLower(e.canonical_name) CONTAINS toLower($name)
            RETURN e.id AS id
            ORDER BY e.mention_count DESC
            LIMIT 1
        """, name=entity_name)
        record = result.single()
        if record is None:
            return {"nodes": [], "edges": []}
        
        eid = record['id']
        
        # Get k-hop subgraph
        result = session.run("""
            MATCH (center:Entity {id: $eid})
            OPTIONAL MATCH path = (center)-[*1..%d]-(n:Entity)
            WITH center, n, min(length(path)) AS dist
            ORDER BY dist ASC, n.mention_count DESC
            LIMIT 50
            WITH center, collect(n) AS neighbours
            WITH neighbours + [center] AS allNodes
            UNWIND allNodes AS node
            WITH DISTINCT node
            RETURN node.id AS id, node.canonical_name AS name,
                   node.entity_type AS type, node.mention_count AS mentions
        """ % min(depth, 3), eid=eid)
        
        nodes = [dict(r) for r in result]
        node_ids = {n['id'] for n in nodes}
        
        # Get edges between those nodes
        result = session.run("""
            MATCH (s:Entity)-[r]->(o:Entity)
            WHERE s.id IN $ids AND o.id IN $ids
            RETURN s.id AS source, o.id AS target, type(r) AS rel_type,
                   r.confidence AS confidence
        """, ids=list(node_ids))
        edges = [dict(r) for r in result]
        
        return {"nodes": nodes, "edges": edges}


def get_graph_stats() -> dict:
    """graph-level stats - tries cached file first, recomputes if missing"""
    stats_path = os.path.join(config.OUTPUT_DIR, "graph_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
    
    # Recompute
    driver = get_driver()
    with driver.session() as session:
        r = session.run("MATCH (n:Entity) RETURN count(n) AS cnt").single()
        node_count = r['cnt']
        r = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()
        rel_count = r['cnt']
    return {"total_nodes": node_count, "total_relationships": rel_count}


# -- fastapi endpoints
@app.get("/health")
def health():
    return {"status": "ok", "graph": config.NEO4J_URI}


@app.get("/stats")
def stats():
    return get_graph_stats()


@app.get("/search")
def search(q: str = Query(..., min_length=1), limit: int = 10):
    results = search_entities(q, limit=limit)
    if not results:
        raise HTTPException(404, f"No entities matching '{q}'")
    return {"query": q, "results": results}


@app.get("/entity/{name}")
def entity(name: str):
    ctx = get_entity_by_name(name)
    if ctx is None:
        raise HTTPException(404, f"Entity '{name}' not found")
    return ctx


@app.post("/query")
def structured_query(req: QueryRequest):
    """flexible query - pass entity_name and/or claim_type with filters"""
    if req.entity_name:
        ctx = get_entity_by_name(req.entity_name)
        if ctx is None:
            raise HTTPException(404, f"Entity '{req.entity_name}' not found")
        # Filter relationships
        rels = ctx['relationships']
        if req.claim_type:
            rels = [r for r in rels if r['rel_type'] == req.claim_type.upper()]
        if req.min_confidence:
            rels = [r for r in rels if (r.get('confidence') or 0) >= req.min_confidence]
        ctx['relationships'] = rels[:req.limit]
        return ctx
    
    if req.claim_type:
        claims = get_claims_by_type(req.claim_type, req.min_confidence, req.limit)
        return {"claim_type": req.claim_type, "results": claims}
    
    raise HTTPException(400, "Provide entity_name or claim_type")


@app.post("/path")
def shortest_path(req: PathRequest):
    result = get_shortest_path(req.source, req.target, req.max_hops)
    if result is None:
        raise HTTPException(404, f"No path found between '{req.source}' and '{req.target}'")
    return result


@app.get("/neighbourhood/{name}")
def neighbourhood(name: str, depth: int = 2):
    result = get_neighbourhood(name, depth=min(depth, 3))
    if not result['nodes']:
        raise HTTPException(404, f"Entity '{name}' not found")
    return result


# -- run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.retrieval_api:app", host="0.0.0.0", port=8000, reload=True)
