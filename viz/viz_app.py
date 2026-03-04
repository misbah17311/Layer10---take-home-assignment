# viz_app.py - streamlit interactive graph explorer
# run with: streamlit run viz/viz_app.py

import json
import os
import sys

# Add project root to path for shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from neo4j import GraphDatabase

import config

# -- page setup
st.set_page_config(
    page_title="Layer10 Memory Graph Explorer",
    page_icon="🧠",
    layout="wide",
)

# -- neo4j connection (streamlit caches this)
@st.cache_resource
def get_driver():
    return GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )


# -- colors for node/edge types
TYPE_COLORS = {
    "person": "#4FC3F7",       # light blue
    "organization": "#FF8A65", # orange
    "project": "#81C784",      # green
    "topic": "#CE93D8",        # purple
    "meeting": "#FFD54F",      # yellow
}

REL_COLORS = {
    "WORKS_AT": "#90A4AE",
    "REPORTS_TO": "#F44336",
    "WORKS_ON": "#4CAF50",
    "PART_OF": "#FF9800",
    "DISCUSSED_IN": "#9C27B0",
    "DECIDED": "#E91E63",
    "COMMUNICATED_WITH": "#2196F3",
    "ATTENDED": "#FFC107",
    "ASSIGNED_TO": "#009688",
    "MENTIONED_IN": "#607D8B",
    "CONTRADICTS": "#F44336",
    "SUPERSEDES": "#795548",
}

# -- query functions that talk to neo4j
def search_entities(query: str, limit: int = 15):
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            WHERE toLower(e.canonical_name) CONTAINS toLower($q)
               OR any(a IN e.aliases WHERE toLower(a) CONTAINS toLower($q))
            RETURN e.id AS id, e.canonical_name AS name, e.entity_type AS type,
                   e.mention_count AS mentions
            ORDER BY e.mention_count DESC
            LIMIT $limit
        """, q=query, limit=limit)
        return [dict(r) for r in result]


def get_neighbourhood(entity_id: str, depth: int = 1, max_nodes: int = 50):
    """Get subgraph around an entity."""
    driver = get_driver()
    with driver.session() as session:
        # Get nodes
        result = session.run("""
            MATCH (center:Entity {id: $eid})
            OPTIONAL MATCH path = (center)-[*1..%d]-(n:Entity)
            WITH center, n, min(length(path)) AS dist
            ORDER BY dist ASC, n.mention_count DESC
            LIMIT $max_nodes
            WITH center, collect(n) AS neighbours
            WITH neighbours + [center] AS allNodes
            UNWIND allNodes AS node
            WITH DISTINCT node
            RETURN node.id AS id, node.canonical_name AS name,
                   node.entity_type AS type, node.mention_count AS mentions
        """ % min(depth, 3), eid=entity_id, max_nodes=max_nodes)
        nodes = [dict(r) for r in result]
        node_ids = {n['id'] for n in nodes}

        # Get edges between those nodes
        result = session.run("""
            MATCH (s:Entity)-[r]->(o:Entity)
            WHERE s.id IN $ids AND o.id IN $ids
            RETURN s.id AS source, o.id AS target, type(r) AS rel_type,
                   r.confidence AS confidence, r.detail AS detail
        """, ids=list(node_ids))
        edges = [dict(r) for r in result]

    return nodes, edges


def get_entity_detail(entity_id: str):
    """Get full entity info + relationships."""
    driver = get_driver()
    with driver.session() as session:
        # Entity info
        result = session.run("""
            MATCH (e:Entity {id: $eid})
            RETURN e.canonical_name AS name, e.entity_type AS type,
                   e.aliases AS aliases, e.mention_count AS mentions
        """, eid=entity_id)
        info = dict(result.single())

        # Outgoing
        result = session.run("""
            MATCH (e:Entity {id: $eid})-[r]->(o:Entity)
            RETURN type(r) AS rel_type, o.canonical_name AS target,
                   o.entity_type AS target_type, r.confidence AS confidence,
                   r.detail AS detail, r.mention_count AS mentions,
                   r.excerpts AS evidence
            ORDER BY r.confidence DESC
        """, eid=entity_id)
        outgoing = [dict(r) for r in result]

        # Incoming
        result = session.run("""
            MATCH (e:Entity {id: $eid})<-[r]-(i:Entity)
            RETURN type(r) AS rel_type, i.canonical_name AS source,
                   i.entity_type AS source_type, r.confidence AS confidence,
                   r.detail AS detail, r.mention_count AS mentions,
                   r.excerpts AS evidence
            ORDER BY r.confidence DESC
        """, eid=entity_id)
        incoming = [dict(r) for r in result]

    return info, outgoing, incoming


def get_graph_overview():
    """Get high-level stats."""
    driver = get_driver()
    with driver.session() as session:
        r1 = session.run("MATCH (n:Entity) RETURN count(n) AS cnt").single()['cnt']
        r2 = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()['cnt']
        r3 = session.run("""
            MATCH (n:Entity)
            RETURN n.entity_type AS type, count(n) AS cnt
            ORDER BY cnt DESC
        """)
        type_counts = {r['type']: r['cnt'] for r in r3}
        r4 = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS cnt
            ORDER BY cnt DESC
        """)
        rel_counts = {r['type']: r['cnt'] for r in r4}
        r5 = session.run("""
            MATCH (n:Entity)-[r]-()
            RETURN n.canonical_name AS name, n.entity_type AS type,
                   count(r) AS degree
            ORDER BY degree DESC LIMIT 10
        """)
        top_entities = [dict(r) for r in r5]

    return {
        "nodes": r1, "edges": r2,
        "type_counts": type_counts, "rel_counts": rel_counts,
        "top_entities": top_entities
    }


def find_path(src_name: str, tgt_name: str, max_hops: int = 4):
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (s:Entity), (t:Entity)
            WHERE toLower(s.canonical_name) CONTAINS toLower($src)
              AND toLower(t.canonical_name) CONTAINS toLower($tgt)
            WITH s, t ORDER BY size(s.canonical_name) + size(t.canonical_name) LIMIT 1
            MATCH p = shortestPath((s)-[*1..%d]-(t))
            RETURN [n IN nodes(p) | {id: n.id, name: n.canonical_name, type: n.entity_type}] AS nodes,
                   [r IN relationships(p) | {type: type(r), confidence: r.confidence, src: startNode(r).id, tgt: endNode(r).id}] AS rels
        """ % max_hops, src=src_name, tgt=tgt_name)
        record = result.single()
        if record is None:
            return None, None
        return record['nodes'], record['rels']


# -- renders the interactive graph component
def render_graph(nodes_data, edges_data, height=500):
    """builds agraph nodes/edges from raw data and displays them"""
    if not nodes_data:
        st.info("No data to display.")
        return

    ag_nodes = []
    for n in nodes_data:
        size = max(10, min(40, (n.get('mentions', 1) or 1)))
        ag_nodes.append(Node(
            id=n['id'],
            label=n['name'][:25],
            title=f"{n['name']} ({n['type']})\nMentions: {n.get('mentions', 0)}",
            size=size,
            color=TYPE_COLORS.get(n['type'], '#BDBDBD'),
        ))
    
    ag_edges = []
    for e in edges_data:
        ag_edges.append(Edge(
            source=e['source'] if 'source' in e else e.get('src', ''),
            target=e['target'] if 'target' in e else e.get('tgt', ''),
            label=e.get('rel_type', e.get('type', '')),
            color=REL_COLORS.get(e.get('rel_type', e.get('type', '')), '#999'),
            width=max(1, (e.get('confidence', 0.5) or 0.5) * 3),
        ))
    
    agraph_config = Config(
        width="100%",
        height=height,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={'labelProperty': 'label'},
        link={'labelProperty': 'label', 'renderLabel': True},
    )
    
    return agraph(nodes=ag_nodes, edges=ag_edges, config=agraph_config)


# -- sidebar nav
st.sidebar.title("🧠 Memory Graph")
st.sidebar.markdown("**Layer10 Take-Home**")
st.sidebar.markdown("Enron Email Knowledge Graph")

page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🔍 Entity Search",
    "🕸️ Graph Explorer",
    "🛤️ Path Finder",
])

# -- pages

# --- OVERVIEW ---
if page == "📊 Overview":
    st.title("📊 Graph Overview")
    overview = get_graph_overview()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Entities", overview['nodes'])
    col2.metric("Total Relationships", overview['edges'])
    col3.metric("Entity Types", len(overview['type_counts']))
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Entity Types")
        for etype, cnt in overview['type_counts'].items():
            color = TYPE_COLORS.get(etype, '#999')
            st.markdown(f"<span style='color:{color}'>●</span> **{etype}**: {cnt}", unsafe_allow_html=True)
    
    with col_b:
        st.subheader("Relationship Types")
        for rtype, cnt in overview['rel_counts'].items():
            color = REL_COLORS.get(rtype, '#999')
            st.markdown(f"<span style='color:{color}'>●</span> **{rtype}**: {cnt}", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Top 10 Most Connected Entities")
    for i, ent in enumerate(overview['top_entities'], 1):
        st.markdown(f"{i}. **{ent['name']}** ({ent['type']}) — {ent['degree']} connections")


# --- ENTITY SEARCH ---
elif page == "🔍 Entity Search":
    st.title("🔍 Entity Search")
    
    query = st.text_input("Search entities by name or alias", placeholder="e.g., Tim Belden, Enron, TruOrange")
    
    if query:
        results = search_entities(query)
        if not results:
            st.warning(f"No entities found matching '{query}'")
        else:
            st.success(f"Found {len(results)} entities")
            
            for r in results:
                with st.expander(f"**{r['name']}** ({r['type']}) — {r['mentions']} mentions"):
                    info, outgoing, incoming = get_entity_detail(r['id'])
                    
                    if info.get('aliases'):
                        st.markdown(f"**Aliases**: {', '.join(info['aliases'][:10])}")
                    
                    if outgoing:
                        st.markdown("**Outgoing relationships:**")
                        for rel in outgoing[:15]:
                            conf = f"{rel['confidence']:.0%}" if rel['confidence'] else "?"
                            detail = f" — {rel['detail']}" if rel['detail'] else ""
                            st.markdown(f"- `{rel['rel_type']}` → **{rel['target']}** ({rel['target_type']}) [{conf}]{detail}")
                            if rel.get('evidence'):
                                for excerpt in rel['evidence'][:2]:
                                    if excerpt:
                                        st.caption(f"📧 _{excerpt[:150]}{'...' if len(excerpt) > 150 else ''}_")
                    
                    if incoming:
                        st.markdown("**Incoming relationships:**")
                        for rel in incoming[:15]:
                            conf = f"{rel['confidence']:.0%}" if rel['confidence'] else "?"
                            detail = f" — {rel['detail']}" if rel['detail'] else ""
                            st.markdown(f"- **{rel['source']}** ({rel['source_type']}) `{rel['rel_type']}` → [{conf}]{detail}")
                            if rel.get('evidence'):
                                for excerpt in rel['evidence'][:2]:
                                    if excerpt:
                                        st.caption(f"📧 _{excerpt[:150]}{'...' if len(excerpt) > 150 else ''}_")


# --- GRAPH EXPLORER ---
elif page == "🕸️ Graph Explorer":
    st.title("🕸️ Graph Explorer")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        entity_query = st.text_input("Center entity", placeholder="e.g., Tim Belden")
    with col2:
        depth = st.selectbox("Depth", [1, 2, 3], index=0)
    with col3:
        max_nodes = st.slider("Max nodes", 10, 100, 40)
    
    if entity_query:
        results = search_entities(entity_query, limit=1)
        if not results:
            st.warning(f"Entity '{entity_query}' not found")
        else:
            entity = results[0]
            st.info(f"Showing neighbourhood of **{entity['name']}** ({entity['type']}, {entity['mentions']} mentions)")
            
            nodes_data, edges_data = get_neighbourhood(entity['id'], depth=depth, max_nodes=max_nodes)
            
            st.markdown(f"**{len(nodes_data)} nodes, {len(edges_data)} edges**")
            
            # Legend
            legend_cols = st.columns(5)
            for i, (etype, color) in enumerate(TYPE_COLORS.items()):
                legend_cols[i].markdown(f"<span style='background:{color};padding:2px 8px;border-radius:4px;color:black'>{etype}</span>", unsafe_allow_html=True)
            
            render_graph(nodes_data, edges_data, height=550)


# --- PATH FINDER ---
elif page == "🛤️ Path Finder":
    st.title("🛤️ Path Finder")
    st.markdown("Find the shortest path between two entities in the knowledge graph.")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        src = st.text_input("Source entity", placeholder="e.g., Tim Belden")
    with col2:
        tgt = st.text_input("Target entity", placeholder="e.g., Enron")
    with col3:
        max_hops = st.selectbox("Max hops", [2, 3, 4, 5], index=1)
    
    if src and tgt:
        path_nodes, path_rels = find_path(src, tgt, max_hops)
        
        if path_nodes is None:
            st.warning(f"No path found between '{src}' and '{tgt}' within {max_hops} hops.")
        else:
            st.success(f"Path found: {len(path_rels)} hop(s)")
            
            # Show path as text
            path_str = ""
            for i, node in enumerate(path_nodes):
                path_str += f"**{node['name']}** ({node['type']})"
                if i < len(path_rels):
                    rel = path_rels[i]
                    path_str += f" —[`{rel['type']}`]→ "
            st.markdown(path_str)
            
            # Render as graph
            edges_for_graph = []
            for r in path_rels:
                edges_for_graph.append({
                    'source': r['src'],
                    'target': r['tgt'],
                    'rel_type': r['type'],
                    'confidence': r.get('confidence', 0.8),
                })
            
            render_graph(path_nodes, edges_for_graph, height=300)

# -- footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built for Layer10 Take-Home Assignment")
st.sidebar.markdown("Enron Email Dataset (2000 emails)")
