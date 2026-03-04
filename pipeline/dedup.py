# dedup.py - entity deduplication + claim consolidation
# takes the 670 per-email extraction jsons and merges them into
# one clean graph with canonical entities and deduplicated claims.
#
# three passes:
#   1) exact match (case insensitive, whitespace stripped)
#   2) alias absorption (if name A is listed as alias of B, merge)
#   3) fuzzy match (levenshtein >= 0.90 within same entity type)
#
# output goes to output/deduped_graph.json for neo4j ingestion

import json
import os
import re
import sys
import uuid
from collections import defaultdict
from datetime import datetime

# Add project root to path for shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Optional

from rapidfuzz import fuzz
from tqdm import tqdm

import config
from schema import SCHEMA_VERSION


# -- union-find for grouping entities that should be merged
class UnionFind:
    """disjoint set w/ path compression - used to track which entities belong together"""
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
    
    def groups(self):
        """returns dict of root -> list of all members in that group"""
        g = defaultdict(list)
        for x in self.parent:
            g[self.find(x)].append(x)
        return dict(g)


# -- name normalization helpers
def normalize_name(name: str) -> str:
    """lowercase, strip quotes/whitespace for comparison"""
    name = name.strip().lower()
    # Remove quotes
    name = name.strip('"\'')
    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name)
    return name


def is_email_address(name: str) -> bool:
    """quick check if a name is actually an email address"""
    return '@' in name and '.' in name


def name_key(name: str) -> str:
    """sort key for picking canonical names - prefer longer capitalized ones"""
    return name


# -- load all the per-email extraction jsons
def load_extractions() -> list:
    """reads every json file from the extractions/ folder"""
    files = sorted([
        f for f in os.listdir(config.EXTRACTIONS_DIR) 
        if f.endswith('.json')
    ])
    extractions = []
    for f in files:
        path = os.path.join(config.EXTRACTIONS_DIR, f)
        with open(path, 'r') as fp:
            extractions.append(json.load(fp))
    return extractions


# -- pass 1: exact name match (case insensitive)
def exact_dedup(raw_entities: list) -> tuple:
    """groups entities by normalized name, picks best canonical name by frequency.
    returns (entity_groups dict, name->group_id mapping)"""
    # Group by normalized name
    name_groups = defaultdict(list)
    for ent in raw_entities:
        key = normalize_name(ent['name'])
        if not key or len(key) < 2:
            continue
        name_groups[key].append(ent)
    
    entity_groups = {}
    name_to_group = {}
    
    for norm_name, mentions in name_groups.items():
        group_id = str(uuid.uuid4())
        
        # Pick canonical name: prefer the most common casing, longest form
        name_counts = defaultdict(int)
        for m in mentions:
            name_counts[m['name'].strip()] += 1
        canonical = max(name_counts, key=lambda n: (name_counts[n], len(n)))
        
        # Determine entity type by majority vote
        type_counts = defaultdict(int)
        for m in mentions:
            type_counts[m.get('entity_type', 'topic')] += 1
        entity_type = max(type_counts, key=type_counts.get)
        
        # Collect all aliases
        aliases = set()
        for m in mentions:
            aliases.add(m['name'].strip())
            for a in m.get('aliases', []):
                if a.strip():
                    aliases.add(a.strip())
        aliases.discard(canonical)
        
        # Collect source artifact_ids  
        sources = list(set(m.get('_artifact_id', '') for m in mentions if m.get('_artifact_id')))
        
        entity_groups[group_id] = {
            "id": group_id,
            "canonical_name": canonical,
            "entity_type": entity_type,
            "aliases": aliases,
            "mention_count": len(mentions),
            "sources": sources,
            "properties": {},
        }
        
        name_to_group[norm_name] = group_id
        # Also map aliases
        for a in aliases:
            na = normalize_name(a)
            if na and na not in name_to_group:
                name_to_group[na] = group_id
    
    return entity_groups, name_to_group


# -- pass 2: alias absorption
def alias_merge(entity_groups: dict, name_to_group: dict) -> tuple:
    """if entity A has alias X, and X is also its own entity B, merge B into A.
    only merges within same entity type to avoid false positives."""
    uf = UnionFind()
    
    # Initialize all groups
    for gid in entity_groups:
        uf.find(gid)
    
    merge_count = 0
    for gid, group in entity_groups.items():
        for alias in group['aliases']:
            alias_norm = normalize_name(alias)
            if alias_norm in name_to_group:
                other_gid = name_to_group[alias_norm]
                if other_gid != gid:
                    # Only merge if same entity type
                    if entity_groups.get(other_gid, {}).get('entity_type') == group['entity_type']:
                        uf.union(gid, other_gid)
                        merge_count += 1
    
    if merge_count == 0:
        return entity_groups, name_to_group
    
    # Rebuild groups based on union-find
    merged_groups = {}
    uf_groups = uf.groups()
    
    for root, members in uf_groups.items():
        # Merge all member groups into one
        all_aliases = set()
        all_sources = set()
        total_mentions = 0
        best_canonical = ""
        best_count = 0
        entity_type = None
        
        for member in members:
            if member not in entity_groups:
                continue
            g = entity_groups[member]
            all_aliases.update(g['aliases'])
            all_aliases.add(g['canonical_name'])
            all_sources.update(g['sources'])
            total_mentions += g['mention_count']
            
            if g['mention_count'] > best_count:
                best_count = g['mention_count']
                best_canonical = g['canonical_name']
                entity_type = g['entity_type']
        
        all_aliases.discard(best_canonical)
        
        new_gid = root
        merged_groups[new_gid] = {
            "id": new_gid,
            "canonical_name": best_canonical,
            "entity_type": entity_type or "topic",
            "aliases": all_aliases,
            "mention_count": total_mentions,
            "sources": list(all_sources),
            "properties": {},
        }
        
        # Update name→group mapping for all members
        for member in members:
            if member in entity_groups:
                g = entity_groups[member]
                name_to_group[normalize_name(g['canonical_name'])] = new_gid
                for a in g['aliases']:
                    name_to_group[normalize_name(a)] = new_gid
    
    print(f"  Alias absorption: merged {merge_count} entity pairs")
    return merged_groups, name_to_group


# -- pass 3: fuzzy matching (levenshtein within same entity type)
def fuzzy_merge(entity_groups: dict, name_to_group: dict, threshold: float = 0.90) -> tuple:
    """fuzzy match names within same type using token_sort_ratio.
    handles word order differences like 'John Smith' vs 'Smith, John'.
    also catches prefix matches like 'Chris Foster' vs 'Chris H Foster'."""
    uf = UnionFind()
    for gid in entity_groups:
        uf.find(gid)
    
    # Group entities by type for pairwise comparison
    type_groups = defaultdict(list)
    for gid, group in entity_groups.items():
        type_groups[group['entity_type']].append(gid)
    
    merge_count = 0
    
    for etype, gids in type_groups.items():
        if len(gids) < 2:
            continue
        
        # Sort by canonical name for efficient comparison
        gids_sorted = sorted(gids, key=lambda g: normalize_name(entity_groups[g]['canonical_name']))
        
        # Compare each pair — but use blocking to limit comparisons
        # Block by first 3 chars of normalized name
        blocks = defaultdict(list)
        for gid in gids_sorted:
            name = normalize_name(entity_groups[gid]['canonical_name'])
            if len(name) >= 3:
                blocks[name[:3]].append(gid)
        
        for block_gids in blocks.values():
            if len(block_gids) < 2:
                continue
            for i in range(len(block_gids)):
                for j in range(i + 1, len(block_gids)):
                    g1 = entity_groups[block_gids[i]]
                    g2 = entity_groups[block_gids[j]]
                    
                    n1 = normalize_name(g1['canonical_name'])
                    n2 = normalize_name(g2['canonical_name'])
                    
                    # Skip very short names (too many false positives)
                    if len(n1) < 4 or len(n2) < 4:
                        continue
                    
                    # Skip email addresses (different emails = different people)
                    if is_email_address(n1) or is_email_address(n2):
                        continue
                    
                    # Skip pure numbers
                    if n1.isdigit() or n2.isdigit():
                        continue
                    
                    # Compute fuzzy score
                    score = fuzz.token_sort_ratio(n1, n2) / 100.0
                    
                    if score >= threshold:
                        uf.union(block_gids[i], block_gids[j])
                        merge_count += 1
        
        # Also check: one name is a substring/prefix of the other 
        # e.g., "Chris Foster" vs "Chris H Foster"
        for i in range(len(gids_sorted)):
            for j in range(i + 1, min(i + 5, len(gids_sorted))):  # Only check nearby in sorted order
                g1 = entity_groups[gids_sorted[i]]
                g2 = entity_groups[gids_sorted[j]]
                n1 = normalize_name(g1['canonical_name'])
                n2 = normalize_name(g2['canonical_name'])
                
                if len(n1) < 5 or len(n2) < 5:
                    continue
                
                # Check if one is a strict prefix of the other with word boundary
                shorter, longer = (n1, n2) if len(n1) <= len(n2) else (n2, n1)
                if longer.startswith(shorter + ' ') or longer.startswith(shorter + '.'):
                    uf.union(gids_sorted[i], gids_sorted[j])
                    merge_count += 1
    
    if merge_count == 0:
        return entity_groups, name_to_group
    
    # Rebuild groups
    merged_groups = {}
    uf_groups = uf.groups()
    
    for root, members in uf_groups.items():
        all_aliases = set()
        all_sources = set()
        total_mentions = 0
        best_canonical = ""
        best_count = 0
        entity_type = None
        
        for member in members:
            if member not in entity_groups:
                continue
            g = entity_groups[member]
            all_aliases.update(g['aliases'])
            all_aliases.add(g['canonical_name'])
            all_sources.update(g['sources'])
            total_mentions += g['mention_count']
            
            if g['mention_count'] > best_count or (
                g['mention_count'] == best_count and 
                len(g['canonical_name']) > len(best_canonical)
            ):
                best_count = g['mention_count']
                best_canonical = g['canonical_name']
                entity_type = g['entity_type']
        
        all_aliases.discard(best_canonical)
        
        merged_groups[root] = {
            "id": root,
            "canonical_name": best_canonical,
            "entity_type": entity_type or "topic",
            "aliases": all_aliases,
            "mention_count": total_mentions,
            "sources": list(all_sources),
            "properties": {},
        }
        
        for member in members:
            if member in entity_groups:
                g = entity_groups[member]
                name_to_group[normalize_name(g['canonical_name'])] = root
                for a in g['aliases']:
                    name_to_group[normalize_name(a)] = root
    
    print(f"  Fuzzy merge: merged {merge_count} entity pairs")
    return merged_groups, name_to_group


# -- claim consolidation
def consolidate_claims(raw_claims: list, name_to_group: dict, entity_groups: dict) -> list:
    """maps claim subjects/objects to canonical entity ids, then merges
    claims with same (type, subject, object) and aggregates their evidence."""
    # Build claim key → merged claim
    claim_map = {}  # (claim_type, subject_group, object_group) → claim
    
    skipped = 0
    for c in raw_claims:
        subj_norm = normalize_name(c['subject'])
        obj_norm = normalize_name(c.get('object', ''))
        
        subj_gid = name_to_group.get(subj_norm)
        obj_gid = name_to_group.get(obj_norm)
        
        if not subj_gid:
            skipped += 1
            continue
        
        # For claims with no object, use subject as object
        if not obj_gid:
            if obj_norm:
                # Object entity wasn't extracted — skip this claim
                skipped += 1
                continue
            obj_gid = subj_gid
        
        claim_key = (c['claim_type'], subj_gid, obj_gid)
        
        if claim_key not in claim_map:
            subj_group = entity_groups.get(subj_gid, {})
            obj_group = entity_groups.get(obj_gid, {})
            
            claim_map[claim_key] = {
                "id": str(uuid.uuid4()),
                "claim_type": c['claim_type'],
                "subject_id": subj_gid,
                "subject_name": subj_group.get('canonical_name', c['subject']),
                "object_id": obj_gid,
                "object_name": obj_group.get('canonical_name', c.get('object', '')),
                "detail": c.get('detail', ''),
                "evidence": [],
                "confidence": 0.0,
                "mention_count": 0,
            }
        
        # Add evidence
        claim_map[claim_key]['evidence'].append({
            "excerpt": c.get('excerpt', ''),
            "confidence": c.get('confidence', 0.7),
            "artifact_id": c.get('_artifact_id', ''),
            "date": c.get('_date', ''),
        })
        claim_map[claim_key]['mention_count'] += 1
        
        # Aggregate confidence: max of all evidence
        conf = c.get('confidence', 0.7)
        if conf > claim_map[claim_key]['confidence']:
            claim_map[claim_key]['confidence'] = conf
    
    claims = list(claim_map.values())
    
    if skipped:
        print(f"  Claims skipped (unresolved entities): {skipped}")
    
    return claims


# -- main dedup pipeline
def run_dedup():
    """runs the whole thing: load extractions -> 3 pass dedup -> consolidate claims -> save"""
    
    print("=" * 70)
    print(f"DEDUPLICATION PIPELINE — Schema {SCHEMA_VERSION}")
    print("=" * 70)
    
    # 1. Load extractions
    print("\n[1/5] Loading extractions...")
    extractions = load_extractions()
    print(f"  Loaded {len(extractions)} extraction files")
    
    # 2. Flatten all entities and claims, tagging with source artifact
    print("\n[2/5] Flattening entities & claims...")
    raw_entities = []
    raw_claims = []
    
    for ext in extractions:
        artifact_id = ext.get('artifact_id', '')
        date = ext.get('date', '')
        
        for e in ext.get('entities', []):
            e['_artifact_id'] = artifact_id
            raw_entities.append(e)
        
        for c in ext.get('claims', []):
            c['_artifact_id'] = artifact_id
            c['_date'] = date
            raw_claims.append(c)
    
    print(f"  Raw entities: {len(raw_entities)}")
    print(f"  Raw claims:   {len(raw_claims)}")
    
    # 3. Deduplicate entities
    print("\n[3/5] Deduplicating entities...")
    
    print("  Pass 1: Exact name match...")
    entity_groups, name_to_group = exact_dedup(raw_entities)
    print(f"    {len(raw_entities)} mentions → {len(entity_groups)} unique entities")
    
    print("  Pass 2: Alias absorption...")
    entity_groups, name_to_group = alias_merge(entity_groups, name_to_group)
    print(f"    → {len(entity_groups)} entities after alias merge")
    
    print("  Pass 3: Fuzzy matching (≥90% similarity)...")
    entity_groups, name_to_group = fuzzy_merge(entity_groups, name_to_group, threshold=0.90)
    print(f"    → {len(entity_groups)} entities after fuzzy merge")
    
    # 4. Consolidate claims
    print("\n[4/5] Consolidating claims...")
    claims = consolidate_claims(raw_claims, name_to_group, entity_groups)
    print(f"  {len(raw_claims)} raw claims → {len(claims)} unique claims")
    
    # 5. Build and save the deduped graph
    print("\n[5/5] Saving deduped graph...")
    
    # Convert sets to lists for JSON serialization
    entities_out = []
    for gid, g in entity_groups.items():
        entities_out.append({
            "id": g['id'],
            "canonical_name": g['canonical_name'],
            "entity_type": g['entity_type'],
            "aliases": sorted(list(g['aliases'])),
            "mention_count": g['mention_count'],
            "sources": g['sources'],
            "properties": g['properties'],
        })
    
    # Sort entities by mention count (most important first)
    entities_out.sort(key=lambda e: -e['mention_count'])
    
    # Sort claims by confidence then mention count
    claims.sort(key=lambda c: (-c['confidence'], -c['mention_count']))
    
    graph = {
        "schema_version": SCHEMA_VERSION,
        "dedup_version": "v1.0",
        "created_at": datetime.utcnow().isoformat(),
        "stats": {
            "raw_entities": len(raw_entities),
            "raw_claims": len(raw_claims),
            "deduped_entities": len(entities_out),
            "deduped_claims": len(claims),
            "source_emails": len(extractions),
            "dedup_ratio_entities": f"{len(raw_entities)/max(len(entities_out),1):.1f}x",
            "dedup_ratio_claims": f"{len(raw_claims)/max(len(claims),1):.1f}x",
        },
        "entities": entities_out,
        "claims": claims,
    }
    
    output_path = os.path.join(config.OUTPUT_DIR, "deduped_graph.json")
    with open(output_path, 'w') as f:
        json.dump(graph, f, indent=2, default=str)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    
    # Final report
    print("\n" + "=" * 70)
    print("DEDUPLICATION COMPLETE")
    print("=" * 70)
    print(f"  Entities: {len(raw_entities)} → {len(entities_out)} ({len(raw_entities)/max(len(entities_out),1):.1f}x reduction)")
    print(f"  Claims:   {len(raw_claims)} → {len(claims)} ({len(raw_claims)/max(len(claims),1):.1f}x reduction)")
    print(f"  Output:   {output_path} ({file_size:.1f} MB)")
    
    # Show top entities
    print(f"\n  Top 15 entities by mention count:")
    for e in entities_out[:15]:
        alias_str = f" (aka: {', '.join(list(e['aliases'])[:3])})" if e['aliases'] else ""
        print(f"    {e['mention_count']:3d}x [{e['entity_type']:12s}] {e['canonical_name']}{alias_str}")
    
    # Show claim type distribution
    from collections import Counter
    claim_types = Counter(c['claim_type'] for c in claims)
    print(f"\n  Claim type distribution:")
    for ct, count in claim_types.most_common():
        print(f"    {count:4d} {ct}")
    
    return graph


if __name__ == "__main__":
    run_dedup()
