#!/usr/bin/env python3
# runs the full pipeline: extraction -> dedup -> graph build
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

steps = [
    ("Step 1/3: Extraction", [sys.executable, "-m", "pipeline.extraction"]),
    ("Step 2/3: Deduplication", [sys.executable, "-m", "pipeline.dedup"]),
    ("Step 3/3: Graph Build (Neo4j)", [sys.executable, "-m", "pipeline.graph_builder"]),
]

for label, cmd in steps:
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

print(f"\n{'='*60}\nPipeline complete!\n{'='*60}")
