# extraction.py - the main LLM extraction loop
# reads emails from our subset csv, sends each one to the LLM,
# and gets back structured entities + claims with evidence.
#
# main ideas:
# - every claim has to be grounded in an actual email excerpt
# - pydantic validates the outputs so bad json gets caught
# - exponential backoff on failures, checkpoints so we can resume
# - parallel mode for cloud backends, sequential for local ollama

import json
import time
import hashlib
import os
import re
import signal
import sys

# Add project root to path for shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import email as email_lib
from email import policy
from email.utils import parsedate_to_datetime
from datetime import datetime
from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests
import pandas as pd
from tqdm import tqdm

# -- graceful shutdown: catch ctrl+c so we can save progress
_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        print("\n[FORCE QUIT] Exiting immediately.")
        sys.exit(1)
    _shutdown_requested = True
    print("\n[SHUTDOWN] Finishing current email then saving progress...")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

from schema import (
    ExtractionResult, ExtractedEntity, ExtractedClaim,
    Artifact, ENTITY_TYPES, CLAIM_TYPES, SCHEMA_VERSION
)
import config


# -- the system prompt that goes to the LLM
# (don't touch this unless you want to change what gets extracted)
SYSTEM_PROMPT = """You are a knowledge-graph extraction engine for corporate emails. Your job is to identify entities and factual claims, grounding each claim in exact text from the email.

## Entity Types
- person: named individuals (use "Firstname Lastname" when possible)
- organization: companies, departments, teams, divisions
- project: named projects, deals, initiatives, codenames
- topic: recurring discussion subjects (e.g., "gas pricing", "FERC regulation")
- meeting: named meetings, calls, conferences with approximate dates

## Claim Types
- works_at: person → organization
- reports_to: person → person (manager/supervisor)
- works_on: person → project
- part_of: entity → entity (team membership, subsidiary, sub-project)
- discussed_in: topic → meeting/email
- decided: person/org made a decision (detail = what was decided)
- communicated_with: person → person (based on email headers and body)
- attended: person → meeting
- assigned_to: task/action → person
- mentioned_in: entity → email (entity was referenced but role unclear)

## Rules
1. Extract ALL named people, organizations, projects, and topics — don't skip anyone in To/CC/From
2. Derive the sender's name from the "FROM" field or email signature
3. Every claim MUST have "excerpt": a SHORT exact quote (≤100 chars) from the email that supports it
4. Confidence scoring: 1.0 = explicitly stated, 0.8 = strongly implied, 0.6 = weakly implied
5. Use "communicated_with" for sender→recipient relationships evident from headers
6. If someone is assigned an action item, use "assigned_to"
7. Maximum 15 entities, 20 claims per email
8. Return ONLY valid JSON — no markdown, no explanation, no commentary

## Output Schema
{"entities":[{"name":"...","entity_type":"person|organization|project|topic|meeting","aliases":[],"properties":{}}],"claims":[{"claim_type":"...","subject":"entity name","object":"entity name","detail":"","excerpt":"exact quote from email","confidence":0.9}]}"""


def build_user_prompt(artifact: Artifact) -> str:
    """formats one email into the user prompt we send to the LLM"""
    return f"""Extract entities and claims from this email:

FROM: {artifact.sender} ({artifact.sender_name or 'unknown'})
TO: {', '.join(artifact.recipients[:10])}
CC: {', '.join(artifact.cc[:5])}
SUBJECT: {artifact.subject}
DATE: {artifact.date}
IS_FORWARD: {artifact.is_forward}
IS_REPLY: {artifact.is_reply}

--- EMAIL BODY ---
{artifact.body[:2000]}
--- END ---

Return ONLY valid JSON matching the schema above."""


# -- email parser: raw message string -> our Artifact model
def parse_email(file_path: str, raw_message: str) -> Optional[Artifact]:
    """takes raw email text and turns it into a structured Artifact"""
    try:
        msg = email_lib.message_from_string(raw_message, policy=policy.default)
        
        # Extract fields
        sender = (msg.get('From', '') or '').strip().lower()
        sender_name = msg.get('X-From', '') or ''
        to_raw = (msg.get('To', '') or '')
        cc_raw = (msg.get('Cc', '') or '')
        subject = msg.get('Subject', '') or ''
        message_id = msg.get('Message-ID', '') or ''
        
        # Parse recipients
        recipients = [r.strip().lower() for r in to_raw.split(',') if r.strip()]
        cc = [r.strip().lower() for r in cc_raw.split(',') if r.strip()]
        
        # Get body
        body = msg.get_payload()
        if not isinstance(body, str):
            body = str(body) if body else ''
        
        # Parse date
        date = None
        try:
            date_str = msg.get('Date', '')
            if date_str:
                date = parsedate_to_datetime(date_str)
        except:
            pass
        
        # Detect forward/reply
        is_forward = bool(
            'fw:' in subject.lower() or 
            'fwd:' in subject.lower() or
            '--- forwarded' in body.lower()[:500] if body else False
        )
        is_reply = subject.lower().startswith('re:')
        
        # Body hash for dedup
        body_hash = hashlib.sha256(body.encode('utf-8', errors='replace')).hexdigest() if body else ''
        
        return Artifact(
            message_id=message_id,
            file_path=file_path,
            sender=sender,
            sender_name=sender_name,
            recipients=recipients,
            cc=cc,
            subject=subject,
            date=date,
            body=body,
            body_hash=body_hash,
            is_forward=is_forward,
            is_reply=is_reply,
        )
    except Exception as e:
        print(f"  [WARN] Failed to parse email {file_path}: {e}")
        return None


# -- LLM backends
# we tried a bunch: ollama (slow), groq (rate limited), openrouter (winner)
_groq_client = None

def get_groq_client():
    """lazy init so we don't crash if groq isn't installed"""
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def call_groq(system_prompt: str, user_prompt: str) -> str:
    """groq cloud api - llama 3.1 8b on their hardware, fast but rate limited"""
    client = get_groq_client()
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=8192,
        response_format={"type": "json_object"},  # Force JSON output
    )
    return response.choices[0].message.content


def call_ollama(system_prompt: str, user_prompt: str) -> str:
    """local ollama - works but painfully slow on 4gb vram"""
    url = f"{config.OLLAMA_URL}/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 2048,
            "num_ctx": 4096,       # Limit context window for speed
        },
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def call_openrouter(system_prompt: str, user_prompt: str) -> str:
    """openrouter api - openai compatible, pay per token, what we ended up using"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/layer10-takehome",
        "X-Title": "Layer10 Memory Extraction",
    }
    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """routes to whichever backend is set in config"""
    if config.LLM_BACKEND == "openrouter":
        return call_openrouter(system_prompt, user_prompt)
    elif config.LLM_BACKEND == "groq":
        return call_groq(system_prompt, user_prompt)
    else:
        return call_ollama(system_prompt, user_prompt)


def extract_from_email(artifact: Artifact, retry_count: int = 0) -> Optional[ExtractionResult]:
    """send one email to the LLM, parse out entities + claims. retries on failure."""
    
    user_prompt = build_user_prompt(artifact)
    
    try:
        raw_text = call_llm(SYSTEM_PROMPT, user_prompt)
        raw_text = raw_text.strip()
        
        # Clean up markdown code blocks if present
        if raw_text.startswith('```'):
            lines = raw_text.split('\n')
            lines = [l for l in lines if not l.strip().startswith('```')]
            raw_text = '\n'.join(lines)
        
        # Try to extract JSON from the response if it contains extra text
        if not raw_text.startswith('{'):
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                raw_text = match.group(0)
        
        # Parse JSON
        data = json.loads(raw_text)
        
        # Validate with Pydantic — skip individual bad items instead of failing all
        entities = []
        for e in data.get('entities', []):
            try:
                entities.append(ExtractedEntity(**e))
            except Exception:
                pass  # Skip malformed entity
        
        claims = []
        for c in data.get('claims', []):
            try:
                # Coerce null fields to empty strings
                if c.get('object') is None:
                    c['object'] = c.get('subject', '')
                if c.get('excerpt') is None:
                    c['excerpt'] = ''
                if c.get('confidence') is None:
                    c['confidence'] = 0.7
                claims.append(ExtractedClaim(**c))
            except Exception:
                pass  # Skip malformed claim
        
        result = ExtractionResult(entities=entities, claims=claims)
        
        # Filter claims below confidence threshold
        result.claims = [c for c in result.claims if c.confidence >= config.CONFIDENCE_THRESHOLD]
        
        # Validate claim types
        valid_claim_types = set(CLAIM_TYPES)
        result.claims = [c for c in result.claims if c.claim_type in valid_claim_types]
        
        # Validate entity types
        valid_entity_types = set(ENTITY_TYPES)
        result.entities = [e for e in result.entities if e.entity_type in valid_entity_types]
        
        # Retry if LLM returned empty for a non-trivial email (transient failure)
        body_len = len(artifact.body) if artifact.body else 0
        if len(result.entities) == 0 and len(result.claims) == 0 and body_len > 100:
            if retry_count < config.MAX_RETRIES:
                print(f"  [RETRY {retry_count+1}] Empty extraction for non-trivial email, retrying...")
                time.sleep(2 ** retry_count)
                return extract_from_email(artifact, retry_count + 1)
            print(f"  [WARN] Empty extraction after {config.MAX_RETRIES} retries: {artifact.subject[:50]}")
        
        return result
        
    except json.JSONDecodeError as e:
        if retry_count < config.MAX_RETRIES:
            print(f"  [RETRY {retry_count+1}] JSON parse error, retrying...")
            time.sleep(2 ** retry_count)
            return extract_from_email(artifact, retry_count + 1)
        print(f"  [FAIL] JSON parse failed after {config.MAX_RETRIES} retries: {e}")
        return None
        
    except requests.exceptions.ConnectionError:
        if retry_count < config.MAX_RETRIES:
            print(f"  [RETRY {retry_count+1}] Connection error, waiting 5s...")
            time.sleep(5)
            return extract_from_email(artifact, retry_count + 1)
        print(f"  [FAIL] Connection failed")
        return None
        
    except Exception as e:
        error_msg = str(e).lower()
        if retry_count < config.MAX_RETRIES:
            # Handle rate limits (Groq: 429)
            if 'rate_limit' in error_msg or '429' in error_msg or 'rate limit' in error_msg:
                wait_time = 10 * (retry_count + 1)
                print(f"  [RATE LIMIT] Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  [RETRY {retry_count+1}] Error: {e}")
                time.sleep(2 ** retry_count)
            return extract_from_email(artifact, retry_count + 1)
        print(f"  [FAIL] Extraction failed: {e}")
        return None


# -- checkpoint management
# saves progress to disk so we can resume if interrupted
def get_checkpoint_path() -> str:
    return os.path.join(config.OUTPUT_DIR, "extraction_checkpoint.json")


def load_checkpoint() -> dict:
    """load which emails we already processed (so we can skip them)"""
    path = get_checkpoint_path()
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            # Ensure all expected keys exist
            data.setdefault("processed_ids", [])
            data.setdefault("failed_ids", [])
            data.setdefault("stats", {"success": 0, "failed": 0, "total_entities": 0, "total_claims": 0})
            data.setdefault("version", SCHEMA_VERSION)
            return data
        except (json.JSONDecodeError, IOError):
            # Corrupted checkpoint — try backup
            backup = path + ".bak"
            if os.path.exists(backup):
                with open(backup, 'r') as f:
                    return json.load(f)
    return {
        "processed_ids": [], "failed_ids": [],
        "stats": {"success": 0, "failed": 0, "total_entities": 0, "total_claims": 0},
        "version": SCHEMA_VERSION
    }


def save_checkpoint(checkpoint: dict):
    """write checkpoint to disk atomically (tmp file + rename so we don't corrupt on crash)"""
    path = get_checkpoint_path()
    tmp_path = path + ".tmp"
    bak_path = path + ".bak"
    
    with open(tmp_path, 'w') as f:
        json.dump(checkpoint, f)
    
    # Keep one backup of the previous checkpoint
    if os.path.exists(path):
        try:
            os.replace(path, bak_path)
        except OSError:
            pass
    
    os.replace(tmp_path, path)


# -- per-email result saving
def save_extraction(artifact: Artifact, result: ExtractionResult):
    """dump one email's extraction to json"""
    output = {
        "artifact_id": artifact.id,
        "message_id": artifact.message_id,
        "file_path": artifact.file_path,
        "sender": artifact.sender,
        "date": artifact.date.isoformat() if artifact.date else None,
        "subject": artifact.subject,
        "extraction_version": SCHEMA_VERSION,
        "model": {"openrouter": config.OPENROUTER_MODEL, "groq": config.GROQ_MODEL, "ollama": config.OLLAMA_MODEL, "gemini": config.GEMINI_MODEL}.get(config.LLM_BACKEND, "unknown"),
        "extracted_at": datetime.utcnow().isoformat(),
        "entities": [e.model_dump() for e in result.entities],
        "claims": [c.model_dump() for c in result.claims],
    }
    
    filepath = os.path.join(config.EXTRACTIONS_DIR, f"{artifact.id}.json")
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    return filepath


# -- main extraction pipeline
# uses thread pool for cloud backends, sequential for local ollama

# thread-safe lock for checkpoint updates
_checkpoint_lock = threading.Lock()

def _process_one(artifact: 'Artifact') -> tuple:
    """thread pool worker - extract one email, returns (artifact, result or None)"""
    try:
        result = extract_from_email(artifact)
        return (artifact, result)
    except Exception as e:
        print(f"  [THREAD ERROR] {artifact.subject[:40]}: {e}")
        return (artifact, None)


def run_extraction(limit: Optional[int] = None):
    """the big one - loads emails, sends to LLM, saves results.
    handles checkpointing, dedup, parallel workers, all of it."""
    
    model_name = {"openrouter": config.OPENROUTER_MODEL, "groq": config.GROQ_MODEL, "ollama": config.OLLAMA_MODEL, "gemini": config.GEMINI_MODEL}.get(config.LLM_BACKEND, "unknown")
    
    # Parallelism: cloud backends can handle concurrent requests
    max_workers = {"openrouter": 10, "groq": 3, "ollama": 1, "gemini": 2}.get(config.LLM_BACKEND, 1)
    
    print("=" * 70)
    print(f"EXTRACTION PIPELINE — Schema {SCHEMA_VERSION}")
    print(f"Backend: {config.LLM_BACKEND.upper()} | Model: {model_name}")
    print(f"Workers: {max_workers} | Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print("=" * 70)
    
    # Verify LLM backend is reachable
    if config.LLM_BACKEND == "openrouter":
        if not config.OPENROUTER_API_KEY:
            print("\n[ERROR] No OpenRouter API key found! Set OPENROUTER_API_KEY in .env")
            return
        try:
            test_resp = call_openrouter("Reply OK", "test")
            print(f"OpenRouter OK — model {config.OPENROUTER_MODEL} responding")
        except Exception as e:
            print(f"\n[ERROR] OpenRouter test failed: {e}")
            return
    elif config.LLM_BACKEND == "groq":
        if not config.GROQ_API_KEY:
            print("\n[ERROR] No Groq API key found! Set GROQ_API_KEY in .env")
            return
        try:
            client = get_groq_client()
            r = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5,
            )
            print(f"Groq OK — model {config.GROQ_MODEL} responding")
        except Exception as e:
            print(f"\n[ERROR] Groq test failed: {e}")
            return
    elif config.LLM_BACKEND == "ollama":
        try:
            r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
            models = [m['name'] for m in r.json().get('models', [])]
            if not any(config.OLLAMA_MODEL in m for m in models):
                print(f"\n[ERROR] Model '{config.OLLAMA_MODEL}' not found in Ollama.")
                print(f"Available: {models}")
                print(f"Run: ollama pull {config.OLLAMA_MODEL}")
                return
            print(f"Ollama OK — {len(models)} model(s) available")
        except requests.exceptions.ConnectionError:
            print(f"\n[ERROR] Cannot connect to Ollama at {config.OLLAMA_URL}")
            print("Run: ollama serve")
            return
    
    # Load data
    print(f"\nLoading subset from {config.SUBSET_PATH}...")
    df = pd.read_csv(config.SUBSET_PATH)
    print(f"Loaded {len(df)} emails")
    
    if limit:
        df = df.head(limit)
        print(f"Limited to first {limit} emails (test mode)")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_ids = set(checkpoint.get("processed_ids", []))
    failed_ids = set(checkpoint.get("failed_ids", []))
    stats = checkpoint.get("stats", {"success": 0, "failed": 0, "total_entities": 0, "total_claims": 0})
    
    prev_done = len(processed_ids)
    print(f"Already processed: {prev_done} emails ({stats['failed']} previously failed)")
    
    # Parse all emails into Artifacts
    print("\nParsing emails...")
    artifacts = []
    for _, row in df.iterrows():
        artifact = parse_email(row['file'], row['message'])
        if artifact:
            artifacts.append(artifact)
    
    print(f"Successfully parsed: {len(artifacts)} / {len(df)}")
    
    # Deduplicate by body hash — Enron dataset stores same email in multiple folders
    seen_hashes = {}  # body_hash → artifact (first occurrence)
    unique_artifacts = []
    dupes = []
    for a in artifacts:
        if a.body_hash in seen_hashes:
            dupes.append((a, seen_hashes[a.body_hash]))  # (dupe, original)
        else:
            seen_hashes[a.body_hash] = a
            unique_artifacts.append(a)
    
    if dupes:
        print(f"Body-level duplicates skipped: {len(dupes)} (same email in multiple folders)")
    print(f"Unique emails to extract: {len(unique_artifacts)}")
    
    # Filter out already processed
    done_ids = processed_ids | failed_ids
    remaining = [a for a in unique_artifacts if a.message_id not in done_ids]
    print(f"Remaining to process: {len(remaining)}")
    
    if not remaining:
        print("All emails already processed!")
        return
    
    # ETA estimate (adjusted for parallelism)
    est_per_email = {"openrouter": 25, "groq": 5, "ollama": 26, "gemini": 8}
    base_time = est_per_email.get(config.LLM_BACKEND, 10)
    est_seconds = len(remaining) * base_time / max_workers
    print(f"Estimated time: ~{est_seconds/3600:.1f} hours ({est_seconds/60:.0f} minutes) with {max_workers} workers")
    print(f"Progress is saved continuously — safe to stop anytime (Ctrl+C)\n")
    
    # Process
    global _shutdown_requested
    start_time = time.time()
    session_success = 0
    session_failed = 0
    
    pbar = tqdm(total=len(remaining), desc="Extracting", unit="email")
    
    if max_workers > 1:
        # -- parallel mode (cloud backends)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_artifact = {}
            for artifact in remaining:
                if _shutdown_requested:
                    break
                future = executor.submit(_process_one, artifact)
                future_to_artifact[future] = artifact
            
            # Collect results as they complete
            for future in as_completed(future_to_artifact):
                if _shutdown_requested:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                artifact, result = future.result()
                
                with _checkpoint_lock:
                    if result:
                        save_extraction(artifact, result)
                        stats["success"] += 1
                        stats["total_entities"] += len(result.entities)
                        stats["total_claims"] += len(result.claims)
                        session_success += 1
                        processed_ids.add(artifact.message_id)
                    else:
                        stats["failed"] += 1
                        session_failed += 1
                        failed_ids.add(artifact.message_id)
                    
                    # Save checkpoint
                    checkpoint["processed_ids"] = list(processed_ids)
                    checkpoint["failed_ids"] = list(failed_ids)
                    checkpoint["stats"] = stats
                    checkpoint["last_saved"] = datetime.utcnow().isoformat()
                    save_checkpoint(checkpoint)
                
                pbar.update(1)
                elapsed = time.time() - start_time
                rate = (session_success + session_failed) / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    "ok": stats["success"],
                    "fail": stats["failed"],
                    "rate": f"{rate*60:.1f}/min",
                })
                
                # Detailed progress report every 100 emails
                total_done = session_success + session_failed
                if total_done % 100 == 0 and total_done > 0:
                    eta = (len(remaining) - total_done) / rate if rate > 0 else 0
                    print(f"\n  [{total_done}/{len(remaining)}] "
                          f"Entities: {stats['total_entities']} | Claims: {stats['total_claims']} | "
                          f"Rate: {rate*60:.1f}/min | ETA: {eta/60:.1f}min")
    else:
        # -- sequential mode (local ollama)
        for i, artifact in enumerate(remaining):
            if _shutdown_requested:
                print(f"\n[STOPPED] Saving progress after {session_success} emails...")
                break
            
            if i > 0:
                time.sleep(config.RATE_LIMIT_DELAY)
            
            result = extract_from_email(artifact)
            
            if result:
                save_extraction(artifact, result)
                stats["success"] += 1
                stats["total_entities"] += len(result.entities)
                stats["total_claims"] += len(result.claims)
                session_success += 1
                processed_ids.add(artifact.message_id)
            else:
                stats["failed"] += 1
                session_failed += 1
                failed_ids.add(artifact.message_id)
            
            checkpoint["processed_ids"] = list(processed_ids)
            checkpoint["failed_ids"] = list(failed_ids)
            checkpoint["stats"] = stats
            checkpoint["last_saved"] = datetime.utcnow().isoformat()
            save_checkpoint(checkpoint)
            
            pbar.update(1)
            elapsed = time.time() - start_time
            rate = (session_success + session_failed) / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                "ok": stats["success"],
                "fail": stats["failed"],
                "rate": f"{rate*60:.1f}/min",
            })
            
            if (i + 1) % 50 == 0:
                eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
                print(f"\n  [{i+1}/{len(remaining)}] "
                      f"Entities: {stats['total_entities']} | Claims: {stats['total_claims']} | "
                      f"Rate: {rate*60:.1f}/min | ETA: {eta/60:.1f}min")
    
    pbar.close()
    
    # Final checkpoint
    checkpoint["processed_ids"] = list(processed_ids)
    checkpoint["failed_ids"] = list(failed_ids)
    checkpoint["stats"] = stats
    checkpoint["last_saved"] = datetime.utcnow().isoformat()
    save_checkpoint(checkpoint)
    
    elapsed = time.time() - start_time
    
    # Final report
    stopped = " (interrupted — PROGRESS SAVED)" if _shutdown_requested else ""
    print("\n" + "=" * 70)
    print(f"EXTRACTION {'PAUSED' if _shutdown_requested else 'COMPLETE'}{stopped}")
    print("=" * 70)
    print(f"  This session:    {session_success + session_failed} emails in {elapsed/60:.1f}min")
    print(f"  Unique extracted: {stats['success']} / {len(unique_artifacts)}")
    print(f"  Duplicates skipped: {len(dupes)}")
    print(f"  Total failed:    {stats['failed']}")
    print(f"  Total entities:  {stats['total_entities']}")
    print(f"  Total claims:    {stats['total_claims']}")
    print(f"  Avg entities/email: {stats['total_entities']/max(stats['success'],1):.1f}")
    print(f"  Avg claims/email:   {stats['total_claims']/max(stats['success'],1):.1f}")
    remaining_count = len(unique_artifacts) - stats['success'] - stats['failed']
    if remaining_count > 0:
        print(f"  Remaining:       {remaining_count} emails")
        print(f"  To continue:     python extraction.py")
    print(f"  Results saved to: {config.EXTRACTIONS_DIR}")
    

if __name__ == "__main__":
    import sys
    
    # Allow test mode: python extraction.py --test 5
    limit = None
    if '--test' in sys.argv:
        idx = sys.argv.index('--test')
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])
        else:
            limit = 5
    
    run_extraction(limit=limit)
