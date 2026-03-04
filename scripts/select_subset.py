# select_subset.py - picks ~1500-2000 emails from the full 517k enron dataset
#
# random sampling would give us disconnected emails with no threads,
# no overlapping people, no dedup opportunities - basically useless.
# instead we find a cluster of interconnected employees whose emails
# form real conversations with threads, forwards, shared projects etc.
#
# how it works:
#   1. stream all 517k emails, extract sender/recipient pairs
#   2. find the most active enron employees
#   3. find which of them email each other the most
#   4. greedily expand the cluster from the strongest pair
#   5. pull all emails from those mailboxes
#   6. trim by time window if we got too many

import pandas as pd
import email
from email import policy
from collections import Counter, defaultdict
from tqdm import tqdm
import re

DATA_PATH = "/home/misbah_ubuntu/LayerAI(task)/data/emails.csv"
OUTPUT_PATH = "/home/misbah_ubuntu/LayerAI(task)/data/enron_subset.csv"

# -- step 1: stream through everything, map the communication network
print("=" * 70)
print("STEP 1: Streaming through all 517K emails to map the network...")
print("=" * 70)

sender_counts = Counter()
mailbox_counts = Counter()
edge_counts = Counter()  # (sender, recipient) pairs
mailbox_to_sender = defaultdict(set)

chunk_size = 10000
total_processed = 0

for chunk in tqdm(pd.read_csv(DATA_PATH, chunksize=chunk_size), 
                  desc="Processing", total=52):
    for _, row in chunk.iterrows():
        try:
            msg = email.message_from_string(row['message'], policy=policy.default)
            sender = msg.get('From', '').strip().lower()
            to_raw = msg.get('To', '') or ''
            cc_raw = msg.get('Cc', '') or ''
            x_origin = msg.get('X-Origin', '').strip()
            
            # Parse file path for mailbox
            mailbox = str(row.get('file', '')).split('/')[0] if pd.notna(row.get('file')) else ''
            
            if sender:
                sender_counts[sender] += 1
                mailbox_counts[mailbox] += 1
                mailbox_to_sender[mailbox].add(sender)
                
                # Parse recipients
                all_recipients = to_raw + ',' + cc_raw
                recipients = [r.strip().lower() for r in all_recipients.split(',') if '@' in r]
                for recip in recipients:
                    edge_counts[(sender, recip)] += 1
        except:
            pass
    
    total_processed += len(chunk)

print(f"\nTotal processed: {total_processed:,}")
print(f"Unique senders: {len(sender_counts):,}")
print(f"Unique mailboxes: {len(mailbox_counts):,}")
print(f"Unique communication edges: {len(edge_counts):,}")

# -- step 2: find the most active enron employees
print("\n" + "=" * 70)
print("STEP 2: TOP 30 SENDERS")
print("=" * 70)

# Filter for @enron.com senders only (not newsletters etc.)
enron_senders = {s: c for s, c in sender_counts.items() if '@enron.com' in s}
top_senders = sorted(enron_senders.items(), key=lambda x: -x[1])[:30]

for rank, (sender, count) in enumerate(top_senders, 1):
    print(f"  {rank:2d}. {count:5d} emails — {sender}")

# -- step 3: find who talks to who the most
print("\n" + "=" * 70)
print("STEP 3: FINDING INTERCONNECTED COMMUNICATION CLUSTERS")
print("=" * 70)

# Take top 20 senders and find who they communicate with most
top20_emails = set(s for s, _ in top_senders[:20])

# For each pair of top senders, count bidirectional communication
pair_scores = Counter()
for (s, r), count in edge_counts.items():
    if s in top20_emails and r in top20_emails:
        pair = tuple(sorted([s, r]))
        pair_scores[pair] += count

print("\nTop 20 communication pairs (among top senders):")
for pair, count in pair_scores.most_common(20):
    print(f"  {count:4d} emails — {pair[0]} <-> {pair[1]}")

# -- step 4: greedily build the best cluster starting from strongest pair
print("\n" + "=" * 70)
print("STEP 4: SELECTING OPTIMAL CLUSTER")
print("=" * 70)

# Find the pair with strongest connection
best_pair = pair_scores.most_common(1)[0][0]
cluster = set(best_pair)

# Greedily add people who have strong connections to the existing cluster
for _ in range(15):  # Try to add up to 15 more people
    best_addition = None
    best_score = 0
    
    for candidate in top20_emails - cluster:
        # Score = total emails between candidate and cluster members
        score = 0
        for member in cluster:
            pair = tuple(sorted([candidate, member]))
            score += pair_scores.get(pair, 0)
        
        if score > best_score:
            best_score = score
            best_addition = candidate
    
    if best_addition and best_score > 5:  # Only add if meaningful connection
        cluster.add(best_addition)
        print(f"  Added {best_addition} (connection score: {best_score})")
    else:
        break

print(f"\nFinal cluster ({len(cluster)} people):")
for person in sorted(cluster):
    print(f"  - {person} ({enron_senders.get(person, 0)} emails)")

# -- step 5: figure out which mailbox folders have these people's emails
print("\n" + "=" * 70)
print("STEP 5: IDENTIFYING RELEVANT MAILBOXES")
print("=" * 70)

# Map email addresses to mailbox names
relevant_mailboxes = set()
for mailbox, senders in mailbox_to_sender.items():
    for sender in senders:
        if sender in cluster:
            relevant_mailboxes.add(mailbox)

print(f"\nRelevant mailboxes: {len(relevant_mailboxes)}")
for mb in sorted(relevant_mailboxes):
    print(f"  - {mb} ({mailbox_counts.get(mb, 0)} emails)")

# -- step 6: actually pull the emails from those mailboxes
print("\n" + "=" * 70)
print("STEP 6: EXTRACTING SUBSET")
print("=" * 70)

# Stream through and extract emails from these mailboxes
# Also include emails TO anyone in our cluster (for complete conversations)
subset_rows = []

for chunk in tqdm(pd.read_csv(DATA_PATH, chunksize=10000), desc="Extracting", total=52):
    for _, row in chunk.iterrows():
        try:
            mailbox = str(row.get('file', '')).split('/')[0]
            
            # Include if from a relevant mailbox
            if mailbox in relevant_mailboxes:
                subset_rows.append(row)
                continue
            
            # Also include if addressed to someone in our cluster
            msg = email.message_from_string(row['message'], policy=policy.default)
            sender = (msg.get('From', '') or '').strip().lower()
            to_raw = (msg.get('To', '') or '').lower()
            
            if sender in cluster or any(person in to_raw for person in cluster):
                subset_rows.append(row)
        except:
            pass

print(f"\nTotal emails in subset (before trimming): {len(subset_rows):,}")

# Create DataFrame
df_subset = pd.DataFrame(subset_rows)

# -- step 7: trim if too big, flag if too small
print("\n" + "=" * 70)
print("STEP 7: SIZE ADJUSTMENT")
print("=" * 70)

TARGET_MIN = 1200
TARGET_MAX = 2000

if len(df_subset) > TARGET_MAX:
    print(f"Subset is {len(df_subset)} which is > {TARGET_MAX}. Trimming by time window...")
    
    # Parse dates and sort
    def get_date(msg_str):
        try:
            msg = email.message_from_string(msg_str, policy=policy.default)
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(msg.get('Date', ''))
        except:
            return None
    
    df_subset['parsed_date'] = df_subset['message'].apply(get_date)
    df_subset = df_subset.dropna(subset=['parsed_date'])
    df_subset = df_subset.sort_values('parsed_date')
    
    # Find the densest time window that gives us ~TARGET_MAX emails
    # Focus on the most active period (mid-2000 to mid-2001)
    from datetime import datetime, timezone
    start = pd.Timestamp('2000-06-01', tz='UTC')
    end = pd.Timestamp('2001-12-31', tz='UTC')
    
    mask = (df_subset['parsed_date'] >= start) & (df_subset['parsed_date'] <= end)
    df_subset = df_subset[mask]
    
    if len(df_subset) > TARGET_MAX:
        # Sample within the window, keeping thread continuity where possible
        df_subset = df_subset.head(TARGET_MAX)
    
    df_subset = df_subset.drop(columns=['parsed_date'])
    print(f"After time windowing: {len(df_subset)} emails")

elif len(df_subset) < TARGET_MIN:
    print(f"Subset is {len(df_subset)} which is < {TARGET_MIN}. Consider expanding cluster.")
else:
    print(f"Subset size ({len(df_subset)}) is within target range [{TARGET_MIN}, {TARGET_MAX}].")

# -- step 8: final stats on what we got
print("\n" + "=" * 70)
print("STEP 8: FINAL SUBSET STATISTICS")
print("=" * 70)

# Reparse for stats
parsed_list = []
for _, row in df_subset.iterrows():
    try:
        msg = email.message_from_string(row['message'], policy=policy.default)
        parsed_list.append({
            'from': (msg.get('From', '') or '').strip().lower(),
            'to': msg.get('To', '') or '',
            'subject': msg.get('Subject', '') or '',
            'date': msg.get('Date', '') or '',
            'body_len': len(str(msg.get_payload() or '')),
            'has_forward': 1 if '--- Forwarded' in str(msg.get_payload() or '') else 0,
            'has_reply': 1 if str(msg.get('Subject', '')).lower().startswith('re:') else 0,
        })
    except:
        pass

df_stats = pd.DataFrame(parsed_list)
print(f"\n  Total emails:        {len(df_subset):,}")
print(f"  Unique senders:      {df_stats['from'].nunique()}")
print(f"  Reply threads:       {df_stats['has_reply'].sum()} ({df_stats['has_reply'].mean()*100:.1f}%)")
print(f"  Forwards:            {df_stats['has_forward'].sum()} ({df_stats['has_forward'].mean()*100:.1f}%)")
print(f"  Avg body length:     {df_stats['body_len'].mean():.0f} chars")
print(f"  Unique subjects:     {df_stats['subject'].nunique()}")

# -- step 9: save to csv
print("\n" + "=" * 70)
print(f"SAVING SUBSET to {OUTPUT_PATH}")
print("=" * 70)

df_subset.to_csv(OUTPUT_PATH, index=False)
file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"  Saved: {len(df_subset)} rows, {file_size_mb:.1f} MB")
print("\nDone!")
