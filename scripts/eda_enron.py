# eda_enron.py - exploratory data analysis on the enron email dataset
# goal: understand the data before picking our subset
# we're ram constrained (7.6gb total) so we peek at small chunks first
# then stream for stats without loading everything at once

import pandas as pd
import os
from collections import Counter

DATA_PATH = "/home/misbah_ubuntu/LayerAI(task)/data/emails.csv"

# -- step 1: basic structure (first 100 rows just to see whats there)
print("=" * 70)
print("STEP 1: BASIC STRUCTURE (first 100 rows)")
print("=" * 70)

df_sample = pd.read_csv(DATA_PATH, nrows=100)

print(f"\nColumns: {list(df_sample.columns)}")
print(f"Shape (sample): {df_sample.shape}")
print(f"\nData types:\n{df_sample.dtypes}")
print(f"\nFirst 3 rows (truncated):")
for i in range(3):
    print(f"\n--- Row {i} ---")
    for col in df_sample.columns:
        val = str(df_sample.iloc[i][col])
        print(f"  {col}: {val[:200]}{'...' if len(val) > 200 else ''}")

# -- step 2: count total rows by streaming (don't want to blow up ram)
print("\n" + "=" * 70)
print("STEP 2: TOTAL ROW COUNT")
print("=" * 70)

total_rows = 0
for chunk in pd.read_csv(DATA_PATH, chunksize=10000, usecols=[0]):
    total_rows += len(chunk)
print(f"\nTotal emails in dataset: {total_rows:,}")

# -- step 3: parse email headers from the raw message column
print("\n" + "=" * 70)
print("STEP 3: EMAIL STRUCTURE ANALYSIS (sample of 5000 rows)")
print("=" * 70)

import email
from email import policy

def parse_email_fields(raw_message):
    """pull out key headers from a raw email string"""
    try:
        msg = email.message_from_string(raw_message, policy=policy.default)
        return {
            'from': msg.get('From', ''),
            'to': msg.get('To', ''),
            'cc': msg.get('Cc', ''),
            'bcc': msg.get('Bcc', ''),
            'subject': msg.get('Subject', ''),
            'date': msg.get('Date', ''),
            'x_from': msg.get('X-From', ''),
            'x_to': msg.get('X-To', ''),
            'x_folder': msg.get('X-Folder', ''),
            'x_origin': msg.get('X-Origin', ''),
            'body': msg.get_payload() if isinstance(msg.get_payload(), str) else str(msg.get_payload()),
        }
    except Exception as e:
        return {'error': str(e)}

# Parse a larger sample for statistics
df_5k = pd.read_csv(DATA_PATH, nrows=5000)
parsed = df_5k['message'].apply(parse_email_fields)
df_parsed = pd.DataFrame(parsed.tolist())

print(f"\nParsed columns: {list(df_parsed.columns)}")
print(f"\nSample parsed email:")
for col in df_parsed.columns:
    val = str(df_parsed.iloc[0][col])
    print(f"  {col}: {val[:150]}{'...' if len(val) > 150 else ''}")

# -- step 4: who sends the most emails?
print("\n" + "=" * 70)
print("STEP 4: TOP SENDERS (from 5000-row sample)")
print("=" * 70)

sender_counts = df_parsed['from'].value_counts()
print(f"\nTotal unique senders in sample: {len(sender_counts)}")
print(f"\nTop 20 senders:")
for sender, count in sender_counts.head(20).items():
    print(f"  {count:4d} emails — {sender}")

# -- step 5: X-Origin (which employee's mailbox)
print("\n" + "=" * 70)
print("STEP 5: MAILBOX ORIGINS (X-Origin)")
print("=" * 70)

origin_counts = df_parsed['x_origin'].value_counts()
print(f"\nTotal unique origins in sample: {len(origin_counts)}")
print(f"\nTop 20 origins:")
for origin, count in origin_counts.head(20).items():
    print(f"  {count:4d} emails — {origin}")

# -- step 6: what folders are emails stored in?
print("\n" + "=" * 70)
print("STEP 6: FOLDER TYPES (X-Folder)")
print("=" * 70)

# Extract folder type (first part of path)
folder_types = df_parsed['x_folder'].apply(
    lambda x: str(x).split('\\')[-1] if pd.notna(x) else 'unknown'
)
folder_counts = folder_types.value_counts()
print(f"\nTop 20 folder types:")
for folder, count in folder_counts.head(20).items():
    print(f"  {count:4d} — {folder}")

# -- step 7: how long are the email bodies?
print("\n" + "=" * 70)
print("STEP 7: EMAIL BODY LENGTH STATS")
print("=" * 70)

body_lengths = df_parsed['body'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
print(f"\n  Min length:    {body_lengths.min():,} chars")
print(f"  Max length:    {body_lengths.max():,} chars")
print(f"  Mean length:   {body_lengths.mean():,.0f} chars")
print(f"  Median length: {body_lengths.median():,.0f} chars")
print(f"  Std dev:       {body_lengths.std():,.0f} chars")

# Distribution buckets
print(f"\n  Length distribution:")
buckets = [(0, 100), (100, 500), (500, 1000), (1000, 5000), (5000, 10000), (10000, 50000), (50000, float('inf'))]
for low, high in buckets:
    count = ((body_lengths >= low) & (body_lengths < high)).sum()
    pct = count / len(body_lengths) * 100
    label = f"{low}-{high}" if high != float('inf') else f"{low}+"
    print(f"    {label:>12s} chars: {count:4d} ({pct:.1f}%)")

# -- step 8: date range of the emails
print("\n" + "=" * 70)
print("STEP 8: DATE RANGE")
print("=" * 70)

from email.utils import parsedate_to_datetime

dates = []
for d in df_parsed['date']:
    try:
        dates.append(parsedate_to_datetime(str(d)))
    except:
        pass

if dates:
    dates.sort()
    print(f"\n  Earliest email: {dates[0]}")
    print(f"  Latest email:   {dates[-1]}")
    print(f"  Date range:     {(dates[-1] - dates[0]).days} days")
    
    # Monthly distribution
    from collections import Counter
    monthly = Counter(d.strftime('%Y-%m') for d in dates)
    print(f"\n  Emails by month (top 15):")
    for month, count in sorted(monthly.items(), key=lambda x: -x[1])[:15]:
        print(f"    {month}: {count}")

# -- step 9: how many forwarded/quoted emails?
print("\n" + "=" * 70)
print("STEP 9: FORWARDING & QUOTING PATTERNS")
print("=" * 70)

fwd_count = df_parsed['subject'].str.contains(r'(?i)(fw:|fwd:)', na=False).sum()
re_count = df_parsed['subject'].str.contains(r'(?i)(re:)', na=False).sum()
original_msg = df_parsed['body'].str.contains(r'-----\s*Original Message', na=False).sum()
fwd_by = df_parsed['body'].str.contains(r'-----\s*Forwarded by', na=False).sum()

print(f"\n  Forwarded emails (FW:/FWD: in subject): {fwd_count} ({fwd_count/len(df_parsed)*100:.1f}%)")
print(f"  Reply emails (RE: in subject):          {re_count} ({re_count/len(df_parsed)*100:.1f}%)")
print(f"  Contains 'Original Message' marker:     {original_msg} ({original_msg/len(df_parsed)*100:.1f}%)")
print(f"  Contains 'Forwarded by' marker:         {fwd_by} ({fwd_by/len(df_parsed)*100:.1f}%)")

# -- step 10: multi-recipient patterns
print("\n" + "=" * 70)
print("STEP 10: RECIPIENT ANALYSIS")
print("=" * 70)

to_counts = df_parsed['to'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) and str(x).strip() else 0)
print(f"\n  Avg recipients per email: {to_counts.mean():.1f}")
print(f"  Max recipients:           {to_counts.max()}")
print(f"  Emails with CC:           {(df_parsed['cc'].notna() & (df_parsed['cc'] != '')).sum()}")
print(f"  Emails with BCC:          {(df_parsed['bcc'].notna() & (df_parsed['bcc'] != '')).sum()}")

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
