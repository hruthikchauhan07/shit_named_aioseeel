#!/usr/bin/env python3
"""
Merge new manual counts from d2.xlsx (CSV) into existing trusted_counts.py.
For each plate, combine old and new records, and choose the most frequent total.
Existing plates are updated if new data changes the majority; new plates are added.
Usage: python merge_trusted_with_update.py
Output: trusted_counts.py (updated with merged counts)
"""

import pandas as pd
import os
import sys
from collections import defaultdict, Counter
import importlib.util

# ─── Load existing trusted counts ──────────────────────────────────────────
def load_existing_trusted():
    trusted = {}
    try:
        spec = importlib.util.spec_from_file_location("trusted_counts", "trusted_counts.py")
        if spec is None or spec.loader is None:
            print("⚠️ No trusted_counts.py found – starting fresh.")
            return trusted
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'TRUSTED_PLATE_COUNTS'):
            trusted = module.TRUSTED_PLATE_COUNTS
            print(f"✅ Loaded {len(trusted)} plates from existing trusted_counts.py")
        else:
            print("⚠️ Existing file missing TRUSTED_PLATE_COUNTS – starting fresh.")
    except Exception as e:
        print(f"⚠️ Could not load existing trusted_counts.py: {e}")
        print("Starting fresh.")
    return trusted

# ─── Read new data from d2.xlsx (CSV) ────────────────────────────────────
def read_new_data():
    csv_file = "d2.csv"  # actually CSV
    if not os.path.exists(csv_file):
        print(f"❌ File '{csv_file}' not found.")
        sys.exit(1)

    df = pd.read_csv(csv_file, dtype=str)
    df.columns = df.columns.str.strip()

    required = ['Vehicle Number', 'Actual Total', 'KG14', 'KG19', 'KG425']
    for col in required:
        if col not in df.columns:
            print(f"❌ Column '{col}' not found. Available: {df.columns.tolist()}")
            sys.exit(1)

    df['Vehicle Number'] = df['Vehicle Number'].str.upper().str.strip()
    df = df[df['Vehicle Number'].notna() & (df['Vehicle Number'] != '')]

    new_records = defaultdict(list)

    for _, row in df.iterrows():
        plate = row['Vehicle Number']
        def to_int(val):
            try:
                return int(float(str(val).strip()) if str(val).strip() else 0)
            except:
                return 0

        kg14 = to_int(row['KG14'])
        kg19 = to_int(row['KG19'])
        kg425 = to_int(row['KG425'])
        total_actual = to_int(row['Actual Total'])
        total = total_actual if total_actual > 0 else kg14 + kg19 + kg425

        if total == 0:
            continue

        new_records[plate].append({
            'total': total,
            'kg14': kg14,
            'kg19': kg19,
            'kg425': kg425,
        })

    print(f"✅ Loaded {len(new_records)} plates from CSV.")
    return new_records

# ─── Infer truck type ──────────────────────────────────────────────────
def infer_truck_type(kg14, kg19, kg425):
    if kg425 > 0:
        return "425 KG JUMBO TRUCK"
    if kg14 > 0 and kg19 > 0:
        return "MIXED LOAD"
    if kg14 > 0:
        return "14.5 KG TRUCK"
    if kg19 > 0:
        return "19 KG TRUCK"
    return "UNKNOWN"

# ─── Merge with majority vote ─────────────────────────────────────────────
def merge_trusted(existing, new_records):
    all_records = defaultdict(list)

    # Add existing records
    for plate, info in existing.items():
        # existing info is a dict with total, kg14, kg19, kg425, truck_type, frequency, total_records
        # We'll add one record per existing entry (frequency is not used here as a record count; we treat it as one observation)
        # But the existing entry already represents a consensus from previous merges.
        # To be fair, we should treat the existing total as one vote, but with weight = its frequency?
        # The simplest: treat the existing as one record (like a single observation).
        # However, frequency indicates how many times that total was observed historically.
        # We'll use frequency as the weight for that record.
        all_records[plate].append({
            'total': info['total'],
            'kg14': info['kg14'],
            'kg19': info['kg19'],
            'kg425': info['kg425'],
            'weight': info.get('frequency', 1)  # use frequency as weight
        })

    # Add new records
    for plate, recs in new_records.items():
        for rec in recs:
            all_records[plate].append({
                'total': rec['total'],
                'kg14': rec['kg14'],
                'kg19': rec['kg19'],
                'kg425': rec['kg425'],
                'weight': 1
            })

    final = {}
    updated_count = 0
    added_count = 0
    unchanged_count = 0

    for plate, records in all_records.items():
        # Weighted majority vote: sum weights for each total
        total_votes = Counter()
        for rec in records:
            total_votes[rec['total']] += rec['weight']

        most_common_total, total_weight = total_votes.most_common(1)[0]

        # Find a record with that total (prefer one with non-zero weights)
        chosen = None
        for rec in records:
            if rec['total'] == most_common_total:
                chosen = rec
                break

        if chosen is None:
            continue

        # Determine if this plate already existed
        existing_plate = plate in existing

        if existing_plate:
            old_total = existing[plate]['total']
            if old_total != most_common_total:
                updated_count += 1
                print(f"🔄 Updating '{plate}': {old_total} → {most_common_total} (weighted votes: {total_weight})")
            else:
                unchanged_count += 1
                # Keep old entry exactly (to preserve truck_type etc. if unchanged)
                final[plate] = existing[plate]
                continue
        else:
            added_count += 1
            print(f"➕ Adding new plate '{plate}' with total={most_common_total}")

        # Build new entry
        kg14 = chosen['kg14']
        kg19 = chosen['kg19']
        kg425 = chosen['kg425']
        truck_type = infer_truck_type(kg14, kg19, kg425)

        # Calculate total_records as count of all records (including weights)
        total_records = len(records)

        final[plate] = {
            'total': most_common_total,
            'kg14': kg14,
            'kg19': kg19,
            'kg425': kg425,
            'kg47': 0,
            'kg5': 0,
            'truck_type': truck_type,
            'frequency': total_weight,  # total weighted votes for this total
            'total_records': total_records
        }

    print(f"\n📊 Summary: {added_count} added, {updated_count} updated, {unchanged_count} unchanged.")
    return final

# ─── Write the new trusted_counts.py ──────────────────────────────────────
def write_trusted_counts(trusted):
    with open('trusted_counts.py', 'w') as f:
        f.write('#!/usr/bin/env python3\n')
        f.write('"""\nTrusted plate counts built from manual actual entries (merged with updates).\n')
        f.write('Ignored 5kg and 47kg. Only 14kg, 19kg, 425kg are used.\n')
        f.write('"""\n\n')
        f.write('TRUSTED_PLATE_COUNTS = {\n')
        for plate, info in sorted(trusted.items()):
            f.write(f'    "{plate}": {{\n')
            f.write(f'        "total": {info["total"]},\n')
            f.write(f'        "kg14": {info["kg14"]},\n')
            f.write(f'        "kg19": {info["kg19"]},\n')
            f.write(f'        "kg425": {info["kg425"]},\n')
            f.write(f'        "kg47": 0,\n')
            f.write(f'        "kg5": 0,\n')
            f.write(f'        "truck_type": "{info["truck_type"]}",\n')
            f.write(f'        "frequency": {info["frequency"]},\n')
            f.write(f'        "total_records": {info["total_records"]}\n')
            f.write('    },\n')
        f.write('}\n')

    print(f"✅ Saved updated trusted_counts.py with {len(trusted)} total plates.")

def main():
    print("=" * 60)
    print("MERGING TRUSTED COUNTS (with updates)")
    print("=" * 60)

    existing = load_existing_trusted()
    new_records = read_new_data()
    if not new_records and not existing:
        print("❌ No data found. Exiting.")
        return

    trusted = merge_trusted(existing, new_records)
    write_trusted_counts(trusted)

    print("\n✅ Done! Restart your system to use the updated trusted counts.")

if __name__ == '__main__':
    main()