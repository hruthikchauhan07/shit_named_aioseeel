#!/usr/bin/env python3
"""
Build trusted_counts.py from dbb.xlsx (sheet: Manual Actual Entries).
Uses actual counts for 14kg, 19kg, 425kg. Ignores 5kg and 47kg.
Overwrites existing trusted_counts.py.
"""

import pandas as pd
import os
from collections import defaultdict, Counter

def infer_truck_type(kg14, kg19, kg425):
    """Infer truck type from cylinder counts."""
    if kg425 > 0:
        return "425 KG JUMBO TRUCK"
    if kg14 > 0 and kg19 > 0:
        return "MIXED LOAD"
    if kg14 > 0:
        return "14.5 KG TRUCK"
    if kg19 > 0:
        return "19 KG TRUCK"
    return "UNKNOWN"

def main():
    excel_file = "dbb.xlsx"
    if not os.path.exists(excel_file):
        print(f"❌ File '{excel_file}' not found.")
        return

    # Read the sheet
    df = pd.read_excel(excel_file, sheet_name='Manual Actual Entries', dtype=str)
    df['Vehicle Number'] = df['Vehicle Number'].str.upper().str.strip()
    df = df[df['Vehicle Number'].notna() & (df['Vehicle Number'] != '')]

    # Determine which columns exist: prefer Actual 14kg, etc., or fallback to KG14 etc.
    # We'll try to map to standard names
    col_map = {}
    for col in df.columns:
        col_clean = col.strip().lower()
        if 'vehicle' in col_clean and 'number' in col_clean:
            col_map['vehicle'] = col
        elif '14' in col_clean or '14kg' in col_clean:
            col_map['kg14'] = col
        elif '19' in col_clean or '19kg' in col_clean:
            col_map['kg19'] = col
        elif '42.5' in col_clean or '425' in col_clean or '42.5kg' in col_clean:
            col_map['kg425'] = col
        elif 'actual total' in col_clean:
            col_map['total'] = col

    # If we don't find the columns, fallback to default names (like in your original script)
    if 'kg14' not in col_map:
        col_map['kg14'] = 'Actual 14kg'
    if 'kg19' not in col_map:
        col_map['kg19'] = 'Actual 19kg'
    if 'kg425' not in col_map:
        col_map['kg425'] = 'Actual 42.5kg'

    # Convert count columns to numeric
    for key in ['kg14', 'kg19', 'kg425']:
        if key in col_map:
            df[col_map[key]] = pd.to_numeric(df[col_map[key]], errors='coerce').fillna(0).astype(int)
        else:
            df[col_map[key]] = 0

    # Compute total from sum if not present, else use provided total
    if 'total' in col_map:
        df['total'] = pd.to_numeric(df[col_map['total']], errors='coerce').fillna(0).astype(int)
    else:
        df['total'] = df[col_map['kg14']] + df[col_map['kg19']] + df[col_map['kg425']]

    # Group by vehicle
    plate_data = defaultdict(list)
    for _, row in df.iterrows():
        plate = row['Vehicle Number']
        total = row['total']
        kg14 = row[col_map['kg14']]
        kg19 = row[col_map['kg19']]
        kg425 = row[col_map['kg425']]
        if total == 0 or total > 5000:
            continue
        plate_data[plate].append({
            'total': total,
            'kg14': kg14,
            'kg19': kg19,
            'kg425': kg425,
        })

    trusted = {}
    for plate, records in plate_data.items():
        total_counts = Counter([r['total'] for r in records])
        most_common_total, freq = total_counts.most_common(1)[0]
        example = next(r for r in records if r['total'] == most_common_total)

        trusted[plate] = {
            'total': most_common_total,
            'kg14': example['kg14'],
            'kg19': example['kg19'],
            'kg425': example['kg425'],
            'kg47': 0,
            'kg5': 0,
            'truck_type': infer_truck_type(example['kg14'], example['kg19'], example['kg425']),
            'frequency': freq,
            'total_records': len(records)
        }

    # Write to trusted_counts.py
    with open('trusted_counts.py', 'w') as f:
        f.write('#!/usr/bin/env python3\n')
        f.write('"""\n')
        f.write('Trusted plate counts built from dbb.xlsx (Manual Actual Entries).\n')
        f.write('Uses only 14kg, 19kg, 425kg. 5kg and 47kg are set to 0.\n')
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

    print(f"✅ Updated trusted_counts.py with {len(trusted)} plates.")
    print(f"Truck types assigned: 14.5 KG TRUCK / 19 KG TRUCK / MIXED LOAD / 425 KG JUMBO TRUCK / UNKNOWN")

if __name__ == '__main__':
    main()