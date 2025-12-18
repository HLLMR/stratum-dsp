#!/usr/bin/env python3
"""Select random tracks for testing"""

import csv
import random
import sys

test_batch = "../validation-data/results/test_batch_20251217_080129.csv"

with open(test_batch, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

selected = random.sample(rows, 5)
for row in selected:
    print(f"{row['track_id']}|{row['bpm_gt']}|{row['filename']}")

