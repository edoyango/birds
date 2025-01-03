#!/usr/bin/env python3

import csv, sys

INSTANCE_START_IDX = 8

with open(sys.argv[1], "r") as f:
    reader = csv.reader(f)
    headers = next(reader)[INSTANCE_START_IDX:]
    ninstances = [0]*len(headers)
    for row in reader:
        for i, n in enumerate(row[INSTANCE_START_IDX:]):
            ninstances[i] += int(n)

    for h, n in zip(headers, ninstances):
        print(f"<li>{h}: {n}</li>")
    
