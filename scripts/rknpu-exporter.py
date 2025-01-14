#!/usr/bin/env python3

from flask import Flask, Response
import re

app = Flask(__name__)

# Function to parse NPU load data
def parse_npu_load():
    try:
        with open('/rknpu-load.txt', 'r') as file:
            data = file.read()
        # Use regex to extract load percentages for each core
        pattern = r"Core(\d+):\s*(\d+)%"
        matches = re.findall(pattern, data)
        if not matches:
            raise ValueError("No valid core data found.")
        # Return a dictionary with core numbers and their loads
        return {f"core_{core}": int(load) for core, load in matches}
    except Exception as e:
        return {}

@app.route('/metrics')
def metrics():
    npu_load = parse_npu_load()
    # Generate Prometheus metrics output
    metrics = []
    metrics.append("# HELP rknpu_core_load_percentage NPU load percentage per core")
    metrics.append("# TYPE rknpu_core_load_percentage gauge")
    for core, load in npu_load.items():
        metrics.append(f"rknpu_core_load_percentage{{core=\"{core}\"}} {load}")
    return Response("\n".join(metrics), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9092)
