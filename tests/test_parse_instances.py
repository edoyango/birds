"""Tests for birdector.post_process.parse_instances."""

import csv

from birdector.post_process.parse_instances import parse_instances


def test_parse_instances_sums_counts_across_rows(tmp_path):
    csv_path = tmp_path / "meta.csv"
    header = ["a", "b", "c", "d", "e", "f", "g", "h", "sparrow", "robin"]
    rows = [
        ["na"] * 8 + [2, 0],
        ["na"] * 8 + [1, 3],
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    html = parse_instances(csv_path)

    assert "<li>sparrow: 3</li>" in html
    assert "<li>robin: 3</li>" in html


def test_parse_instances_with_no_data_rows(tmp_path):
    csv_path = tmp_path / "meta.csv"
    header = ["a"] * 8 + ["sparrow"]
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    assert parse_instances(csv_path) == "<li>sparrow: 0</li>"
