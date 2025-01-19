#!/usr/bin/env python3

import csv, sys

INSTANCE_START_IDX = 8


def parse_instances(input_csv: str) -> str:
    """
    Parse the CSV file to generate an HTML list of instance counts.

    This function reads a CSV file, counts the occurrences of different instances, and returns the
    counts as an HTML unordered list.

    Parameters:
    input_csv (str): Path to the input CSV file containing instance data.

    Returns:
    str: HTML string representing an unordered list of instance counts.
    """

    with open(input_csv, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)[INSTANCE_START_IDX:]
        ninstances = [0] * len(headers)
        for row in reader:
            for i, n in enumerate(row[INSTANCE_START_IDX:]):
                ninstances[i] += int(n)

        html_list_items = []
        for h, n in zip(headers, ninstances):
            html_list_items.append(f"<li>{h}: {n}</li>")

    return "\n".join(html_list_items)


if __name__ == "__main__":

    print(parse_instances(sys.argv[1]))
