#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from ..config import load_config

WEIGHTS = {i[0]: i[1]["weight"] for i in load_config().items()}

def calculate_simpsons_diversity(species_counts: list[int]) -> float:
    """
    Calculate Simpson's Diversity Index for a video.

    Parameters:
    species_counts: List of species counts for the video.

    Returns:
    float: Simpson's Diversity Index.
    """
    total_count = sum(species_counts)
    if total_count == 0:
        return 0  # Handle cases where no species are present
    proportions = [count / total_count for count in species_counts]
    return 1 - sum(p**2 for p in proportions)


def main(duration: float, num_videos: int, csv_path: Path) -> list[Path]:
    """
    Process a CSV file to filter and rank videos based on duration and Simpson's Diversity Index.

    Parameters:
    duration (float): Minimum duration of videos in seconds.
    num_videos (int): Number of top videos to return.
    csv_path (Path): Path to the CSV file containing video data.

    Returns:
    list: List of top video paths based on the Simpson's Diversity Index.
    """

    # Load CSV file
    data = pd.read_csv(csv_path, header=0)  # No header assumed; adjust if needed

    # Extract relevant columns
    video_column = "trigger video path"  # 5th column (0-based index)
    frames_column = "nframes"  # 7th column (0-based index)
    species_columns = list(WEIGHTS.keys())  # 9th column onward (0-based index)
    metric_column = "simpsons_diversity"

    # Filter rows by minimum duration
    min_frames = duration * 10  # assume 10 fps
    filtered_data = data[data[frames_column] >= min_frames].copy()

    # apply weights to species columns
    filtered_data[species_columns] = filtered_data[species_columns].mul(WEIGHTS, axis=1)

    # Calculate Simpson's Diversity Index for each video
    filtered_data[metric_column] = filtered_data[species_columns].apply(
        lambda row: calculate_simpsons_diversity(row.tolist()), axis=1
    )

    # weight simpson's diversity with no. of frames
    filtered_data[metric_column] = filtered_data[metric_column] * np.log(
        filtered_data[frames_column]
    )

    # Sort by Simpson's Diversity Index in descending order
    sorted_data = filtered_data.sort_values(by="simpsons_diversity", ascending=False)

    # Select top n videos
    top_videos = [Path(v) for v in sorted_data[video_column].head(num_videos)]

    # Print the video names space-delimited
    return top_videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select videos based on diversity and duration."
    )
    parser.add_argument(
        "--duration",
        type=int,
        required=True,
        help="Minimum duration in seconds.",
        default=11,
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        required=True,
        help="Number of videos to select.",
        default=4,
    )
    parser.add_argument(
        "--csv", type=str, default="meta.csv", help="Path to the input CSV file."
    )
    args = parser.parse_args()

    top_videos = main(args.duration, args.num_videos, args.csv)

    print(" ".join(top_videos.astype(str)))
