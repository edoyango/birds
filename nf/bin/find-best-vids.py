#!/usr/bin/env python3
import pandas as pd
import argparse

def calculate_simpsons_diversity(species_counts):
    """
    Calculate Simpson's Diversity Index for a video.
    :param species_counts: List of species counts for the video.
    :return: Simpson's Diversity Index.
    """
    total_count = sum(species_counts)
    if total_count == 0:
        return 0  # Handle cases where no species are present
    proportions = [count / total_count for count in species_counts]
    return 1 - sum(p ** 2 for p in proportions)

def main(args):
    # Load CSV file
    csv_path = args.csv
    data = pd.read_csv(csv_path, header=0)  # No header assumed; adjust if needed

    # Extract relevant columns
    video_column = "trigger video path"  # 5th column (0-based index)
    frames_column = "nframes"  # 7th column (0-based index)
    species_columns = ["Dove", "Myna", "Wattlebird", "blackbird", "currawong", "magpie", "sparrow", "starling"]  # 9th to 16th columns (0-based index)

    # Filter rows by minimum duration
    min_frames = args.duration*10 # assume 10 fps
    filtered_data = data[data[frames_column] >= min_frames].copy()

    # Calculate Simpson's Diversity Index for each video
    filtered_data['simpsons_diversity'] = filtered_data[species_columns].apply(
        lambda row: calculate_simpsons_diversity(row.tolist()), axis=1
    )

    # Sort by Simpson's Diversity Index in descending order
    sorted_data = filtered_data.sort_values(by='simpsons_diversity', ascending=False)

    # Select top n videos
    top_videos = sorted_data[video_column].head(args.num_videos)

    # Print the video names space-delimited
    print(" ".join(top_videos.astype(str)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select videos based on diversity and duration.")
    parser.add_argument("--duration", type=int, required=True, help="Minimum duration in seconds.", default=11)
    parser.add_argument("--num-videos", type=int, required=True, help="Number of videos to select.", default=4)
    parser.add_argument("--csv", type=str, default="meta.csv", help="Path to the input CSV file.")
    args = parser.parse_args()
    main(args)

