"""Tests for birdector.post_process.find_best_vids: Simpson's diversity
scoring and fps-aware duration filtering."""

import csv

import pytest

from birdector.post_process import find_best_vids


@pytest.mark.parametrize(
    "counts,expected",
    [
        ([0, 0, 0], 0),  # no detections at all
        ([5, 0, 0], 0.0),  # single species -> zero diversity
        ([1, 1], 0.5),  # two equally-common species
    ],
)
def test_calculate_simpsons_diversity(counts, expected):
    assert find_best_vids.calculate_simpsons_diversity(counts) == pytest.approx(expected)


def _write_meta_csv(path, nframes: int):
    species = list(find_best_vids.WEIGHTS.keys())
    header = [
        "reference video path", "start time", "original video path",
        "original first frame path", "trigger video path",
        "trigger first frame path", "nframes", "average ninstance per frame",
    ] + species
    row = ["na", "na", "orig.mp4", "of.png", "trigger.mp4", "tf.png", nframes, 1.0] + [1] * len(species)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)


def test_default_fps_matches_extract_birbs_default(tmp_path):
    """duration is filtered as `duration * fps` frames; the default fps
    here must stay in sync with extract_birbs.py's own --fps default,
    since that's what a meta.csv's nframes column was actually recorded
    against unless a caller passes --fps explicitly."""
    csv_path = tmp_path / "meta.csv"
    _write_meta_csv(csv_path, nframes=100)

    # 100 frames at the assumed 10fps default is 10s of footage; asking for
    # an 11s minimum should exclude it
    assert find_best_vids.main(duration=11, num_videos=5, csv_path=csv_path) == []


def test_fps_parameter_changes_the_frame_threshold(tmp_path):
    csv_path = tmp_path / "meta.csv"
    _write_meta_csv(csv_path, nframes=100)

    # at 5fps, 100 frames is 20s of footage - now well over an 11s minimum
    result = find_best_vids.main(duration=11, num_videos=5, csv_path=csv_path, fps=5.0)
    assert len(result) == 1
