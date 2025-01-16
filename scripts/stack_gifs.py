#!/usr/bin/env python3
from PIL import Image
from pathlib import Path


def stack_gifs(gif_paths: list[Path], output_path: Path):
    """
    Stack multiple GIFs into a single GIF with all frames vertically aligned.

    This function takes a list of GIF file paths, ensures they all have the same number of frames,
    and stacks them vertically frame by frame into a single GIF. The resulting stacked GIF is saved
    to the specified output path.

    Parameters:
    gif_paths (list[Path]): List of paths to the input GIF files.
    output_path (Path): Path to save the resulting stacked GIF.

    Returns:
    None
    """

    gifs = [Image.open(gif) for gif in gif_paths]

    # Ensure all GIFs have the same number of frames
    num_frames = min(gif.n_frames for gif in gifs)

    frames = []
    for frame_idx in range(num_frames):
        # Create a new blank image for each frame
        widths, heights = zip(*(gif.size for gif in gifs))
        total_height = sum(heights)
        max_width = max(widths)
        new_frame = Image.new("RGBA", (max_width, total_height))

        # Paste each gif frame into the new image
        current_height = 0
        for gif in gifs:
            gif.seek(frame_idx)
            frame = gif.convert("RGBA")
            new_frame.paste(frame, (0, current_height))
            current_height += gif.size[1]

        frames.append(new_frame)

    # Save all frames as a new GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=gifs[0].info["duration"],
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output_gif", type=Path)
    parser.add_argument("input_gifs", nargs="+", type=Path)
    args = parser.parse_args()

    stack_gifs(args.input_gifs, args.output_gif)
