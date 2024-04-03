#!/usr/bin/env python3

from PIL import Image
import os


def stack_images(image_paths, output_filename, lower):
    images = [Image.open(path) for path in image_paths]

    # Get widths and total height
    widths, heights = zip(*(i.size for i in images))
    total_height = sum(heights)
    max_width = max(widths)

    # Create a new image with the calculated dimensions
    new_image = Image.new("RGB", (max_width, total_height))

    # Paste the images, positioning them vertically
    y_offset = 0
    for img in images:
        new_image.paste(img, (0, y_offset))
        y_offset += img.size[1]

    res = new_image.size
    resized_image = new_image.resize(
        (int(res[0] / lower), int(res[1] / lower)), Image.LANCZOS
    )

    # Save the stacked image
    resized_image.save(output_filename)


if __name__ == "__main__":

    import argparse, random

    parser = argparse.ArgumentParser(
        "stack_sample_images.py",
        description="Collects a sample of images in a folder and stacks them on top of each other.",
    )

    parser.add_argument("input_path", help="Folder to sample from.")
    parser.add_argument("output_img", help="Path to image to save.")
    parser.add_argument("-n", help="Number of samples.", default=10)
    parser.add_argument(
        "-l",
        help="Reduce resolution by the given factor. E.g., 2 will half the resolution.",
        default=2,
    )

    args = parser.parse_args()

    input_path = args.input_path

    all_image_files = [
        os.path.join(input_path, file)
        for file in os.listdir(input_path)
        if file.endswith((".jpg", ".png"))
    ]  # Taking up to 10 images

    image_sample = random.sample(all_image_files, args.n)

    stack_images(image_sample, args.output_img, args.l)
