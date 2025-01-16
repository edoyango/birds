#!/usr/bin/env python3

from datetime import datetime
import find_best_vids
from pathlib import Path
import cv2
import os
import tempfile
from stack_gifs import stack_gifs
from parse_instances import parse_instances
from send_birb_summary import parse_csv_and_send

def vids2gif(best_vids: list[Path], vid_duration: float = 15) -> list[Path]:

    gifs2stack = []
    for vid in best_vids:
        print(vid)
        # get starting time of cut
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            raise RuntimeError("Failed to open input video!")
        
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)
        start_time = max(0, (duration - vid_duration)/2)

        output_gif = Path(tempfile.gettempdir()) / (vid.stem + ".gif")

        # resize and convert to gif
        os.system(f"ffmpeg -y -ss {start_time} -t {vid_duration} -i {vid} -vf 'scale=480:-1:flags=lanczos' {output_gif}")

        # save gif for stacking
        gifs2stack.append(output_gif)

    return gifs2stack

def main(date: str, output_dir: Path, vid_duration: float, nvideos: int, input_csv: Path, mailing_list_csv: Path) -> None:

    best_vids = find_best_vids.main(vid_duration, nvideos, input_csv)

    gifs2stack = vids2gif(best_vids, vid_duration)

    stacked_gif = Path(tempfile.gettempdir()) / "stacked.gif"

    stack_gifs(gifs2stack, stacked_gif)

    output_gif = output_dir / 'sample.gif'
    output_panel = output_dir / 'panel.png'

    os.system(f"gifsicle -O3 --lossy=35 -i {stacked_gif} --colors 128 -o {output_gif}")

    os.system(f"curl -o {output_panel} 'admin:orangepi@grafana:3000/render/d-solo/de8twne7u2r5sc?orgId=1&from=now-12h&to=now&panelId=10&width=720&height=480&tz=Australia%2FSydney'")

    html_bird_list = parse_instances(input_csv)

    pwd = os.getenv("GMAIL_APP_PWD")
    if not pwd:
        raise RuntimeError(
            'Gmail password not found in "GMAIL_APP_PWD" environment variable.'
        )

    parse_csv_and_send(
        "Birb Watcher",
        "eds.birb.watcher@gmail.com",
        pwd,
        mailing_list_csv,
        f"Birb watcher update - {date}",
        f"""<html>
<p>Hi,</p>
<p>I've been recording videos all day, and across all video frames I saw:</p>
<ul>
{html_bird_list}
</ul>
<p>Here's a heatmap of all the birds I saw throughout the day!</p>
<img src=\"cid:{{image_cid1}}\">
<p>And here's one of the videos with birds:</p>
<img src=\"cid:{{image_cid0}}\">
<p>Hope you have a great day!</p>
<p>Regards,<br>Ed's Birb Watcher</p>
</body>
</html>""",
        [output_panel, output_gif]
    )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--date", default=datetime.now().strftime("%Y-%m-%d"), type=str)
    parser.add_argument("-t", "--video-duration", type=float, default=15)
    parser.add_argument("-n", "--nvideos", type=int, default=5)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("mailing_list_csv", type=Path)

    args = parser.parse_args()

    main(args.date, args.output_dir, args.video_duration, args.nvideos, args.input_csv, args.mailing_list_csv)