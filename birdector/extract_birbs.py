import os
import cv2
import sys
import argparse
import csv
import datetime
from pathlib import Path
import random
from collections import Counter

import numpy as np
import multiprocessing as mp

from flask import Flask, Response
from prometheus_client import Gauge, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

OBJ_THRESH = 0.5
NMS_THRESH = 0.5

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (800, 800)  # (width, height), such as (1280, 736)

from .rknn_yolov5 import CLASS_NAMES, object_detect_result
from . import rknn_yolov5

# add 1 to nice value so ffmpeg doesn't get in the way of the main processes
FFMPEG_CMD = "nice -n 1 ffmpeg -y -hide_banner -loglevel error -i {input_video} -init_hw_device rkmpp=hw -filter_hw_device hw -vf hwupload,scale_rkrga=w=864:h=486 -c:v hevc_rkmpp -qp_init 20 {output_video}"


def open_video(vname: Path) -> cv2.VideoCapture:
    """
    Opens a video file for reading and sets the video capture properties.
    Args:
        vname (str): The path to the video file.
    Returns:
        cv2.VideoCapture: The video capture object.
    Raises:
        RuntimeError: If the video file cannot be opened.
    """

    cap = cv2.VideoCapture(vname)

    # should make these params configurable
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if cap.isOpened() == False:
        raise RuntimeError("Error reading video file")

    return cap

#need to change this
def count_detections(inference_results: object_detect_result) -> dict:
    """
    Count the occurrences of each class in the given list of classes.
    Args:
        classes (list): A list of class identifiers (as integers or strings that can be converted to integers).
    Returns:
        dict: A dictionary where the keys are class names (from the global CLASS_NAMES dictionary) and the values are the counts of each class in the input list.
    """

    counts = Counter({i: 0 for i in range(len(inference_results.class_names))})
    counts.update((d.class_id for d in inference_results.detections))

    return {inference_results.class_names[k]: v for k, v in counts.items()}


class detected_bird_video:
    """
    A class to manage video processing for detected bird instances, handling writing,
    metadata collection, and compression.

    Attributes:
        vid_name (str): The generated name for the video, based on the current timestamp.
        fps (float): Frames per second for the output videos.
        width (int): Width of the video frames.
        height (int): Height of the video frames.
        minframes (int): Minimum number of frames required to retain output videos.
        p_keep (float): Probability of retaining uncompressed videos for training purposes.
        trigger_vid_path (Path): Path to save the uncompressed trigger video.
        original_vid_path (Path): Path to save the uncompressed original video.
        compressed_original_vid_path (Path): Path to save the compressed original video.
        compressed_trigger_vid_path (Path): Path to save the compressed trigger video.
        trigger_firstframe_path (Path): Path to save the first frame of the trigger video.
        original_firstframe_path (Path): Path to save the first frame of the original video.
        meta_csv (Path): Path to save the metadata CSV file for all processed videos.
        nframes (int): Total number of frames written to the video.
        total_instances (int): Total number of detected instances across all frames.
        total_class_count (dict): Dictionary containing the count of detected instances per class.
        opened (bool): Indicates whether the video writer worker is alive and processing frames.

    Methods:
        write(trigger_frame, original_frame, classes):
            Write frames and update instance counts.
        release():
            Finalize the video, write metadata, and close resources.
        isOpened():
            Check if the video writer worker is still active.
    """

    def __init__(
        self,
        output_path: Path,
        fps: float,
        width: int,
        height: int,
        minframes: int,
        p_keep: float,
        prefix: str,
    ) -> None:
        """
        Initialize the video processor for bird detection.

        Args:
            output_path (Path): Directory to store output videos and metadata.
            fps (float): Frames per second for the output videos.
            width (int): Width of the video frames.
            height (int): Height of the video frames.
            minframes (int): Minimum number of frames required to retain output videos.
            p_keep (float): Probability of retaining uncompressed videos for training purposes.
            prefix (str): Prefix for generated video names.

        Raises:
            RuntimeError: If the video writer worker fails to initialize.
        """

        # define video name
        # assumes there is <1s difference between when the frame is captured
        # and video is opened by worker
        self.vid_name = (
            f"{prefix}{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )

        # define and create output directories
        self.trigger_dir = Path(output_path) / "triggers"
        self.trigger_firstframes_dir = self.trigger_dir / "first_frames"
        self.original_dir = Path(output_path) / "originals"
        self.original_firstframes_dir = self.original_dir / "first_frames"
        self.trigger_firstframes_dir.mkdir(parents=True, exist_ok=True)
        self.original_firstframes_dir.mkdir(parents=True, exist_ok=True)

        # save video properties
        self.fps = fps
        self.width = width
        self.height = height
        self.minframes = minframes
        self.p_keep = p_keep

        # paths
        self.trigger_vid_path = self.trigger_dir / f"trigger-{self.vid_name}.mp4"
        self.original_vid_path = self.original_dir / f"original-{self.vid_name}.mp4"
        self.compressed_original_vid_path = self.original_vid_path.parent / (
            self.original_vid_path.stem + "-compressed" + self.original_vid_path.suffix
        )
        self.compressed_trigger_vid_path = self.trigger_vid_path.parent / (
            self.trigger_vid_path.stem + "-compressed" + self.trigger_vid_path.suffix
        )
        self.trigger_firstframe_path = (
            self.trigger_firstframes_dir / f"trigger-{self.vid_name}.png"
        )
        self.original_firstframe_path = (
            self.original_firstframes_dir / f"original-{self.vid_name}.png"
        )
        self.meta_csv = Path(output_path) / "meta.csv"  # stores all video metadata

        # get video writer worker ready
        self.frame_queue = mp.Queue(maxsize=100) # limit queue for when write becomes a problem
        self.worker = mp.Process(
            target=video_writer_worker,
            args=(
                self.frame_queue,
                self.fps,
                self.width,
                self.height,
                self.trigger_vid_path,
                self.compressed_trigger_vid_path,
                self.original_vid_path,
                self.compressed_original_vid_path,
                self.minframes,
                p_keep,
            ),
        )
        self.worker.start()

        # initialise output video metadata
        self.nframes = 0
        self.total_instances = 0
        self.total_class_count = {c: 0 for c in CLASS_NAMES}
        self.opened = self.worker.is_alive()
        if not self.opened:
            raise RuntimeError("Problem opening trigger or original video")

    def write(
        self, trigger_frame: np.ndarray, original_frame: np.ndarray, infres: object_detect_result
    ) -> None:
        """
        Write frames to the video and update instance statistics.

        Args:
            trigger_frame (np.ndarray): Frame corresponding to the trigger video.
            original_frame (np.ndarray): Frame corresponding to the original video.
            classes (np.ndarray): Array of class detections for the current frame.

        Raises:
            RuntimeError: If the video writer worker process terminates unexpectedly.
        """

        if self.nframes == 0:
            cv2.imwrite(self.trigger_firstframe_path, trigger_frame)
            cv2.imwrite(self.original_firstframe_path, original_frame)

        # check video writer worker is still alive
        if self.worker.exitcode:
            self.worker.join()
            raise RuntimeError(
                f"Worker has died with error code {self.worker.exitcode}"
            )

        self.frame_queue.put(
            {
                "trigger_frame": trigger_frame,
                "original_frame": original_frame,
            }
        )

        self.nframes += 1

        classes_count = count_detections(infres)
        for k, v in classes_count.items():
            self.total_class_count[k] += v
            self.total_instances += v

    def release(self) -> None:
        """
        Finalize video processing, write metadata, and release resources.

        Notes:
            - Compresses output videos using an external FFMPEG command.
            - Deletes uncompressed videos based on the `minframes` threshold and `p_keep` probability.
            - Appends metadata to an existing CSV file or creates a new one.
        """

        # close the queue so worker knows to finish
        self.frame_queue.put("DONE")

        # initialise row and header
        row = [
            "N/A",
            "N/A",
            self.compressed_original_vid_path,
            self.original_firstframe_path,
            self.compressed_trigger_vid_path,
            self.trigger_firstframe_path,
            self.nframes,
            self.total_instances / self.nframes,
        ]
        header = [
            "reference video path",
            "start time",
            "original video path",
            "original first frame path",
            "trigger video path",
            "trigger first frame path",
            "nframes",
            "average ninstance per frame",
        ]

        # add class names and data to header/row
        for k, v in self.total_class_count.items():
            header.append(k)
            row.append(str(v))

        # append to meta.csv if it exists, otherwise create a new one with headers
        if self.meta_csv.exists() and self.meta_csv.is_file():
            with self.meta_csv.open(mode="a") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(row)
        else:
            with self.meta_csv.open(mode="w") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(header)
                csvwriter.writerow(row)

        # tag as closed
        self.opened = False

    def isOpened(self) -> bool:
        """
        Check if the video writer worker process is still active.

        Returns:
            bool: True if the worker process is alive, False otherwise.
        """
        return self.opened


def inference_worker(model_path: Path, anchors_path: Path, batch_size: int, video: str, metrics_port: int, frame_queue: mp.Queue, fps: int, wait_limit: int, output_dir: Path, w: int, h: int, video_name_prefix: str):        

    videos_to_wait = []

    # start worker to broadcast data for prometheus
    manager = mp.Manager()
    shared_dict = manager.dict()
    prom_exporter_worker = mp.Process(
        target=start_flask_app,
        args=(
            shared_dict,
            video,
            metrics_port
        ),
    )
    prom_exporter_worker.start()
    
    try:

        # load anchors
        with open(anchors_path, "r") as f:
            values = [float(_v) for _v in f.readlines()]
            anchors = np.array(values).reshape(-1, 6)
        print("use anchors from '{}', which is {}".format(anchors_path, anchors), file=sys.stderr)

        model = rknn_yolov5.model(model_path, anchors, npu_cores=[0,1,2])

        # begin reading
        iframe = 0
        wait_counter = sys.maxsize
        output_video = None
        frames_loaded = 0
        frames = None
        while True:

            # read frame from input feed (expecting 10fps)
            input_frame = frame_queue.get(timeout=10)

            if type(input_frame) != np.ndarray:
                # finish off batch
                frames[frames_loaded+1:batch_size, :, :, :] = 0
                frames_loaded = 0
            else:
                if frames is None:
                    frames = np.empty((batch_size, input_frame.shape[0], input_frame.shape[1], input_frame.shape[2]), dtype=input_frame.dtype)
                frames[frames_loaded, :, :, :] = input_frame.copy()[:, :, :]
                frames_loaded = (frames_loaded + 1) % batch_size

            if frames_loaded == 0:
                # inference
                inf_res = model.inference(frames)
                shared_dict.update(count_detections(inf_res[-1]))

                for i, resi in enumerate(inf_res):
                    bird_detected = bool(resi.detections) # detections is a list
                    wait_counter = 0 if bird_detected else wait_counter + 1

                    # print number of detections to terminal
                    print(
                        f"frame {iframe} {len(resi.detections)} birds",
                        file=sys.stderr,
                    )

                    if wait_counter < wait_limit:
                        if not output_video or not output_video.isOpened():
                            output_video = detected_bird_video(
                                output_dir, 10, w, h, 50, 0.9, video_name_prefix
                            )
                        output_video.write(
                            # I think the draw api has changed
                            trigger_frame=resi.draw(conf=False),
                            original_frame=frames[i].copy(),
                            infres=resi,
                        )

                    elif output_video and output_video.isOpened():
                        output_video.release()
                        videos_to_wait.append(output_video)

                    iframe += 1

            # cleanup any finished video writer workers
            to_keep = []
            for i, v in enumerate(videos_to_wait):
                if v.worker.exitcode is not None:
                    v.worker.join()
                else:
                    to_keep.append(i)
            videos_to_wait[:] = [videos_to_wait[i] for i in to_keep]
            
            if type(input_frame) != np.ndarray:
                break
    
    finally:
        manager.shutdown()
        prom_exporter_worker.terminate()
        prom_exporter_worker.join()
        if output_video and output_video.isOpened():
            output_video.release()
        model.release()


def video_writer_worker(
    queue: mp.Queue,
    fps: int,
    w: int,
    h: int,
    trigger_path: Path,
    compressed_trigger_path: Path,
    original_path: Path,
    compressed_original_path: Path,
    minframes: int,
    p_keep: float = 0.9,
) -> None:
    """
    Processes video frames from a queue and writes them to video files. Handles compression and cleanup
    based on frame count and probabilistic retention.

    Args:
        queue (mp.Queue): Multiprocessing queue used to receive video frames or termination signal ("DONE").
        fps (int): Frames per second for the output videos.
        w (int): Width of the video frames.
        h (int): Height of the video frames.
        trigger_path (Path): Path to save the uncompressed trigger video.
        compressed_trigger_path (Path): Path to save the compressed trigger video.
        original_path (Path): Path to save the uncompressed original video.
        compressed_original_path (Path): Path to save the compressed original video.
        minframes (int): Minimum number of frames required to retain output videos.
        p_keep (float): Probability of retaining uncompressed videos for training purposes. Defaults to 0.9.

    Returns:
        None

    Raises:
        AssertionError: If compression fails or the compression command (`FFMPEG_CMD`) returns a non-zero error code.

    Notes:
        - The function processes frames from the provided queue in real-time, writing them to two separate video files.
        - Once the queue signals completion ("DONE"), the function:
            - Releases video resources.
            - Compresses the output videos using an external `FFMPEG_CMD`.
            - Deletes uncompressed videos if the total frame count is below the `minframes` threshold.
            - Retains uncompressed videos probabilistically based on `p_keep` if compression succeeds.
    """

    # open caps
    cap_trigger = cv2.VideoWriter(
        trigger_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    cap_original = cv2.VideoWriter(
        original_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    # start reading frames from master
    nframes = 0
    while True:
        res = queue.get(timeout=10)  # expecting frames to come in 10fps

        # handle result
        if res == "DONE":
            # release output video capture objects
            cap_trigger.release()
            cap_original.release()

            # compress output videos, removing if they don't meet the minframes threshold + 1s worth of frames
            if nframes < minframes + fps:
                trigger_path.unlink(missing_ok=True)
                original_path.unlink(missing_ok=True)
            else:
                err = os.system(
                    f"""{FFMPEG_CMD.format(input_video=trigger_path, output_video=compressed_trigger_path)}
                        {FFMPEG_CMD.format(input_video=original_path, output_video=compressed_original_path)}
                    """
                )
                # randomly keep fullres video for training
                if random.random() < p_keep:
                    trigger_path.unlink(missing_ok=True)
                    original_path.unlink(missing_ok=True)
                assert err == 0, "Error compressing output videos."
            # break loop
            break
        else:
            cap_trigger.write(res["trigger_frame"])
            cap_original.write(res["original_frame"])
            nframes += 1

# Define Prometheus Gauges for metrics
app = Flask(__name__)
registry = CollectorRegistry()
bird_gauge = Gauge(
    "bird_species_count", 
    "Number of birds per species", 
    ["species", "video"], 
    registry=registry
)

# Flask route to serve metrics
@app.route('/metrics')
def metrics():
    """Endpoint to expose Prometheus metrics."""
    return Response(generate_latest(registry), content_type=CONTENT_TYPE_LATEST)

def start_flask_app(shared_dict, video, port):
    """Start the Flask app and periodically update Prometheus metrics."""
    @app.before_request
    def update_metrics():
        """Update bird species metrics before each request."""
        for species in CLASS_NAMES:
            bird_gauge.labels(species=species, video=video).set(shared_dict[species])
    app.run(host="0.0.0.0", port=port)

def main():
    parser = argparse.ArgumentParser(description="Detect birds from a video feed.")
    # basic params
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="model path, could be .pt or .rknn file",
    )

    # data params
    parser.add_argument(
        "--video",
        "-v",
        type=Path,
        default=Path("/dev/video0"),
        help="Video to watch. Can be either video device index, e.g. 0, or video file e.g. video.mkv.",
    )
    parser.add_argument(
        "--anchors",
        type=str,
        default="../model/anchors_yolov5.txt",
        help="target to anchor file, only yolov5, yolov7 need this param",
    )

    # output
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("."),
        help="Output directory to save inference results.",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=0,
        help="Maximum number of frames to record for.",
    )
    parser.add_argument(
        "--video-name-prefix",
        "-p",
        type=str,
        default="",
        help="Prefix for the output video files.",
    )
    parser.add_argument(
        "--metrics-port",
        "-t",
        type=int,
        default=9093,
        help="Port for metrics exporter to listen on."
    )
    parser.add_argument(
        "--npu-cores",
        "-c",
        type=str,
        default="-1",
        help="RK3588 NPU cores to use e.g. 0,1 will use cores 0 and 1. Passing -1 will choose 1 random core.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size of images to pass to RKNPU for inference. Must match with input RKNN model.",
    )

    args = parser.parse_args()

    # init model
    try:
        npu_cores = [int(n) for n in args.npu_cores.split(",")]
    except TypeError:
        raise TypeError("Selected NPU cores must be integers.")
    #model = rknn_yolov5.model(args.model_path, anchors, npu_cores=npu_cores)

    # open video and define some metadata
    cap = open_video(args.video)
    fps = 10.0  # temporary fix
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames == 0:
        args.max_frames = sys.maxsize
    total_frames = (
        args.max_frames if total_frames < 0 else min(total_frames, args.max_frames)
    )
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wait_limit = 50
    print(
        f"""INPUT VIDEO: {args.video}
    Resolution: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}x{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}
    FPS:        {fps}
OUTPUT DIRECTORY: result/{{originals,triggers}}
DETECTION MODEL: {args.model_path}
WAIT LIMIT: {wait_limit} frames
MAX FRAMES: {args.max_frames} frames

Saving detections to MySQL database: http://localhost:3306, grafana_data.metrics ...

Beginning processing...
""",
    file=sys.stderr)

    # start try except finally clause for handling of multiprocessing
    try:
        
        # get inference worker ready
        frame_queue = mp.Queue(maxsize=100) # limit queue for when write becomes a problem
        worker = mp.Process(
            target=inference_worker,
            args=(
                args.model_path, 
                args.anchors,
                args.batch_size, 
                args.video, 
                args.metrics_port, 
                frame_queue, 
                fps, 
                wait_limit, 
                args.output_dir, 
                w, 
                h, 
                args.video_name_prefix,
            ),
        )
        worker.start()

        # begin reading
        suc = True
        iframe = 0
        while suc and iframe < total_frames:

            # read frame from input feed
            suc, frame = cap.read()

            if not suc:
                raise ("Error reading frame from input feed.")

            frame_queue.put(frame.copy())

            iframe += 1

    finally:
        # release cap (done first so any other process can use it asap)
        cap.release()

        frame_queue.put("DONE")

if __name__ == "__main__":
    main()
