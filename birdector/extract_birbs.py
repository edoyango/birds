import os
import cv2
import sys
import argparse
import csv
import datetime
import logging
import signal
import subprocess
import threading
from pathlib import Path
import random
from collections import Counter
from queue import Empty as QueueEmpty, Full as QueueFull

import numpy as np
import multiprocessing as mp

from flask import Flask, Response
from prometheus_client import (
    Gauge,
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
)

from .rknn_yolov5 import CLASS_NAMES, object_detect_result
from . import rknn_yolov5
from .frame_pool import FrameBufferPool, FramePoolExhausted

logger = logging.getLogger(__name__)

# max frames in flight between the reader and inference processes; also
# sizes the shared-memory frame buffer pool (see frame_pool.py)
FRAME_QUEUE_MAXSIZE = 100


def _ffmpeg_compress_cmd(input_video: Path, output_video: Path) -> list:
    """Build the argv list for compressing one video with RK3588 HW acceleration."""
    return [
        # nice'd so ffmpeg doesn't get in the way of the main processes
        "nice", "-n", "1",
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_video),
        "-init_hw_device", "rkmpp=hw",
        "-filter_hw_device", "hw",
        "-vf", "hwupload,scale_rkrga=w=864:h=486",
        "-c:v", "hevc_rkmpp",
        "-qp_init", "20",
        str(output_video),
    ]


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
    requested_props = (
        ("CAP_PROP_FOURCC", cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV")),
        ("CAP_PROP_FRAME_WIDTH", cv2.CAP_PROP_FRAME_WIDTH, 1280),
        ("CAP_PROP_FRAME_HEIGHT", cv2.CAP_PROP_FRAME_HEIGHT, 720),
    )
    for name, prop_id, value in requested_props:
        # set() returning True only means the driver accepted the request,
        # not that it took effect - some V4L2 devices silently ignore
        # unsupported values, so read the property back to confirm
        if not cap.set(prop_id, value):
            logger.warning("%s could not be set to %s on %s", name, value, vname)
        elif cap.get(prop_id) != value:
            logger.warning(
                "%s was set to %s on %s, but the device reports %s",
                name,
                value,
                vname,
                cap.get(prop_id),
            )

    if cap.isOpened() == False:
        raise RuntimeError("Error reading video file")

    return cap


# need to change this
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


class DetectedBirdVideo:
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
        # each camera gets its own metadata file: two containers sharing one
        # output-dir (as happens when cameras are grouped by day) must not
        # append to the same meta.csv without coordination.
        meta_name = f"meta-{prefix.rstrip('-_')}.csv" if prefix else "meta.csv"
        self.meta_csv = Path(output_path) / meta_name

        # get video writer worker ready
        self.frame_queue = mp.Queue(
            maxsize=100
        )  # limit queue for when write becomes a problem
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
        self,
        trigger_frame: np.ndarray,
        original_frame: np.ndarray,
        infres: object_detect_result,
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

        # check video writer worker is still alive. exitcode is None while
        # running and 0 on a clean exit, so a plain truthiness check misses a
        # worker that exited cleanly but early (e.g. it gave up on a stalled
        # frame supply) - that must still be treated as "no longer consuming
        # frames", or frame_queue fills up and this call blocks forever.
        if self.worker.exitcode is not None:
            self.worker.join()
            raise RuntimeError(
                f"Worker for {self.trigger_vid_path} is no longer running "
                f"(exit code {self.worker.exitcode})"
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

        # append to meta_csv, writing the header first only if the file is new/empty.
        # opened purely in append mode (never "w") so a concurrent writer can only
        # ever add rows, never truncate another writer's data.
        write_header = not (self.meta_csv.exists() and self.meta_csv.stat().st_size > 0)
        with self.meta_csv.open(mode="a", newline="") as f:
            csvwriter = csv.writer(f)
            if write_header:
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


def inference_worker(
    model_path: Path,
    anchors_path: Path,
    batch_size: int,
    video: str,
    metrics_port: int,
    frame_queue: mp.Queue,
    frame_pool: FrameBufferPool,
    fps: int,
    wait_limit: int,
    output_dir: Path,
    w: int,
    h: int,
    video_name_prefix: str,
    npu_cores: list,
):
    """
    Performs batched inference on video frames using an RKNN YOLOv5 model, writes results to video files,
    and exports detection metrics via a Prometheus-compatible Flask server.

    Args:
        model_path (Path): Path to the compiled RKNN model.
        anchors_path (Path): Path to the file containing anchor box definitions.
        batch_size (int): Number of frames to process in each inference batch.
        video (str): Identifier for the input video source (used for metrics labeling).
        metrics_port (int): Port to expose Prometheus metrics via a Flask app.
        frame_queue (mp.Queue): Multiprocessing queue that receives frame_pool buffer indices (int),
            or the string "DONE" as a termination signal, from a video reader.
        frame_pool (FrameBufferPool): Shared-memory pool the actual frame pixel data lives in - only
            its buffer index travels through frame_queue.
        fps (int): Expected frame rate (used in video writer setup).
        wait_limit (int): Maximum number of consecutive frames without detections before video writer is stopped.
        output_dir (Path): Directory to store output video files containing bird detections.
        w (int): Width of the video frames.
        h (int): Height of the video frames.
        video_name_prefix (str): Prefix to prepend to all output video filenames.
        npu_cores (list): List of NPU core indices to use during inference.

    Returns:
        None

    Notes:
        - Starts a child process to serve Prometheus metrics for detection statistics.
        - Loads anchors and initializes the RKNN model for inference.
        - Batches incoming frames from `frame_queue`/`frame_pool` and runs detection inference when a
          full batch is collected.
        - For each frame, draws detections and writes annotated frames to a video file if birds are detected.
        - Automatically handles cleanup of video writers and Prometheus exporter process on completion.
        - Stops processing when the "DONE" sentinel is received from frame_queue.
        - Updates Prometheus Gauge objects directly as each batch is processed; a background thread
          in this same process serves them over HTTP for scraping.
    """

    # main()'s SIGTERM/SIGINT handler is inherited across fork - reset to
    # the default disposition here, before this process forks any children
    # of its own (a video_writer_worker per recording). Without this,
    # terminate() calls elsewhere in the pipeline would just run the
    # inherited handler instead of actually killing the target process, and
    # a .join() right after would hang forever.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    videos_to_wait = []

    # serve Prometheus metrics on a background thread in this process rather
    # than a separate one: it only needs to read counters that this process
    # already computes, so a thread avoids both a second process and the
    # multiprocessing.Manager server (its own process plus a pickled-proxy
    # round trip for every dict access) that would otherwise be needed to
    # share data with it. daemon=True since Flask's dev server has no clean
    # programmatic shutdown - it exits with the process instead.
    metrics_thread = threading.Thread(
        target=start_flask_app, args=(metrics_port,), daemon=True
    )
    metrics_thread.start()

    # load anchors
    with open(anchors_path, "r") as f:
        values = [float(_v) for _v in f.readlines()]
        anchors = np.array(values).reshape(-1, 6)
    logger.info("use anchors from '%s', which is %s", anchors_path, anchors)

    model = rknn_yolov5.model(model_path, anchors, npu_cores=npu_cores)

    # begin reading
    iframe = 0
    wait_counter = sys.maxsize
    output_video = None
    frames_loaded = 0
    frames = None
    while True:

        # read the next buffer index from the reader (expecting 10fps) - the
        # actual frame pixels live in frame_pool's shared memory, not here
        try:
            item = frame_queue.get(timeout=10)
        except QueueEmpty:
            # no frame for 10s at an expected ~10fps means the reader process
            # died or the camera feed stalled. Exit (after the same cleanup
            # the normal termination path does, below) instead of hanging
            # here indefinitely - main()'s put() will notice this process is
            # gone and fail loudly so the container can actually restart.
            logger.error(
                "No frames received for 10s; assuming the frame reader died "
                "or the camera feed stalled. Shutting down."
            )
            break

        is_terminator = item == "DONE"

        if is_terminator:
            # flush whatever's been loaded into this partial batch, rather
            # than padding it out to batch_size with zeroed/stale frames and
            # running inference (and potentially recording) on black frames
            n_valid = frames_loaded
            frames_loaded = 0
        else:
            buf_idx = item
            input_frame = frame_pool.view(buf_idx)
            if frames is None:
                frames = np.empty(
                    (
                        batch_size,
                        input_frame.shape[0],
                        input_frame.shape[1],
                        input_frame.shape[2],
                    ),
                    dtype=input_frame.dtype,
                )
            # copies out of shared memory into our own batch buffer, so it's
            # safe to release the shared buffer immediately after
            frames[frames_loaded, :, :, :] = input_frame
            frame_pool.release(buf_idx)
            frames_loaded = (frames_loaded + 1) % batch_size
            n_valid = batch_size if frames_loaded == 0 else None

        if n_valid:
            # a full batch, or (only on shutdown) the final partial one
            batch = frames[:n_valid]
            inf_res = model.inference(batch)

            # aggregate detections across every frame in the batch, not just
            # the last one - with batch_size > 1 the old code only ever
            # reflected 1 in every batch_size inferred frames in the
            # exported bird_species_count metric
            batch_counts = Counter()
            for resi in inf_res:
                batch_counts.update(count_detections(resi))
            for species, count in batch_counts.items():
                bird_gauge.labels(species=species, video=video).set(count)

            for i, resi in enumerate(inf_res):
                bird_detected = bool(resi.detections)  # detections is a list
                wait_counter = 0 if bird_detected else wait_counter + 1

                logger.debug("frame %d %d birds", iframe, len(resi.detections))

                if wait_counter < wait_limit:
                    try:
                        if not output_video or not output_video.isOpened():
                            output_video = DetectedBirdVideo(
                                output_dir, fps, w, h, 50, 0.9, video_name_prefix
                            )
                        output_video.write(
                            # I think the draw api has changed
                            trigger_frame=resi.draw(conf=False),
                            original_frame=frames[i].copy(),
                            infres=resi,
                        )
                    except RuntimeError:
                        # writer process for this recording died or failed to
                        # start; drop it and let the next detected frame open
                        # a fresh recording instead of taking down the whole
                        # camera's inference process over one bad recording
                        logger.exception(
                            "Lost the video writer for the current recording; "
                            "starting a new one on the next detection."
                        )
                        output_video = None

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

        if is_terminator:
            break

    if output_video and output_video.isOpened():
        output_video.release()
    model.release()
    frame_pool.close()
    # metrics_thread is a daemon thread serving Flask's dev server, which has
    # no clean programmatic shutdown - it's simply abandoned when this
    # process exits right after this


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

    Notes:
        - The function processes frames from the provided queue in real-time, writing them to two separate video files.
        - It finalizes (releases, compresses, cleans up) either when the queue signals completion
          ("DONE") or after 10s of no new frames, so a stalled feed can't leave a video stuck
          uncompressed and un-cataloged forever.
        - Compresses trigger and original independently via ffmpeg; a failure on one is logged and
          does not prevent the other from being compressed.
        - Deletes uncompressed videos if the total frame count is below the `minframes` threshold.
        - Otherwise, discards the full-resolution originals with probability `1 - p_keep` (i.e. keeps
          them for training with probability `p_keep`), but never discards a video whose compression
          failed.
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

    def finalize(nframes: int) -> None:
        """Release the writers and compress (or discard) whatever was captured."""
        cap_trigger.release()
        cap_original.release()

        # compress output videos, removing if they don't meet the minframes threshold + 1s worth of frames
        if nframes < minframes + fps:
            trigger_path.unlink(missing_ok=True)
            original_path.unlink(missing_ok=True)
            return

        # compress trigger and original independently so one failing doesn't
        # mask the other's exit code (the old code ran both through a single
        # os.system() call and only ever saw the last command's status)
        compressed_ok = {}
        for name, src, dst in (
            ("trigger", trigger_path, compressed_trigger_path),
            ("original", original_path, compressed_original_path),
        ):
            try:
                subprocess.run(
                    _ffmpeg_compress_cmd(src, dst),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                compressed_ok[name] = True
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                compressed_ok[name] = False
                detail = e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) and e.stderr else e
                logger.error("Failed to compress %s video %s: %s", name, src, detail)

        # randomly keep the full-resolution original for training (with
        # probability p_keep), but never delete the only copy of a video
        # whose compression failed
        discard_fullres = random.random() >= p_keep
        if discard_fullres and compressed_ok["trigger"]:
            trigger_path.unlink(missing_ok=True)
        if discard_fullres and compressed_ok["original"]:
            original_path.unlink(missing_ok=True)

    # start reading frames from master
    nframes = 0
    try:
        while True:
            try:
                res = queue.get(timeout=10)  # expecting frames to come in 10fps
            except QueueEmpty:
                # camera feed stalled mid-recording: finalize what we have
                # instead of leaving an uncompressed, un-cataloged video behind
                logger.warning(
                    "No frames received for 10s while recording %s; "
                    "finalizing the %d frame(s) captured so far.",
                    trigger_path,
                    nframes,
                )
                finalize(nframes)
                return

            # handle result
            if res == "DONE":
                finalize(nframes)
                return
            else:
                cap_trigger.write(res["trigger_frame"])
                cap_original.write(res["original_frame"])
                nframes += 1
    except Exception:
        logger.exception("video_writer_worker for %s failed unexpectedly", trigger_path)
        cap_trigger.release()
        cap_original.release()
        raise


# Define Prometheus Gauges for metrics
app = Flask(__name__)
registry = CollectorRegistry()
bird_gauge = Gauge(
    "bird_species_count",
    "Number of birds per species",
    ["species", "video"],
    registry=registry,
)


# Flask route to serve metrics
@app.route("/metrics")
def metrics():
    """Endpoint to expose Prometheus metrics."""
    return Response(generate_latest(registry), content_type=CONTENT_TYPE_LATEST)


def start_flask_app(port):
    """Serve the /metrics endpoint. bird_gauge is updated directly by inference_worker
    as each batch is processed; this just exposes whatever it currently holds.

    Note: this is Werkzeug's development server, not a production WSGI server
    (waitress, gunicorn, etc.) - acceptable here since it only serves periodic,
    low-volume Prometheus scrapes rather than public traffic. threaded=True so
    one slow/stalled scraping client can't block others from being served.
    """
    app.run(host="0.0.0.0", port=port, threaded=True)


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
        help="Port for metrics exporter to listen on.",
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
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("BIRDECTOR_LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Defaults to the "
        "BIRDECTOR_LOG_LEVEL env var, or INFO.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frame rate of the input feed. cv2.CAP_PROP_FPS is unreliable "
        "on some V4L2 devices, so this is a fixed value rather than being "
        "read from the capture device; override if your source differs "
        "from the 10fps default.",
    )

    args = parser.parse_args()

    # configured here (before any worker processes are forked) so child
    # processes inherit the same logging setup
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
        stream=sys.stderr,
    )

    # init model
    try:
        npu_cores = [int(n) for n in args.npu_cores.split(",")]
    except TypeError:
        raise TypeError("Selected NPU cores must be integers.")

    # open video and define some metadata
    cap = open_video(args.video)
    fps = args.fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames == 0:
        args.max_frames = sys.maxsize
    total_frames = (
        args.max_frames if total_frames < 0 else min(total_frames, args.max_frames)
    )
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wait_limit = 50
    logger.info(
        "INPUT VIDEO: %s | resolution: %dx%d | fps: %s | output dir: %s | "
        "model: %s | wait limit: %d frames | max frames: %s | "
        "metrics: http://0.0.0.0:%d/metrics | beginning processing...",
        args.video,
        w,
        h,
        fps,
        args.output_dir,
        args.model_path,
        wait_limit,
        args.max_frames,
        args.metrics_port,
    )

    # trap SIGTERM/SIGINT so `docker stop` (or Ctrl-C) stops reading new
    # frames and lets the inference worker finish flushing/compressing its
    # current recording instead of being killed mid-write. Without a
    # handler, SIGTERM's default disposition kills the process immediately -
    # no Python code, including cleanup, gets to run at all.
    shutdown_requested = threading.Event()

    def _handle_shutdown_signal(signum, _frame):
        logger.info(
            "Received signal %d; finishing up and shutting down.", signum
        )
        shutdown_requested.set()

    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)

    # get inference worker ready. Frame pixel data travels through
    # frame_pool's shared memory instead of being pickled through
    # frame_queue on every frame - the queue only ever carries a small
    # buffer index (or the "DONE" sentinel), sized to match the pool so it
    # can never itself become a tighter bottleneck than the pool already is.
    frame_pool = FrameBufferPool(shape=(h, w, 3), dtype=np.uint8, n_buffers=FRAME_QUEUE_MAXSIZE)
    frame_queue = mp.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
    worker = mp.Process(
        target=inference_worker,
        args=(
            args.model_path,
            args.anchors,
            args.batch_size,
            args.video,
            args.metrics_port,
            frame_queue,
            frame_pool,
            fps,
            wait_limit,
            args.output_dir,
            w,
            h,
            args.video_name_prefix,
            npu_cores
        ),
    )
    worker.start()

    try:
        # begin reading
        suc = True
        iframe = 0
        while suc and iframe < total_frames and not shutdown_requested.is_set():

            # read frame from input feed
            suc, frame = cap.read()

            if not suc:
                raise RuntimeError("Error reading frame from input feed.")

            # both puts below would otherwise block forever if the inference
            # worker died and stopped draining them - a dead consumer would
            # hang this process indefinitely instead of exiting, silently
            # defeating the container's restart policy.
            while True:
                try:
                    buf_idx = frame_pool.put(frame, timeout=10)
                    break
                except FramePoolExhausted:
                    if not worker.is_alive():
                        raise RuntimeError(
                            f"Inference worker process died (exit code "
                            f"{worker.exitcode}); aborting so the container "
                            "can restart."
                        )
                    logger.warning(
                        "Frame buffer pool has been exhausted for 10s; the "
                        "inference worker may be falling behind."
                    )

            while True:
                try:
                    frame_queue.put(buf_idx, timeout=10)
                    break
                except QueueFull:
                    if not worker.is_alive():
                        raise RuntimeError(
                            f"Inference worker process died (exit code "
                            f"{worker.exitcode}); aborting so the container "
                            "can restart."
                        )
                    logger.warning(
                        "Frame queue has been full for 10s; the inference "
                        "worker may be falling behind."
                    )

            iframe += 1

        # release cap (done first so any other process can use it asap)
        cap.release()

        frame_queue.put("DONE")

        if shutdown_requested.is_set():
            # give the inference worker (and, transitively, any in-progress
            # video_writer_worker) a real chance to flush and compress the
            # current recording before this process exits. Bounded so a
            # stuck worker can't block shutdown forever - pair with a docker
            # stop_grace_period long enough to cover this wait, or Docker
            # will SIGKILL everything before it elapses.
            logger.info("Waiting up to 25s for the inference worker to finish...")
            worker.join(timeout=25)
            if worker.is_alive():
                logger.warning(
                    "Inference worker did not finish within the shutdown "
                    "grace period; it may be terminated abruptly."
                )
        else:
            worker.join()
    finally:
        cap.release()  # safe to call again if already released above
        frame_pool.close()
        if not worker.is_alive():
            # only unlink once nothing (including this process) still needs
            # the shared memory; if the worker is still alive here (e.g. the
            # graceful-shutdown join above timed out) we leave it allocated
            # rather than risk unlinking out from under it
            frame_pool.unlink()


if __name__ == "__main__":
    main()
