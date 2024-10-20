#!/usr/bin/env python3

import cv2
import numpy as  np
import os
import datetime
import sys
import torch
import copy
import ultralytics
import pathlib
import csv
from collections import Counter

def open_video(vname):

    cap = cv2.VideoCapture(vname)

    if (cap.isOpened() == False):  
        raise RuntimeError("Error reading video file")

    return cap

#def get_model(model_path):
#
#    model = ultralytics.YOLO(model_path, task="detect")
#
#    for k, v in model.names.items():
#        if v == "bird":
#            bird_class = k
#    return model, bird_class

def create_output_video(vname, cap):

    out = cv2.VideoWriter(
        vname,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(cap.get(cv2.CAP_PROP_FPS)),
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    )

    if not out.isOpened():
        raise RuntimeError(f"Failed to open {vname}")
    else:
        return out

def add_seconds_to_timestring(timestring: str, seconds: int):
    time_obj = datetime.datetime.strptime(timestring, "%H-%M-%S")
    time_diff = datetime.timedelta(seconds=seconds)
    return (time_obj + time_diff).strftime("%H-%M-%S")
        
def create_new_fname(vidpath, frame_idx, fps, outdir=None, prefix=""):
    vid_dir = os.path.dirname(vidpath)
    split_vidname = os.path.basename(vidpath).split(".")
    ext = split_vidname[-1]
    time_str = ".".join(split_vidname[:-1])
    outfile = f"{prefix}{add_seconds_to_timestring(time_str, frame_idx/fps)}.{ext}"
    if outdir:
        outpath = os.path.join(outdir, outfile)
    else:
        outpath = os.path.join(vid_dir, outfile)
    return outpath

@torch.no_grad()
def subclassify(detection_result, cls_model):

    ret = copy.deepcopy(detection_result)

    # clone results so classes can be edited
    tmp_box_data = ret.boxes.data.clone()

    # extract the detected parts of the image and classify
    res = cls_model(
        [
            ret.orig_img[box[1]:box[3], box[0]:box[2], :]
            for box in tmp_box_data.round().int()
        ],
        imgsz=64,
        verbose=False
    )

    # replace subclass if classifier is highly confident
    if res[0].probs.top1conf.item() > 0.8:
        # extract existing classes
        nexisting_names = len(ret.names)
        tmp_box_data[:, -1] = torch.Tensor([i.probs.top1 + nexisting_names for i in res])
        ret.boxes.data = tmp_box_data
        for k, v in res[0].names.items():
            ret.names[k  + nexisting_names] = v

    return ret

def count_detections(detection_result):

    counts = {}
    for i in detection_result.boxes.cls.cpu().int():
        ii = int(i)
        if counts.get(ii):
            counts[ii] += 1
        else:
            counts[ii] = 1
    
    return {detection_result.names[k]: v for k, v in counts.items()}

class detected_birb_vid:

    def __init__(self, reference_cap, reference_vidname, frame_offset, outdir, minframes, instances_names):

        # check that reference_cap is open
        if not reference_cap.isOpened():
            raise RuntimeError("Input reference video capture not opened!")
        
        # save reference cap meta data
        self.fps = float(reference_cap.get(cv2.CAP_PROP_FPS))
        self.width = int(reference_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(reference_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.minframes = minframes

        # create paths for output original/trigger subvideos/images
        rv = pathlib.Path(reference_vidname)
        self.reference_vidpath = rv
        outpath = pathlib.Path(outdir)
        if not rv.is_file():
            raise RuntimeError("Input reference video path is not a file!")
        reference_timestr = rv.stem[-8:]
        self.prefix = rv.stem[:-8]
        reference_time = datetime.datetime.strptime(reference_timestr, "%H-%M-%S")
        self.start_time = reference_time + datetime.timedelta(seconds=frame_offset/self.fps)
        self.start_timestr = self.start_time.strftime("%H-%M-%S")
        self.original_vidpath = outpath.joinpath("originals", f"original-{self.prefix}{self.start_timestr}{rv.suffix}")
        self.trigger_vidpath = outpath.joinpath("triggers", f"trigger-{self.prefix}{self.start_timestr}{rv.suffix}")
        self.original_firstframepath = outpath.joinpath("originals", "first_frames", f"original-{self.prefix}{self.start_timestr}.jpg")
        self.trigger_firstframepath = outpath.joinpath("triggers", "first_frames", f"trigger-{self.prefix}{self.start_timestr}.jpg")
        self.meta_csvpath = outpath.joinpath("meta.csv")

        # open output subvideos
        self.original_cap = cv2.VideoWriter(
            str(self.original_vidpath),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height)
        )
        if not self.original_cap.isOpened(): raise RuntimeError("Couldn't create original video file")
        self.trigger_cap = cv2.VideoWriter(
            str(self.trigger_vidpath),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height)
        )
        if not self.original_cap.isOpened(): raise RuntimeError("Couldn't create trigger video file")

        # video metadata
        self.nframes = 0
        self.ninstances = 0
        self.ninstances_each = {n: 0 for n in instances_names}
        self.opened = True

    def write(self, original_frame, trigger_frame, instances):

        # save frame as image in case no frames have been written yet
        if not self.nframes:
            cv2.imwrite(str(self.original_firstframepath), original_frame)
            cv2.imwrite(str(self.trigger_firstframepath), trigger_frame)
        
        self.original_cap.write(original_frame)
        self.trigger_cap.write(trigger_frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            pass

        self.nframes += 1

        for k, v in instances.items():
            self.ninstances_each[k] += v
            self.ninstances += v
    
    def release(self):
        # release caps
        self.original_cap.release()
        self.trigger_cap.release()

        # delete video if too short
        if self.nframes < self.minframes:
            os.remove(str(self.original_vidpath))
            os.remove(str(self.trigger_vidpath))

        self.opened = False

        # write to metadatacsv
        row = [
            str(self.reference_vidpath),
            self.start_timestr,
            str(self.original_vidpath),
            str(self.original_firstframepath),
            str(self.trigger_vidpath),
            str(self.trigger_firstframepath),
            str(self.nframes),
            str(self.ninstances/self.nframes)
        ]
        header = ["reference video path", "start time", "original video path", "original first frame path", "trigger video path", "trigger first frame path", "nframes", "average ninstance per frame"]
        for k, v in self.ninstances_each.items():
            header.append(k)
            row.append(str(v))
        if self.meta_csvpath.exists() and self.meta_csvpath.is_file():
            with self.meta_csvpath.open(mode="a") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(row)
        else:
            with self.meta_csvpath.open(mode="w") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(header)
                csvwriter.writerow(row)

    def isOpened(self):
        return self.opened

def main(vidpath, model_detect_path, outdir, model_cls_path = None, save_instances = True, imgsz=864):

    vidname = ".".join(os.path.basename(vidpath).split(".")[:-1])

    # setting up output directory
    triggers_dir = os.path.join(outdir, "triggers")
    originals_dir = os.path.join(outdir, "originals")
    instances_dir = os.path.join(outdir, "instances")
    originals_first_frames_dir = os.path.join(originals_dir, "first_frames")
    triggers_first_frames_dir = os.path.join(triggers_dir, "first_frames")
    os.makedirs(triggers_dir, exist_ok=True)
    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(instances_dir, exist_ok=True)
    os.makedirs(originals_first_frames_dir, exist_ok=True)
    os.makedirs(triggers_first_frames_dir, exist_ok=True)

    cap = open_video(vidpath)
    #fps = cap.get(cv2.CAP_PROP_FPS)
    fps=10.0 # temporary fix
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #model, bird_class_idx = get_model(model_detect_path)
    model = ultralytics.YOLO(model_detect_path, task="detect")

    #model_cls = ultralytics.YOLO(model_cls_path) if model_cls_path else None
    model_cls = None

    print(f"""INPUT VIDEO: {vidpath}
    Resolution: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}x{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}
    FPS:        {fps}
OUTPUT DIRECTORY: {outdir}/{{originals,triggers}}
DETECTION MODEL: {model_detect_path}
CLASSIFICATION MODEL: {model_cls_path}

Beginning processing...
""")

    subvid = None
    success = cap.isOpened()
    nframe = 0
    WAITLIMIT=2*fps # wait two seconds before closing video
    wait_counter = WAITLIMIT
    batch_size = WAITLIMIT
    while success and nframe < total_frames:

        # read frames into batch
        frames = []
        while len(frames) < batch_size:
            success, frame = cap.read()
            if success: 
                frames.append(frame)
            else:
                frames.append(np.zeros_like(frames[0])) # loop logic ensures first frame is always present

        # inference on frames batch
        if frames: 
            batch_res = model(
                source=frames,
                #classes=[bird_class_idx],
                conf=0.46,
                iou=0.5,
                imgsz=imgsz,
                verbose=False,
                half=True
            )
        else:
            batch_res = []
      
        for yolo_res in batch_res:
            print(f"frame {nframe+1}/{total_frames}", end=" ")
            # save bool indicating whether a bird was detected
            #bird_detected = bird_class_idx in yolo_res.boxes.cls
            bird_detected = len(yolo_res.boxes.cls) > 0
            # update wait_counter if bird not detected
            wait_counter = 0 if bird_detected else wait_counter + 1
            # keep writing to video if wait limit hasn't been reached
            if wait_counter < WAITLIMIT:
                # subclassify
                if bird_detected:
                    #if model_cls: yolo_res = subclassify(yolo_res, model_cls)
                    instance_count = count_detections(yolo_res)
                    for k, v in instance_count.items():
                        print(f"{k}: {v}", end=" ")
                    print()
                    if save_instances: yolo_res.save_crop(instances_dir, add_seconds_to_timestring(vidname, nframe/fps))
                else:
                    instance_count = {}
                    print("no bird detected")
                # create video if a video isn't currently open
                if not subvid or not subvid.isOpened():
                    #subvid = detected_birb_vid(cap, vidpath, nframe, outdir, WAITLIMIT, list(model.names.values()) + list(model_cls.names.values()))
                    subvid = detected_birb_vid(cap, vidpath, nframe, outdir, WAITLIMIT, list(model.names.values()))
                
                subvid.write(yolo_res.orig_img, yolo_res.plot(conf=False), instance_count)
            else:
                print("no bird detected")
                # close the output video if it's open
                if subvid and subvid.isOpened():
                    subvid.release()
            nframe += 1
    
    cap.release() 
    if subvid and subvid.isOpened(): subvid.release() 
        
    # Closes all the frames 
    cv2.destroyAllWindows() 
       
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v")
    parser.add_argument("--model", "-m", default="yolov8n.pt")
    parser.add_argument("--output-directory", "-o", default=".")
    parser.add_argument("--cls-model", "-c", default=None)
    parser.add_argument("--save-instances", "-s", action="store_true")
    parser.add_argument("--imgsz", "-i", default=864)
    args = parser.parse_args()
    main(args.video, args.model, args.output_directory, args.cls_model, args.save_instances)
