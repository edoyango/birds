#!/usr/bin/env python3

import cv2
import os
import datetime
import sys
import torch
import copy
import ultralytics

def open_video(vname):

    cap = cv2.VideoCapture(vname)

    if (cap.isOpened() == False):  
        raise RuntimeError("Error reading video file")

    return cap

def get_model(model_path):

    model = ultralytics.YOLO(model_path)

    for k, v in model.names.items():
        if v == "bird":
            bird_class = k
    return model, bird_class

def create_output_video(vname, cap):

    out = cv2.VideoWriter(
        vname,
        cv2.VideoWriter_fourcc(*"MJPG"),
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
        species_names = res[0].names
        new_names = ret.names
        for k, v in species_names.items():
            new_names[k  + nexisting_names] = v
        ret.names = new_names

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

def main(vidpath, model_detect_path, outdir, model_cls_path = None):

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

    model, bird_class_idx = get_model(model_detect_path)

    model_cls = ultralytics.YOLO(model_cls_path) if model_cls_path else None

    print(f"""INPUT VIDEO: {vidpath}
    Resolution: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}x{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}
    FPS:        {fps}
OUTPUT DIRECTORY: {outdir}/{{originals,triggers}}
DETECTION MODEL: {model_detect_path}
CLASSIFICATION MODEL: {model_cls_path}

Beginning processing...
""")

    trigger_out_vid = None
    original_out_vid = None
    success = cap.isOpened()
    nframe = 0
    vid_idx = 0
    WAITLIMIT=2*fps # wait two seconds before closing video
    wait_counter = WAITLIMIT
    batch_size = WAITLIMIT
    while success:

        # read frames into batch
        frames = []
        while success and len(frames) < batch_size:
            success, frame = cap.read()
            if success: frames.append(frame)

        # inference on frames batch
        batch_res = model(
            source=frames,
            classes=[bird_class_idx],
            augment=True,
            conf=0.46,
            iou=0.5,
            imgsz=864,
            verbose=False
        )
      
        for yolo_res in batch_res:
            print(f"frame {nframe+1}/{total_frames}", end=" ")
            # save bool indicating whether a bird was detected
            bird_detected = bird_class_idx in yolo_res.boxes.cls
            # update wait_counter if bird not detected
            wait_counter = 0 if bird_detected else wait_counter + 1
            # keep writing to video if wait limit hasn't been reached
            if wait_counter < WAITLIMIT:
                # subclassify
                if bird_detected:
                    if model_cls: yolo_res = subclassify(yolo_res, model_cls)
                    for k, v in count_detections(yolo_res).items():
                        print(f"{k}: {v}", end=" ")
                    print()
                    yolo_res.save_crop(instances_dir, add_seconds_to_timestring(vidname, nframe/fps))
                else:
                    print("no bird detected")
                # get annotated frame from yolo results
                annotated_frame = yolo_res.plot(conf=False)
                original_frame = yolo_res.orig_img
                # create video if a video isn't currently open
                if not trigger_out_vid or not trigger_out_vid.isOpened():
                    trigger_subvid = create_new_fname(vidpath, nframe, fps, outdir, os.path.join("triggers", "trigger-"))
                    trigger_out_vid = create_output_video(trigger_subvid, cap)
                    cv2.imwrite(
                        os.path.join(triggers_first_frames_dir, os.path.basename(trigger_subvid).split(".")[0] + ".jpg"), 
                        annotated_frame
                    )
                if not original_out_vid or not original_out_vid.isOpened():
                    original_subvid = create_new_fname(vidpath, nframe, fps, outdir, os.path.join("originals", "original-"))
                    original_out_vid = create_output_video(original_subvid, cap)
                    cv2.imwrite(
                        os.path.join(originals_first_frames_dir, os.path.basename(original_subvid).split(".")[0] + ".jpg"), 
                        original_frame
                    )
                # write the annotated frame
                trigger_out_vid.write(annotated_frame)
                original_out_vid.write(original_frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
            else:
                print("no bird detected")
                # close the output video if it's open
                if trigger_out_vid and trigger_out_vid.isOpened():
                    # release video resources
                    trigger_out_vid.release()
                    # reopen video to check number of frames and get first frame
                    trigger_out_vid = open_video(trigger_subvid)
                    out_vid_nframes = trigger_out_vid.get(cv2.CAP_PROP_FRAME_COUNT)
                    trigger_out_vid.release()
                    # save first frame as jpg and delete video if video is too short
                    if out_vid_nframes < 2*WAITLIMIT:
                        os.remove(trigger_subvid)
                # do the same but for the original video
                if original_out_vid and original_out_vid.isOpened():
                    original_out_vid.release()
                    original_out_vid = open_video(original_subvid)
                    out_vid_nframes = original_out_vid.get(cv2.CAP_PROP_FRAME_COUNT)
                    original_out_vid.release()
                    if out_vid_nframes < 2*WAITLIMIT:
                        os.remove(original_subvid)
                    vid_idx += 1
            nframe += 1
    
    cap.release() 
    if trigger_out_vid: trigger_out_vid.release() 
    if original_out_vid: original_out_vid.release()
        
    # Closes all the frames 
    cv2.destroyAllWindows() 
       
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v")
    parser.add_argument("--model", "-m", default="yolov8n.pt")
    parser.add_argument("--output-directory", "-o", default=".")
    parser.add_argument("--cls-model", "-c", default=None)
    args = parser.parse_args()
    main(args.video, args.model, args.output_directory, args.cls_model)
