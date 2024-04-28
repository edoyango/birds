import cv2
import os
import datetime
import sys

def open_video(vname):

    cap = cv2.VideoCapture(vname)

    if (cap.isOpened() == False):  
        raise RuntimeError("Error reading video file")

    return cap

def get_model(model_path):

    import ultralytics

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
        
def create_new_fname(vidpath, frame_idx, fps, outdir=None, prefix=""):
    vid_dir = os.path.dirname(vidpath)
    split_vidname = os.path.basename(vidpath).split(".")
    ext = split_vidname[-1]
    time_str = ".".join(split_vidname[:-1])
    time_obj = datetime.datetime.strptime(time_str, "%H-%M-%S")
    time_diff = datetime.timedelta(seconds=frame_idx/fps)
    new_time = time_obj + time_diff
    outfile = f"{prefix}{new_time.strftime('%H-%M-%S')}.{ext}"
    if outdir:
        outpath = os.path.join(outdir, outfile)
    else:
        outpath = os.path.join(vid_dir, outfile)
    return outpath

def main(vidpath, model, outdir):

    # setting up output directory
    os.makedirs(os.path.join(outdir, "triggers"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "originals"), exist_ok=True)

    cap = open_video(vidpath)
    fps = cap.get(cv2.CAP_PROP_FPS)

    model, bird_class_idx = get_model(model)

    trigger_out_vid = None
    original_out_vid = None
    success = True
    nframe = 0
    vid_idx = 0
    WAITLIMIT=2*fps # wait two seconds before closing video
    wait_counter = WAITLIMIT
    while success:
        if nframe % 10 == 0:
            print(f"frame {nframe}")
        success, frame = cap.read() 
      
        if success:
            # perform inference
            yolo_res = model(
                source=frame,
                classes=[bird_class_idx],
                augment=True,
                conf=0.516,
                iou=0.5,
                imgsz=864,
            )[0]
            # save bool indicating whether a bird was detected
            bird_detected = bird_class_idx in yolo_res.boxes.cls
            # update wait_counter if bird not detected
            wait_counter = 0 if bird_detected else wait_counter + 1
            # keep writing to video if wait limit hasn't been reached
            if wait_counter < WAITLIMIT:
                # get annotated frame from yolo results
                annotated_frame = yolo_res.plot(conf=False)
                original_frame = yolo_res.orig_img
                # create video if a video isn't currently open
                if not trigger_out_vid or not trigger_out_vid.isOpened():
                    trigger_subvid = create_new_fname(vidpath, nframe, fps, outdir, os.path.join("triggers", "trigger-"))
                    trigger_out_vid = create_output_video(trigger_subvid, cap)
                if not original_out_vid or not original_out_vid.isOpened():
                    original_subvid = create_new_fname(vidpath, nframe, fps, outdir, os.path.join("originals", "original-"))
                    original_out_vid = create_output_video(original_subvid, cap)
                # write the annotated frame
                trigger_out_vid.write(annotated_frame)
                original_out_vid.write(original_frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
            else:
                # close the output video if it's open
                if trigger_out_vid and trigger_out_vid.isOpened():
                    # release video resources
                    trigger_out_vid.release()
                    # reopen video to check number of frames and get first frame
                    trigger_out_vid = open_video(trigger_subvid)
                    out_vid_nframes = trigger_out_vid.get(cv2.CAP_PROP_FRAME_COUNT)
                    subvid_success, first_frame = trigger_out_vid.read()
                    trigger_out_vid.release()
                    # save first frame as jpg and delete video if video is too short
                    if out_vid_nframes < 2*WAITLIMIT and subvid_success:
                        cv2.imwrite(".".join(trigger_subvid.split(".")[:-1]) + ".jpg", first_frame)
                        os.remove(trigger_subvid)
                    elif not subvid_success:
                        raise RuntimeError(f"Problem with reading {trigger_subvid}!")
                # do the same but for the original video
                if original_out_vid and original_out_vid.isOpened():
                    original_out_vid.release()
                    original_out_vid = open_video(original_subvid)
                    out_vid_nframes = original_out_vid.get(cv2.CAP_PROP_FRAME_COUNT)
                    subvid_success, first_frame = original_out_vid.read()
                    original_out_vid.release()
                    if out_vid_nframes < 2*WAITLIMIT and subvid_success:
                        cv2.imwrite(".".join(original_subvid.split(".")[:-1]) + ".jpg", first_frame)
                        os.remove(original_subvid)
                    elif not subvid_success:
                        raise RuntimeError(f"Problem with reading {original_subvid}!")
                    vid_idx += 1
            nframe += 1
    
    cap.release() 
    out_vid.release() 
        
    # Closes all the frames 
    cv2.destroyAllWindows() 
       
    print("The video was successfully saved") 

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v")
    parser.add_argument("--model", "-m", default="yolov8n.pt")
    parser.add_argument("--output-directory", "-o", default=".")
    args = parser.parse_args()
    main(args.video, args.model, args.output_directory)
