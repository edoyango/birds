import glob, cv2, numpy as np, argparse, os, albumentations as A, random, shutil
import multiprocessing as mp, itertools


class albumentations_transformer:
    def __init__(self, img_path, boxes_path):
        self.map = {}
        self.classes = {}
        self.data = np.empty((0))
        self.count = 0
        self.orig_img = cv2.imread(img_path)
        self.img_path = img_path
        self.boxes_path = boxes_path

        try:
            self.boxes = np.loadtxt(boxes_path)
        except:
            self.boxes = np.empty(0)
        print(self.img_path, self.boxes_path)
        if len(self.boxes.shape) == 1:
            self.boxes = self.boxes[np.newaxis, :]
        if self.boxes.size > 0:
            self.boxes = self.boxes[:, [1, 2, 3, 4, 0]]
        
    def _apply_transform(self, transform):

        if self.boxes.size > 0:
            res = transform(image=self.orig_img, bboxes=self.boxes)
            return res["image"], np.array(res["bboxes"])
        else:
            res =  transform(image=self.orig_img)
            return res["image"], np.empty(0)

    def __call__(self, n, ignore_bg):
        idx = 0
        img_path_split = os.path.splitext(self.img_path)
        polygons_path_split = os.path.splitext(self.boxes_path)
        if self.boxes.size > 0:
            TRANSFORM = A.Compose(
                [
                    A.HorizontalFlip(p=0.1),
                    A.GaussianBlur(p=0.1),
                    A.GaussNoise(p=0.1),
                    A.RandomBrightnessContrast(p=0.1),
                    A.Affine(p=0.1),
                    A.Perspective( p=0.1),
                    A.ColorJitter(p=0.1),
                    A.Spatter(p=0.1),
                    A.Rotate(limit=(-15,15), crop_border=True, p=0.1, always_apply=True)
                ],
                bbox_params=A.BboxParams(format="yolo"),
            )
        else:
            TRANSFORM = A.Compose(
                [
                    A.HorizontalFlip(p=0.1),
                    A.GaussianBlur(p=0.1),
                    A.GaussNoise(p=0.1),
                    A.RandomBrightnessContrast(p=0.1),
                    A.Affine(p=0.1),
                    A.Perspective(p=0.1),
                    A.ColorJitter(p=0.1),
                    A.Spatter(p=0.1),
                    A.Rotate(limit=(-15,15), crop_border=True, p=0.1, always_apply=True)
                ],
            )
        outfiles = []
        if (not ignore_bg) or self.boxes.size > 0:
            for i in range(n):
                img_t, data_t = self._apply_transform(TRANSFORM)
                cv2.imwrite(
                    img_path_split[0] + f"_augmented{idx}" + img_path_split[1], img=img_t
                )
                try:
                    outdata = data_t[:, [4, 0, 1, 2, 3]]
                    fmt = ["%d", "%f", "%f", "%f", "%f"]
                except:
                    outdata = data_t
                    fmt = "%f"
                fpath = polygons_path_split[0] + f"_augmented{idx}" + polygons_path_split[1]
                np.savetxt(fpath, outdata, fmt=fmt)
                idx += 1
                outfiles.append(fpath)
            return outfiles

def run_transformer(imgpath, lblpath, nfiles, ignore_background):

    a = albumentations_transformer(imgpath, lblpath)
    return a(nfiles, ignore_background)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("nfiles", type=int)
    parser.add_argument("--nprocs", "-n", default=1, type=int)
    parser.add_argument("--ignore-background", "-i", action="store_true")
    args = parser.parse_args()

    images = sorted(glob.glob(os.path.join(args.dir, "train", "images", "*")))
    labels = sorted(glob.glob(os.path.join(args.dir, "train", "labels", "*")))

    with mp.Pool(processes=args.nprocs) as pool:
        files = pool.starmap(
            run_transformer, 
            zip(
                images, 
                labels, 
                itertools.repeat(args.nfiles), 
                itertools.repeat(args.ignore_background)
            )
        )
