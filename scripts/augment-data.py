import glob, cv2, numpy as np, argparse, os, albumentations as A, random, shutil

TRANSFORM = A.Compose(
    [
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.GaussianBlur(p=0.2),
        A.GaussNoise(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        # # A.Affine(p=0.2),
        # A.Perspective(always_apply=True, p=0.2),
        A.ColorJitter(p=0.2),
        A.Spatter(p=0.2),
        A.Downscale(interpolation=cv2.INTER_LINEAR, p=0.2),
    ],
    keypoint_params=A.KeypointParams(format="xy"),
)


class albumentations_transformer:
    def __init__(self, img_path, polygons_path):
        self.map = {}
        self.classes = {}
        self.data = np.empty((0))
        self.count = 0
        self.orig_img = cv2.imread(img_path)
        self.img_path = img_path
        self.polygons_path = polygons_path

        with open(polygons_path, "r") as f:
            for line in f:
                line_data = np.fromstring(line, sep=" ")
                start = len(self.data)
                self.map[self.count] = (start, start + len(line_data) - 1)
                self.classes[self.count] = int(line_data[0])
                self.data = np.concatenate((self.data, line_data[1:]), axis=0)
                self.count += 1

    def _albumentations(self):
        return np.minimum(
            self.data.reshape((-1, 2)) * np.array((1920, 1080)), np.array((1919, 1079))
        )

    def _apply_transform(self, transform):
        res = transform(image=self.orig_img, keypoints=self._albumentations())
        if len(res["keypoints"]) > 0:
            return res["image"], (
                np.array(res["keypoints"]) / np.array((1920, 1080))
            ).reshape(-1)
        else:
            return res["image"], np.empty((0))

    def generate_images(self, transform, n):
        idx = 0
        img_path_split = os.path.splitext(self.img_path)
        polygons_path_split = os.path.splitext(self.polygons_path)
        for i in range(n):
            img_t, data_t = self._apply_transform(transform)
            cv2.imwrite(
                img_path_split[0] + f"_augmented{idx}" + img_path_split[1], img_t
            )
            with open(
                polygons_path_split[0] + f"_augmented{idx}" + polygons_path_split[1],
                "w",
            ) as f:
                for k, v in self.map.items():
                    line = str(self.classes[k])
                    for j in data_t[v[0] : v[1]]:
                        line += f" {j}"
                    line += "\n"
                    f.write(line)
            idx += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("nfiles", type=int)
    args = parser.parse_args()

    images = sorted(glob.glob(os.path.join(args.dir, "*", "images", "*")))
    labels = sorted(glob.glob(os.path.join(args.dir, "*", "labels", "*")))

    for imgpath, lblpath in zip(images, labels):

        a = albumentations_transformer(imgpath, lblpath)
        a.generate_images(TRANSFORM, args.nfiles)

    images = sorted(glob.glob(os.path.join(args.dir, "*", "images", "*")))
    labels = sorted(glob.glob(os.path.join(args.dir, "*", "labels", "*")))

    data = []
    for i, l in zip(images, labels):
        data.append((i, l))

    ntotal = len(data)
    ntest = max(1, int(0.1 * ntotal))
    nval = max(1, int(0.2 * ntotal))
    ntrain = len(images) - ntest - nval
    if ntest < 1 or nval < 1 or ntrain < 1:
        raise ("Not enough samples")

    random.shuffle(data)

    train_data = data[0:ntrain]
    val_data = data[ntrain : ntrain + nval]
    test_data = data[ntrain + nval : ntrain + nval + ntest]

    train_dir = os.path.join(args.dir, "train")
    val_dir = os.path.join(args.dir, "valid")
    test_dir = os.path.join(args.dir, "test")

    for img, lbl in train_data:
        try:
            shutil.move(img, os.path.join(train_dir, "images"))
        except:
            pass
        try:
            shutil.move(lbl, os.path.join(train_dir, "labels"))
        except:
            pass

    for img, lbl in val_data:
        try:
            shutil.move(img, os.path.join(val_dir, "images"))
        except:
            pass
        try:
            shutil.move(lbl, os.path.join(val_dir, "labels"))
        except:
            pass

    for img, lbl in test_data:
        try:
            shutil.move(img, os.path.join(test_dir, "images"))
        except:
            pass
        try:
            shutil.move(lbl, os.path.join(test_dir, "labels"))
        except:
            pass
