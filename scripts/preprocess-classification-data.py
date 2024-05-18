import glob, os, argparse, pprint, shutil, random
import cv2, albumentations


def remove_spaces(f) -> None:

    newname = os.path.basename(f).replace(" ", "_")
    newpath = os.path.join(os.path.dirname(f), newname)

    os.rename(f, newpath)


def preprocess_filenames(d) -> None:

    fs = (
        os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))
    )

    for f in fs:
        remove_spaces(f)


def traverse_subdirs(d) -> None:

    subdirs = (
        os.path.join(d, s) for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))
    )

    for s in subdirs:
        preprocess_filenames(s)


def split_data(d) -> None:

    classes = {}

    for s in (s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))):
        classes[s] = [
            os.path.join(d, s, f)
            for f in os.listdir(os.path.join(d, s))
            if os.path.isfile(os.path.join(d, s, f))
        ]
        random.shuffle(classes[s])

    dataset = {
        i: {k: [] for k in classes.keys()} for i in ("train", "validation", "test")
    }

    for c in classes.keys():
        ntotal = len(classes[c])
        ntrain = int(0.7 * ntotal)
        nval = int(0.2 * ntotal)
        ntest = ntotal - ntrain - nval
        dataset["train"][c] = [classes[c].pop() for _ in range(ntrain)]
        dataset["validation"][c] = [
            classes[c].pop() for _ in range(ntrain, ntrain + nval)
        ]
        dataset["test"][c] = [
            classes[c].pop() for _ in range(ntrain + nval, ntrain + nval + ntest)
        ]

    for i in ("train", "validation", "test"):
        for c in classes.keys():
            os.makedirs(os.path.join(d, i, c), exist_ok=True)
            for f in dataset[i][c]:
                fname = os.path.basename(f)
                shutil.move(f, os.path.join(d, i, c, fname))

    for c in classes.keys():
        os.rmdir(os.path.join(d, c))

    # augment the split data
    TRANSFORM = albumentations.Compose(
        [
            albumentations.HorizontalFlip(p=0.1),
            albumentations.VerticalFlip(p=0.1),
            albumentations.GaussianBlur(p=0.1),
            albumentations.GaussNoise(p=0.1),
            albumentations.RandomBrightnessContrast(p=0.1),
            albumentations.ColorJitter(p=0.1),
            albumentations.Spatter(p=0.1),
            albumentations.Rotate(limit=(-15,15), crop_border=True, p=0.1, always_apply=True)
        ]
    )
    # first, find which class hass the most instances
    maxinst = 0
    for c in classes.keys(): maxinst = max(maxinst, len(dataset["train"][c]))
    for c in classes.keys():
        ninst = len(dataset["train"][c])
        originals = os.listdir(os.path.join(d, "train", c))
        noriginals = len(originals)
        i = 0
        while ninst < maxinst:
            ii = i % noriginals
            f = os.path.join(d, "train", c, originals[ii])
            fsplit = f.split(".")
            img = cv2.imread(f)
            fname, fext = ".".join(fsplit[:-1]), fsplit[-1]
            try:
                img_t = TRANSFORM(image=img)["image"]
                cv2.imwrite(f"{fname}_augmented{i//noriginals}.jpg", img_t)
                ninst += 1
            except:
                pass
            i+=1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dir")

    args = parser.parse_args()

    traverse_subdirs(args.dir)

    split_data(args.dir)
