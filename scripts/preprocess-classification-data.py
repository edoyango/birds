import glob, os, argparse, pprint, shutil, random


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dir")

    args = parser.parse_args()

    traverse_subdirs(args.dir)

    split_data(args.dir)
