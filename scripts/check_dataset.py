import argparse
import os
import sys

TEXTURE_CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]
OBJECT_CATEGORIES = [
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
]
ALL_CATEGORIES = TEXTURE_CATEGORIES + OBJECT_CATEGORIES

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")


def count_images(folder):
    if not os.path.isdir(folder):
        return -1
    return sum(1 for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))


def check_category(cat_root):
    problems = []
    n_train = count_images(os.path.join(cat_root, "train", "good"))
    n_good = count_images(os.path.join(cat_root, "test", "good"))
    if n_train <= 0:
        problems.append("train/good missing or empty")
    if n_good <= 0:
        problems.append("test/good missing or empty")

    test_dir = os.path.join(cat_root, "test")
    defects = sorted(
        d for d in (os.listdir(test_dir) if os.path.isdir(test_dir) else [])
        if os.path.isdir(os.path.join(test_dir, d)) and d != "good"
    )
    if not defects:
        problems.append("no defect folders under test/")
    n_defect = 0
    for d in defects:
        n_img = count_images(os.path.join(cat_root, "test", d))
        n_mask = count_images(os.path.join(cat_root, "ground_truth", d))
        n_defect += max(n_img, 0)
        if n_img <= 0:
            problems.append(f"test/{d} empty")
        if n_mask != n_img:
            problems.append(f"ground_truth/{d}: {n_mask} masks vs {n_img} images")
    return n_train, n_good, n_defect, defects, problems


def main():
    parser = argparse.ArgumentParser(description='Verify the MVTec AD folder layout before launching any training.')
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help="MVTec AD root: <data-root>/<category>/{train,test,ground_truth}")
    args = parser.parse_args()

    ok = True
    print(f"data root: {args.data_root}\n")
    print(f"{'category':<12} {'type':<8} {'train':>6} {'good':>6} {'defect':>7} "
          f"{'eval':>6}  defect classes")
    for category in ALL_CATEGORIES:
        ctype = "texture" if category in TEXTURE_CATEGORIES else "object"
        cat_root = os.path.join(args.data_root, category)
        if not os.path.isdir(cat_root):
            print(f"{category:<12} MISSING ({cat_root})")
            ok = False
            continue
        n_train, n_good, n_defect, defects, problems = check_category(cat_root)
        print(f"{category:<12} {ctype:<8} {n_train:>6} {n_good:>6} "
              f"{n_defect:>7} {n_good + n_defect:>6}  {', '.join(defects)}")
        for p in problems:
            print(f"{'':<12} !! {p}")
            ok = False

    if not ok:
        print("\nDataset check FAILED -- fix the issues above before training.")
        sys.exit(1)
    print("\nDataset check passed: all 15 categories look complete.")


if __name__ == "__main__":
    main()
