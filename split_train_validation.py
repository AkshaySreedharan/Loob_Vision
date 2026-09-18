import os
import random
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"E:\AKSHAY\Efiicient_net_B33"

SOURCE_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_DIR = os.path.join(SOURCE_DIR, "train")
VAL_DIR = os.path.join(SOURCE_DIR, "validation")

CLASSES = ["c1", "c2", "c3"]

TRAIN_RATIO = 0.80
VAL_RATIO = 0.20

# Fixed seed = same split every time
RANDOM_SEED = 42

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)


# ============================================================
# FUNCTION TO GET IMAGES
# ============================================================

def get_images(folder):

    images = []

    if not os.path.exists(folder):
        return images

    for filename in os.listdir(folder):

        filepath = os.path.join(folder, filename)

        if os.path.isfile(filepath):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                images.append(filepath)

    return images


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for class_name in CLASSES:

    os.makedirs(
        os.path.join(TRAIN_DIR, class_name),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(VAL_DIR, class_name),
        exist_ok=True
    )


# ============================================================
# GET IMAGES FROM EACH CLASS
# ============================================================

class_images = {}

print("\n" + "=" * 60)
print("DATASET SPLIT")
print("=" * 60)

for class_name in CLASSES:

    source_class_dir = os.path.join(
        SOURCE_DIR,
        class_name
    )

    images = get_images(source_class_dir)

    class_images[class_name] = images

    print(f"{class_name}: {len(images)} images")


# ============================================================
# FIND SMALLEST CLASS
# ============================================================

min_count = min(
    len(class_images[class_name])
    for class_name in CLASSES
)

print("\nSmallest class:", min_count)


# ============================================================
# BALANCE ALL CLASSES
# ============================================================

print("\nBalancing classes...")

random.seed(RANDOM_SEED)

for class_name in CLASSES:

    random.shuffle(class_images[class_name])

    # Use exactly the same number of images
    # from every class
    class_images[class_name] = (
        class_images[class_name][:min_count]
    )


# ============================================================
# CALCULATE TRAIN / VALIDATION COUNTS
# ============================================================

train_count = int(min_count * TRAIN_RATIO)

val_count = min_count - train_count

print(f"\nImages per class: {min_count}")
print(f"Training per class: {train_count}")
print(f"Validation per class: {val_count}")


# ============================================================
# COPY IMAGES
# ============================================================

print("\nCopying images...")

for class_name in CLASSES:

    images = class_images[class_name]

    train_images = images[:train_count]

    val_images = images[train_count:]

    train_class_dir = os.path.join(
        TRAIN_DIR,
        class_name
    )

    val_class_dir = os.path.join(
        VAL_DIR,
        class_name
    )

    # ----------------------------
    # TRAIN
    # ----------------------------

    for image_path in train_images:

        filename = os.path.basename(image_path)

        destination = os.path.join(
            train_class_dir,
            filename
        )

        shutil.copy2(
            image_path,
            destination
        )

    # ----------------------------
    # VALIDATION
    # ----------------------------

    for image_path in val_images:

        filename = os.path.basename(image_path)

        destination = os.path.join(
            val_class_dir,
            filename
        )

        shutil.copy2(
            image_path,
            destination
        )

    print(
        f"{class_name}: "
        f"{len(train_images)} train | "
        f"{len(val_images)} validation"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SPLIT COMPLETED")
print("=" * 60)

print("\nTRAINING DATA:")

for class_name in CLASSES:

    folder = os.path.join(
        TRAIN_DIR,
        class_name
    )

    count = len(get_images(folder))

    print(f"{class_name}: {count}")


print("\nVALIDATION DATA:")

for class_name in CLASSES:

    folder = os.path.join(
        VAL_DIR,
        class_name
    )

    count = len(get_images(folder))

    print(f"{class_name}: {count}")


print("\nDataset structure:")
print(SOURCE_DIR)
print("├── train")
print("│   ├── c1")
print("│   ├── c2")
print("│   └── c3")
print("│")
print("└── validation")
print("    ├── c1")
print("    ├── c2")
print("    └── c3")

print("\nDone!")