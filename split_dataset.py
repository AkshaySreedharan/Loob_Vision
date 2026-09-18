import os
import shutil

# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = r"E:\AKSHAY\Efiicient_net_B33\test"
DEST_DIR = r"E:\AKSHAY\Efiicient_net_B33\test_split"


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_TO_CLASS = {
    "fresh_": "c1",
    "semi_": "c2",
    "fully_": "c3"
}


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# CREATE DESTINATION FOLDERS
# ============================================================

for class_name in ["c1", "c2", "c3"]:
    class_path = os.path.join(DEST_DIR, class_name)
    os.makedirs(class_path, exist_ok=True)


# ============================================================
# COUNTERS
# ============================================================

counts = {
    "c1": 0,
    "c2": 0,
    "c3": 0,
    "unknown": 0
}


# ============================================================
# PROCESS DATASET
# ============================================================

print("=" * 60)
print("DATASET SPLITTING")
print("=" * 60)

for filename in os.listdir(SOURCE_DIR):

    source_path = os.path.join(SOURCE_DIR, filename)

    # Ignore folders
    if not os.path.isfile(source_path):
        continue

    # Check extension
    extension = os.path.splitext(filename)[1].lower()

    if extension not in IMAGE_EXTENSIONS:
        continue

    # Lowercase filename for comparison
    filename_lower = filename.lower()

    destination_class = None

    # Find label
    for label, class_name in LABEL_TO_CLASS.items():

        if filename_lower.startswith(label):
            destination_class = class_name
            break

    # --------------------------------------------------------
    # Unknown label
    # --------------------------------------------------------

    if destination_class is None:

        print(f"[UNKNOWN] {filename}")

        counts["unknown"] += 1

        continue

    # --------------------------------------------------------
    # Destination
    # --------------------------------------------------------

    destination_dir = os.path.join(
        DEST_DIR,
        destination_class
    )

    destination_path = os.path.join(
        destination_dir,
        filename
    )

    # --------------------------------------------------------
    # Avoid overwriting
    # --------------------------------------------------------

    if os.path.exists(destination_path):

        print(f"[SKIPPED - EXISTS] {filename}")

        continue

    # --------------------------------------------------------
    # Copy image
    # --------------------------------------------------------

    shutil.copy2(
        source_path,
        destination_path
    )

    counts[destination_class] += 1

    print(
        f"[{destination_class}] {filename}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("DATASET SPLIT COMPLETE")
print("=" * 60)

print(f"Fresh          → c1 : {counts['c1']}")
print(f"Semi Degraded  → c2 : {counts['c2']}")
print(f"Fully Degraded → c3 : {counts['c3']}")
print(f"Unknown              : {counts['unknown']}")

print("=" * 60)

total = (
    counts["c1"]
    + counts["c2"]
    + counts["c3"]
)

print(f"Total classified images : {total}")
print("=" * 60)