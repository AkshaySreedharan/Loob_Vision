import os
import json
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (300, 300)
BATCH_SIZE = 16

TEST_DIR = "dataset/test"
MODEL_PATH = "models/efficientnet_b3_final.keras"
CLASS_NAMES_PATH = "models/class_names.json"

OUTPUT_CSV = "models/test_predictions.csv"
CONFUSION_MATRIX = "models/test_confusion_matrix_analysis.png"
REPORT_FILE = "models/test_classification_report_analysis.txt"

MISCLASSIFIED_DIR = "models/misclassified"


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("\nModel classes:", class_names)


# ============================================================
# LOAD TEST DATASET
# ============================================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("\nTest classes:", test_dataset.class_names)

# IMPORTANT: Verify class order
if test_dataset.class_names != class_names:
    raise ValueError(
        f"\nCLASS ORDER MISMATCH!"
        f"\nModel classes: {class_names}"
        f"\nTest classes: {test_dataset.class_names}"
    )

print("\nClass verification successful.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# GET TEST FILENAMES
# ============================================================

file_paths = test_dataset.file_paths


# ============================================================
# PREDICTIONS
# ============================================================

y_true = []
y_pred = []
confidences = []

print("\nRunning predictions...")

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)
    prediction_confidence = np.max(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)
    confidences.extend(prediction_confidence)


y_true = np.array(y_true)
y_pred = np.array(y_pred)
confidences = np.array(confidences)


# ============================================================
# ACCURACY
# ============================================================

accuracy = np.mean(y_true == y_pred)

print("\n======================================")
print("TEST ACCURACY")
print("======================================")

print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(class_names))
)

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("EfficientNet-B3 Test Confusion Matrix")

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX,
    dpi=300
)

plt.close()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(report)

with open(REPORT_FILE, "w") as f:
    f.write(report)


# ============================================================
# SAVE CSV
# ============================================================

print("\nSaving prediction CSV...")

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "actual_class",
        "predicted_class",
        "confidence",
        "correct"
    ])

    for i in range(len(file_paths)):

        actual = class_names[y_true[i]]
        predicted = class_names[y_pred[i]]

        writer.writerow([
            file_paths[i],
            actual,
            predicted,
            f"{confidences[i]:.4f}",
            actual == predicted
        ])


# ============================================================
# CREATE MISCLASSIFIED IMAGE FOLDERS
# ============================================================

print("\nSaving misclassified images...")

for actual_idx in range(len(class_names)):

    for predicted_idx in range(len(class_names)):

        if actual_idx == predicted_idx:
            continue

        actual_name = class_names[actual_idx]
        predicted_name = class_names[predicted_idx]

        folder = os.path.join(
            MISCLASSIFIED_DIR,
            f"{actual_name}_as_{predicted_name}"
        )

        os.makedirs(folder, exist_ok=True)


# ============================================================
# COPY MISCLASSIFIED IMAGES
# ============================================================

import shutil

for i in range(len(file_paths)):

    if y_true[i] != y_pred[i]:

        actual_name = class_names[y_true[i]]
        predicted_name = class_names[y_pred[i]]

        destination_folder = os.path.join(
            MISCLASSIFIED_DIR,
            f"{actual_name}_as_{predicted_name}"
        )

        filename = os.path.basename(file_paths[i])

        destination = os.path.join(
            destination_folder,
            filename
        )

        shutil.copy2(
            file_paths[i],
            destination
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n======================================")
print("ANALYSIS COMPLETE")
print("======================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nFiles created:")

print(f"1. {OUTPUT_CSV}")
print(f"2. {CONFUSION_MATRIX}")
print(f"3. {REPORT_FILE}")
print(f"4. {MISCLASSIFIED_DIR}")

print("\nMost important folder to inspect:")

print(
    os.path.join(
        MISCLASSIFIED_DIR,
        "c3_as_c2"
    )
)

print("\n======================================")