import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (300, 300)
BATCH_SIZE = 16

TEST_DIR = "dataset/test"
MODEL_PATH = "models/efficientnet_b3_final.keras"
CLASS_NAMES_PATH = "models/class_names.json"

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("\nClasses:", class_names)

# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("\nTest dataset loaded successfully.")

print("Number of test batches:", tf.data.experimental.cardinality(
    test_dataset
).numpy())

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\nTest Loss:")
print(f"{test_loss:.4f}")

print("\nTest Accuracy:")
print(f"{test_accuracy:.4f}")

# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("GENERATING TEST PREDICTIONS")
print("=" * 60)

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n")
print(cm)

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
    "models/test_confusion_matrix.png",
    dpi=300
)

plt.show()

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("TEST CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\n")
print(report)

# ============================================================
# SAVE REPORT
# ============================================================

report_path = "models/test_classification_report.txt"

with open(report_path, "w") as f:
    f.write(report)

print("\nTest classification report saved to:")
print(report_path)

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TESTING COMPLETE")
print("=" * 60)

print("\nTest Accuracy:")
print(f"{test_accuracy:.4f}")

print("\nTest Loss:")
print(f"{test_loss:.4f}")

print("\nGenerated:")
print("  - test_confusion_matrix.png")
print("  - test_classification_report.txt")

print("\nDONE")